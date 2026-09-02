import sys
from pathlib import Path
import importlib.util

import time
from collections import Counter, defaultdict
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

# Shared base class for Sango backends
class Backend(ABC):
    def __init__(self, net, debug=False, verbose=False):
        self.net = net
        self.timesteps = None

        # Diagnostic flags
        self.debug = debug
        self.verbose = verbose

        # Global graph data
        self.ref_graph = None      # Reference graph
        self.edge_order = 'source' # Source-major order by default
        self.num_nodes = 0
        self.node_index = None   # {node_name: global_index}
        self.node_data = None    # [dict per node]
        self.edge_data = None    # [dict-of-dicts per node] (by edge_order)
        self.input_data = None   # {node_name: [times]}

        # Grouped graph data (by shared parameters)
        self.group_index = None  # {group_name: insertion_order}
        self.local_index = None  # [int per node]
        self.group_count = None  # {group_name: count}
        self.group_total = None  # [int per group]
        self.group_offset = None # [int per group] (prefix sum)
        self.node_groups = None  # {model: {param_tuple: variation_index}}
        self.edge_groups = None  # {(model, src_grp, tgt_grp): {param_tuple: variation_index}}
        self.group_models = None # {group_name: model}
        self.group_params = None # {group_name: {param_keys: params}}
        self.node_map = None     # {node_name: group-sorted index}

        # Input
        self.input_spec = None

        # Output
        self.output_spec = None
        self.spike_list = None

        # Recording
        self.record_spec = None

        # Timing
        self.compile_time = None
        self.run_time = None

        # Registry
        self.model_registry = self.import_registry()

    # Dynamically import registry files (from registry/*.py)
    def import_registry(self):
        registry = dict()
        # Walk from the subclass file location
        subclass_file = Path(sys.modules[type(self).__module__].__file__).resolve()
        registry_dir = subclass_file.parent / 'registry'
        if not registry_dir.is_dir():
            return registry
        parent_str = str(registry_dir.parent)
        sys.path.insert(0, parent_str)
        for file_path in sorted(registry_dir.glob("*.py")):
            if file_path.name == "__init__.py":
                continue
            module_name = f"registry.{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            registry.update(getattr(module, 'model_registry', {}))
        sys.path.pop(0)
        return registry

    # Convert between Sango model and backend model
    def rekey_model(self, data):
        model_entry = self.model_registry[data['model']]
        for key, value in model_entry.get('state', {}).items():
            if value['dsl'] is not None:
                data[key] = data.pop(value['dsl'])
            else:
                data[key] = value['default']
        return data

    # Convert between Sango model and backend model shared parameters
    def rekey_param(self, data):
        model_entry = self.model_registry[data['model']]
        keys = []
        param = []
        for key, value in model_entry.get('param', {}).items():
            keys.append(key)
            if value['dsl'] is not None:
                param.append(data.pop(value['dsl']))
            else:
                param.append(value['default'])
        return tuple(keys), tuple(param)

    # Graph processing (with parameter-aware grouping)
    def process_graph(self):
        # Get reference graph from network
        self.ref_graph = self.net._topology.to_nx()
        self.num_nodes = self.ref_graph.number_of_nodes()

        # Set up containers
        self.node_index = dict()
        self.node_data = [dict() for _ in range(self.num_nodes)]
        self.edge_data = [dict() for _ in range(self.num_nodes)]
        self.input_data = dict()
        self.group_count = Counter()
        self.local_index = [0 for _ in range(self.num_nodes)]
        self.node_groups = defaultdict(dict)
        self.edge_groups = defaultdict(dict)
        self.group_models = dict()
        self.group_params = dict()
        
        # Nodes
        for n, (node, data) in enumerate(self.ref_graph.nodes(data=True)):
            self.node_index[node] = n
            self.node_data[n] = self.rekey_model(data)
            param_keys, param_tuple = self.rekey_param(data)
            model_name = self.node_data[n]['model']

            # Parameter-aware variation index
            if param_tuple not in self.node_groups[model_name]:
                self.node_groups[model_name][param_tuple] = len(self.node_groups[model_name])
            variation = self.node_groups[model_name][param_tuple]
            group_name = model_name if variation == 0 else f"{model_name}_{variation}"

            self.node_data[n]['group_name'] = group_name
            self.group_count.update([group_name])
            self.local_index[n] = self.group_count[group_name] - 1

            # Parameter dictionary
            if group_name not in self.group_models:
                self.group_models[group_name] = model_name
                self.group_params[group_name] = dict(zip(param_keys, param_tuple))

            # Stash any input data
            if self.model_registry[model_name]['graph_type'] == 'input':
                self.input_data[node] = data['times']

        # Group ordering (inputs first)
        input_groups = Counter()
        other_groups = Counter()
        for model_name, value in self.node_groups.items():
            for param, variation in value.items():
                group_name = model_name if variation == 0 else f"{model_name}_{variation}"
                if self.model_registry[model_name]['graph_type'] == 'input':
                    input_groups[group_name] = self.group_count[group_name]
                else:
                    other_groups[group_name] = self.group_count[group_name]
        self.group_count = input_groups + other_groups

        # Generate sorted node map
        self.group_index = {key: idx for idx, key in enumerate(self.group_count.keys())}
        self.group_total = [self.group_count[g] for g in self.group_index]
        self.group_offset = [0] + [sum(self.group_total[:i+1]) for i in range(len(self.group_total))]
        self.node_map = {
            key: (self.group_offset[self.group_index[self.node_data[n]['group_name']]]
                  + self.local_index[n])
            for key, n in self.node_index.items()
        }

        # Edges
        for source, target, data in self.ref_graph.edges(data=True):
            s = self.node_index[source]
            t = self.node_index[target]
            if self.edge_order == 'source':
                self.edge_data[s][t] = self.rekey_model(data)
                model_name = self.edge_data[s][t]['model']
            elif self.edge_order == 'target':
                self.edge_data[t][s] = self.rekey_model(data)
                model_name = self.edge_data[t][s]['model']
            param_keys, param_tuple = self.rekey_param(data)
            src_group = self.node_data[s]['group_name']
            tgt_group = self.node_data[t]['group_name']
            edge_tuple = (model_name, src_group, tgt_group)
            
            # Parameter-aware variation index
            if param_tuple not in self.edge_groups[edge_tuple]:
                self.edge_groups[edge_tuple][param_tuple] = len(self.edge_groups[edge_tuple])
            variation = self.edge_groups[edge_tuple][param_tuple]
            base_name = f"{model_name}_{src_group}_{tgt_group}"
            group_name = base_name if variation == 0 else f"{base_name}__{variation}"

            if self.edge_order == 'source':
                self.edge_data[s][t]['group_name'] = group_name
            elif self.edge_order == 'target':
                self.edge_data[t][s]['group_name'] = group_name
            
            # Parameter dictionary
            if group_name not in self.group_models:
                self.group_models[group_name] = model_name
                self.group_params[group_name] = dict(zip(param_keys, param_tuple))

    # Compile to backend
    def compile(self, **kwargs):
        start_time = time.perf_counter()
        
        # Process graph structure
        self.process_graph()

        # Pass to backend compile method
        self.to_backend(**kwargs)
        
        # Performance timing
        end_time = time.perf_counter()
        self.compile_time = end_time - start_time
        if self.verbose:
            print(f"Compile time: {self.compile_time}")

    # Run simulation backend
    def run(self, timesteps=1, **kwargs):
        start_time = time.perf_counter()
        
        # Pass to backend run method
        self.run_backend(timesteps, **kwargs)

        # Performance timing
        end_time = time.perf_counter()
        self.run_time = end_time - start_time
        if self.verbose:
            print(f"Run time: {self.run_time}")

    # Backend compile method
    @abstractmethod
    def to_backend(self, **kwargs):
        ...

    # Backend run method
    @abstractmethod
    def run_backend(self, timesteps=1, **kwargs):
        ...

    # Read spike output into spike event list (in node_map order)
    @abstractmethod
    def read_spikes(self):
        ...

    # Return spikes as event list
    def get_spikes(self, update=False):
        if update or self.spike_list is None:
            return self.read_spikes()
        return self.spike_list

    # Plotting as event plot
    def plot_spikes(self, figsize=(8, 6), linelengths=0.8, linewidths=1.0,
                    color_dict=None, tick_names=False):
        self.get_spikes()

        if color_dict is None:
            color_dict = {'LIF': 'C0', 'IN': 'C1'}

        plt.figure(figsize=figsize)

        # Per-neuron color list from group_count
        event_color = []
        for name, count in self.group_count.items():
            if name not in color_dict:
                color_dict[name] = f"C{len(color_dict) % 10}"
            event_color.extend([color_dict[name]] * count)

        # Legend lines
        for key in color_dict:
            plt.plot(0, 0, '-', color=color_dict[key], linewidth=2.0)

        plt.eventplot(self.spike_list, colors=event_color, lineoffsets=1,
                      linelengths=linelengths, linewidths=linewidths)

        if tick_names:
            plt.yticks(list(self.node_map.values()), list(self.node_map.keys()))

        plt.title('Spike Raster')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron (index)')
        plt.tight_layout()
        plt.legend(color_dict.keys())

