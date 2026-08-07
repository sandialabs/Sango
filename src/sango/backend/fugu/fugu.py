import os
import sys
from pathlib import Path
import importlib.util

try:
	import fugu
except ImportError:
	fugu = None
if fugu is not None:
    from fugu import Scaffold, Brick
    from fugu.scaffold import PortData, ChannelData
    from fugu.backends import snn_Backend

import time
from collections import Counter, defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SimFugu:
    def __init__(self, dsl_net):
        if fugu is None:
            raise ImportError("fugu package is required for SimFugu")
        
        self.dsl_net = dsl_net
        self.timesteps = None
        
        self.node_map = None
        self.spike_list = None
    
        self.model_registry = self.import_registry()
    
    # Dynamically import the model registry files
    def import_registry(self):
        registry = dict()
        registry_dir = Path(__file__).resolve().parent / 'registry'
        sys.path.insert(0, str(registry_dir.parent))
        for file_path in registry_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
            module_name = f"registry.{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            registry.update(getattr(module, 'model_registry', {}))
        sys.path.pop(0)
        return registry
    
    # Convert between dsl model to Fugu model
    def rekey_model(self, data):
        for key, value in self.model_registry[data['model']]['param'].items():
            if value['dsl'] is not None:
                if value['dsl'] == 'delay':
                    data[key] = int(data.pop(value['dsl']))
                else:
                    data[key] = data.pop(value['dsl'])
            else:
                data[key] = value['default']
        return data
        
    def compile(self, backend='snn', backend_args={'record': 'all'}, debug=False):
        self.debug = debug
        if backend == 'snn':
            self.fugu_backend = snn_Backend()
        else:
            raise NotImplementedError(f"Backend '{backend}' is not supported.")
        self.backend_args = backend_args
        if self.debug:
            self.backend_args['debug_mode'] = True
        
        # Convert network to Fugu
        start_time = time.perf_counter()
        self.to_fugu()
        end_time = time.perf_counter()
        self.compile_time = end_time - start_time
        if self.debug:
            print(f"Compile time: {self.compile_time}")
    
    # Run the network
    def run(self, timesteps=10.0, verbose=False):
        self.timesteps = int(timesteps)
        self.verbose = verbose

        # Run network
        start_time = time.perf_counter()
        self.fugu_result = self.fugu_backend.run(n_steps=timesteps)
        end_time = time.perf_counter()
        self.run_time = end_time - start_time
        if self.verbose:
            print(f"Run time: {self.run_time}")
    
    # Translates a Sango network into a Fugu Scaffold
    def to_fugu(self):
        # Get a flattened graph object
        self.dsl_graph = self.dsl_net._topology.to_nx()

        # Convert networkx to scaffold and graph
        self.scaffold = self.to_scaffold(self.dsl_graph)

        # Compile scaffold in Fugu backend
        self.fugu_backend.compile(self.scaffold, self.backend_args)

    # Dummy brick object for Fugu scaffold
    class Dummy(Brick):
        def __init__(self, name='dummy'):
            super().__init__(name)
            self.vector = []
            self.index = 0
            self.is_built = True
        # This object is mainly used for managing Fugu inputs
        def __iter__(self):
            self.index = 0
            return iter(self.vector)
        def __next__(self):
            if self.index < len(self.data):
                result = self.vector[self.index]
                self.index += 1
                return result
            else:
                raise StopIteration

    # Convert Sango network to Fugu Scaffold
    def to_scaffold(self, net):
        # Set up a dummy scaffold in Fugu
        scaffold = Scaffold()
        scaffold_circuit = nx.DiGraph()
        # Input and main Sango bricks
        input_brick = {'tag': 'input',
                       'name': 'InputBrick',
                       'brick': self.Dummy('input'),
                       'layer': 'input',
                       'ports': {'output': None},
                       'is_built': True}
        sango_brick = {'tag': 'sango',
                       'name': 'SangoBrick',
                       'brick': self.Dummy('sango'),
                       'layer': 'output',
                       'ports': {'input': None},
                       'is_built': True}
        # Connect dummy circuit
        scaffold_circuit.add_node(0, **input_brick)
        scaffold_circuit.add_node(1, **sango_brick)
        scaffold_circuit.add_edge(0, 1, bind={'input': 'output'})
        scaffold.tag_to_name = {'input': 'InputBrick', 'sango': 'SangoBrick'}
        scaffold.brick_to_number = {'InputBrick': 0, 'SangoBrick': 1}

        # Set up underlying graph
        scaffold_graph = nx.DiGraph()
        self.num_nodes = self.dsl_graph.number_of_nodes()
        self.node_data = [dict() for _ in range(self.num_nodes)]
        self.node_index = dict()
        self.group_count = Counter()
        self.local_index = [0 for _ in range(self.num_nodes)]
        self.group_count['IN'] = 0
        self.spike_input = dict()

        # Global node data
        for n, (node, data) in enumerate(self.dsl_graph.nodes(data=True)):
            self.node_index[node] = n
            self.node_data[n] = self.rekey_model(data)
            self.group_count.update([self.node_data[n]['model']])
            self.local_index[n] = self.group_count[self.node_data[n]['model']] - 1
            # Add spike times for inputs
            if self.model_registry[self.node_data[n]['model']]['node_class'] == 'InputNeuron':
                self.spike_input[node]=data['times']
        
        # Remove the input model (IN) if counts are zero
        if self.group_count['IN'] == 0:
            del self.group_count['IN']
        
        # Get the model insertion order of our counter, and organize node map by group
        self.group_index = {key: index for index, key in enumerate(self.group_count.keys())}
        self.group_sorted_count = [self.group_count[group] for group in self.group_index]
        self.group_offset = [0] + [sum(self.group_sorted_count[:i+1]) for i in range(len(self.group_sorted_count))]
        self.node_map = {key: self.group_offset[self.group_index[self.node_data[n]['model']]] + self.local_index[n]
                         for key, n in self.node_index.items()}

        # Attach spike times to input brick
        spike_times = defaultdict(list)
        for n, (neuron, times) in enumerate(self.spike_input.items()):
            for t in times:
                spike_times[t].append(neuron)
        spike_vector = [spike_times[key] for key in range(max(spike_times.keys())+1)]
        scaffold_circuit.nodes[0]['brick'].vector = spike_vector
        scaffold_circuit.nodes[0]['ports']['output'] = PortData(spec=None,
            channels = {'data': ChannelData(spec=None, neurons = list(self.spike_input.keys()))})
        scaffold_circuit.nodes[1]['ports']['input'] = PortData(spec=None,
            channels = {'data': ChannelData(spec=None, neurons = [])})
        scaffold.circuit = scaffold_circuit

        # Insert nodes by group
        for n, (node, nidx) in enumerate(self.node_map.items()):
            node_data = self.node_data[n]
            if node_data['model'] == 'IN':
                scaffold_graph.add_node(node, neuron_number=nidx,
                                        brick='InputBrick', **node_data)
            else:
                scaffold_graph.add_node(node, neuron_number=nidx,
                                        brick='SangoBrick', **node_data)
        
        # Global edge data
        for source, target, data in self.dsl_graph.edges(data=True):
            edge_data = self.rekey_model(data)
            scaffold_graph.add_edge(source, target, **edge_data)
        
        # Attach the built graph to the scaffold and mark as built
        scaffold.graph = scaffold_graph
        scaffold.is_built = True

        return scaffold

    # Collect any output from the simulation
    def read_spikes(self):
        # Reading in Fugu's dataframe results
        spike_dict = defaultdict(list)
        for spike in self.fugu_result.itertuples(index=False):
            spike_dict[int(spike.neuron_number)].append(spike.time)
        self.spike_list = [spike_dict[key] for key in range(len(self.node_map))]
        return self.spike_list
        
    # Return spikes as event list
    def get_spikes(self):
        if self.spike_list is None:
            return self.read_spikes()
        else:
            return self.spike_list

    def plot_spikes(self, figsize=(8,6), linelengths=0.8, linewidths=1.0,
                    color_dict={'LIF': 'C0', 'IN': 'C1'}, tick_names=False):
        if self.spike_list is None:
            self.read_spikes()
            
        # Plot the event list information
        plt.figure(figsize=figsize)

        # We can also color the rows according to population
        if color_dict is None:
            color_dict = {key: f"C{i%10}" for i, key in enumerate(self.group_count.keys())}
        event_color = []
        for name, count in self.group_count.items():
            if name not in color_dict:
                color_dict[name] = f"C{len(color_dict)%10}"
            event_color.extend([color_dict[name]] * count)
        # colored lines (for legend)
        for key in color_dict.keys():
            plt.plot(0,0,'-',color=color_dict[key],linewidth=2.0)
        
        # The spike raster is plotted using eventplot
        plt.eventplot(self.spike_list, colors=event_color, lineoffsets=1,
                      linelengths=linelengths, linewidths=linewidths)

        # Tick names (may be too crowded with many neurons)
        if tick_names:
            plt.yticks(list(self.node_map.values()), list(self.node_map.keys()))
        
        plt.title('Spike Raster')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron (index)')
        plt.tight_layout()
        plt.legend(color_dict.keys())
