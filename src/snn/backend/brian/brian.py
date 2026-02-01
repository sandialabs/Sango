import os
import sys
from pathlib import Path
import importlib.util

import brian2
from brian2 import NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor
from brian2 import ms, defaultclock

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Dynamically import the model registry files
model_registry = dict()
def import_registry():
    registry_dir = Path(__file__).resolve().parent / 'registry'
    sys.path.insert(0, str(registry_dir.parent))
    for file_path in registry_dir.glob("*.py"):
        module_name = f"registry.{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_registry.update(module.model_registry)
    sys.path.pop(0)
import_registry()

# Brian Simulation Backend
class SimBrian:
    def __init__(self, dsl_net):
        self.dsl_net = dsl_net
        self.tstep = 1.0*ms
        self.timesteps = None

        self.spike_list = None
        
    # Convert between dsl model to brian2 model
    def rekey_model(self, data):
        for key, value in model_registry[data['model']]['state'].items():
            if value['dsl'] is not None:
                data[key] = data.pop(value['dsl'])
            else:
                data[key] = value['default']
        return data

    def compile(self, debug=False):
        self.debug = debug
        
        # Convert network to stacs
        self.to_brian()
        
    def run(self, timesteps=10.0):
        self.timesteps = timesteps*self.tstep
        # Run network
        self.brian_net.run(self.timesteps)

    def to_brian(self):
        # Get a flattened graph object
        self.dsl_graph = self.dsl_net._topology.to_nx()
        defaultclock.dt = 1.0*ms

        # Convert to Brian2 network objects
        self.num_nodes = self.dsl_graph.number_of_nodes()
        self.node_map = dict()
        self.node_data = [dict() for _ in range(self.num_nodes)]
        self.edge_data = [dict() for _ in range(self.num_nodes)]
        self.group_count = Counter()
        self.local_index = [0 for _ in range(self.num_nodes)]
        self.edge_set = set()
        self.spike_input = []
        
        # Global node data
        for n, (node, data) in enumerate(self.dsl_graph.nodes(data=True)):
            self.node_map[node] = n
            self.node_data[n] = self.rekey_model(data)
            self.group_count.update([self.node_data[n]['model']])
            self.local_index[n] = self.group_count[self.node_data[n]['model']] - 1
            if model_registry[self.node_data[n]['model']]['graph_type'] == 'input':
                self.spike_input.append(data['times'])
        
        # Global edge data
        for source, target, data in self.dsl_graph.edges(data=True):
            s = self.node_map[source]
            t = self.node_map[target]
            self.edge_data[s][t] = self.rekey_model(data)
            self.edge_set.add((self.edge_data[s][t]['model'], self.node_data[s]['model'], self.node_data[t]['model']))
            
        # Spike generator inputs (sorted)
        spike_index = []
        spike_times = []
        for i, times in enumerate(self.spike_input):
            for t in times:
                spike_index.append(i)
                spike_times.append(t*ms)
        self.spikegen_times, self.spikegen_index = [list(t) for t in zip(*sorted(zip(spike_times, spike_index)))]
        
        # Container for local neuron states (by group)
        self.neuron_states = dict()
        for name in self.group_count.keys():
            self.neuron_states[name] = dict()
            for state in model_registry[name]['state']:
                self.neuron_states[name][state] = []

        # Container for local synapse states (by connection)
        self.synapse_connections = {f"{n}_{s}_{t}": {'i': [], 'j': []} for (n,s,t) in self.edge_set}
        self.synapse_states = dict()
        for name, source, target in self.edge_set:
            full_name = f"{name}_{source}_{target}"
            self.synapse_states[full_name] = dict()
            for state in model_registry[name]['state']:
                self.synapse_states[full_name][state] = []
        
        # Neurons
        for n, data in enumerate(self.node_data):
            name = data['model']
            for state in model_registry[name]['state']:
                self.neuron_states[name][state].append(data[state])
        
        # Synapses
        for s in range(self.num_nodes):
            for t, data in self.edge_data[s].items():
                full_name = f"{data['model']}_{self.node_data[s]['model']}_{self.node_data[t]['model']}"
                self.synapse_connections[full_name]['i'].append(self.local_index[s])
                self.synapse_connections[full_name]['j'].append(self.local_index[t])
                for state in model_registry[data['model']]['state']:
                    if state == 'delay':
                        # Brian has a default delay of 0ms (to get to the next timestep)
                        self.synapse_states[full_name]['delay'].append((data['delay']-1.0)*ms)
                    else:
                        self.synapse_states[full_name][state].append(data[state])

        # Brian Network
        self.brian_net = brian2.Network()
        self.input_groups = dict()
        self.neuron_groups = dict()
        self.synapse_groups = dict()
        self.spike_monitors = dict()

        # Create input and neuron groups (and their spike monitors)
        for name, count in self.group_count.items():
            # Spike generator group
            if model_registry[name]['graph_type'] == 'input':
                self.input_groups[name] = SpikeGeneratorGroup(count, self.spikegen_index,
                                                              self.spikegen_times, sorted=True)
                self.spike_monitors[name] = SpikeMonitor(self.input_groups[name])
            # Regular neuron model group
            elif model_registry[name]['graph_type'] == 'neuron':
                self.neuron_groups[name] = NeuronGroup(count, model=model_registry[name]['model_eqs'],
                                                       threshold=model_registry[name]['threshold'],
                                                       reset=model_registry[name]['reset'],
                                                       method=model_registry[name]['method'],
                                                       events=model_registry[name]['events'])
                # These "run regularly" methods bypass the standard Brian integration step
                if 'run_regularly' in model_registry[name]:
                    for program in model_registry[name]['run_regularly']:
                        self.neuron_groups[name].run_regularly(program['eqs'], when=program['when'])
                # These "run on event" methods trigger when a custom event happens
                if 'run_on_event' in model_registry[name]:
                    for program in model_registry[name]['run_on_event']:
                        self.neuron_groups[name].run_on_event(program['event'], program['eqs'])
                # Copy over states
                for state in model_registry[name]['state']:
                    getattr(self.neuron_groups[name], f"{state}")[:] = self.neuron_states[name][state]
                self.spike_monitors[name] = SpikeMonitor(self.neuron_groups[name])
        
        # Create synapse groups
        for (name, source, target) in self.edge_set:
            full_name = f"{name}_{source}_{target}"
            if model_registry[source]['graph_type'] == 'input':
                self.synapse_groups[full_name] = Synapses(self.input_groups[source],
                                                          self.neuron_groups[target],
                                                          model=model_registry[name]['model_eqs'],
                                                          on_pre=model_registry[name]['on_pre'])
            else:
                self.synapse_groups[full_name] = Synapses(self.neuron_groups[source],
                                                          self.neuron_groups[target],
                                                          model=model_registry[name]['model_eqs'],
                                                          on_pre=model_registry[name]['on_pre'])
            # Copy over connections
            self.synapse_groups[full_name].connect(i=self.synapse_connections[full_name]['i'],
                                                   j=self.synapse_connections[full_name]['j'])
            # Copy over states
            for state in model_registry[name]['state']:
                getattr(self.synapse_groups[full_name], f"{state}")[:,:] = self.synapse_states[full_name][state]
        
        # Add all the objects to the network
        for value in self.input_groups.values():
            self.brian_net.add(value)
        for value in self.neuron_groups.values():
            self.brian_net.add(value)
        for value in self.synapse_groups.values():
            self.brian_net.add(value)
        for value in self.spike_monitors.values():
            self.brian_net.add(value)

        # This is the scheduling of events needed for the synapse input not be discarded
        # (the default handling of synapses occurs between thresholds and resets)
        self.brian_net.schedule = ['start', 'groups', 'thresholds', 'resets', 'synapses', 'end']
    
    # Collect any output from the simulation
    def read_spikes(self):
        self.spike_list = []
        offset = 0
        for name, monitor in self.spike_monitors.items():
            self.spike_list.extend([[] for _ in range(self.group_count[name])])
            for s in range(len(monitor)):
                self.spike_list[offset+monitor.i[s]].append(monitor.t[s]/ms)
            offset += self.group_count[name]
        
    # Return spikes as event list
    def get_spikes(self):
        if self.spike_list is None:
            return self.read_spikes()
        else:
            return self.spike_list
    
    def plot_spikes(self, figsize=(8,6), linelengths=0.8, linewidths=1.0,
                    color_dict={'LIF': 'C0', 'IN': 'C1'}):
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
        
        plt.title('Spike Raster')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron (index)')
        plt.tight_layout()
        plt.legend(color_dict.keys())