try:
    import brian2
except ImportError:
    brian2 = None
if brian2 is not None:
    from brian2 import NeuronGroup, Synapses, SpikeGeneratorGroup
    from brian2 import SpikeMonitor, StateMonitor
    from brian2 import ms, defaultclock

from collections import Counter

import numpy as np

from ..backend import Backend

# Brian 2 Simulation Backend
class SimBrian(Backend):
    def __init__(self, net, debug=False, verbose=False):
        if brian2 is None:
            raise ImportError("brian2 package is required for SimBrian")
        super().__init__(net, debug=debug, verbose=verbose)
        self.tstep = 1.0*ms
        self.tsim = None

    # Set up Brian 2 simulator
    def to_backend(self, **kwargs):
        defaultclock.dt = self.tstep

        # Spike generator inputs (sorted)
        if self.input_data:
            spike_index = []
            spike_times = []
            for i, times in enumerate(self.input_data.values()):
                for t in times:
                    spike_index.append(i)
                    spike_times.append(t*ms)
            self.spikegen_times, self.spikegen_index = [
                list(t) for t in zip(*sorted(zip(spike_times, spike_index)))]
        else:
            self.spikegen_index = []
            self.spikegen_times = []
        
        # Container for local neuron states (by group)
        self.neuron_states = dict()
        for name in self.group_count.keys():
            self.neuron_states[name] = dict()
            for state in self.model_registry[name]['state']:
                self.neuron_states[name][state] = []

        # Container for local synapse states (by connection group)
        self.synapse_connections = {}
        self.synapse_states = dict()
        for (n, s, t), variation in self.edge_groups.items():
            for v in variation.values():
                group_name = f"{n}_{s}_{t}" if v == 0 else f"{n}_{s}_{t}__{v}"
                self.synapse_connections[group_name] = {'i': [], 'j': []}
                self.synapse_states[group_name] = dict()
                for state in self.model_registry[n]['state']:
                    self.synapse_states[group_name][state] = []

        # Parameters (needs to be updated to use group_param)
        for group_name, params in self.group_params.items():
            model_name = self.group_models[group_name]
            for key, value in params.items():
                if ('unit' in self.model_registry[model_name]['param'][key] and
                    self.model_registry[model_name]['param'][key]['unit'] == 'ms'):
                    self.group_params[group_name][key] = value*ms

        # Neurons
        for n, data in enumerate(self.node_data):
            model_name = data['model']
            for state, state_dict in self.model_registry[model_name]['state'].items():
                if 'unit' in state_dict and state_dict['unit'] == 'ms':
                    self.neuron_states[model_name][state].append(data[state]*ms)
                else:
                    self.neuron_states[model_name][state].append(data[state])

        # Synapses
        for s in range(self.num_nodes):
            for t, data in self.edge_data[s].items():
                group_name = data['group_name']
                self.synapse_connections[group_name]['i'].append(self.local_index[s])
                self.synapse_connections[group_name]['j'].append(self.local_index[t])
                for state, state_dict in self.model_registry[data['model']]['state'].items():
                    if state == 'delay':
                        # Brian has a default delay of 0ms (to get to the next timestep)
                        self.synapse_states[group_name]['delay'].append((data['delay']-1.0)*ms)
                    elif 'unit' in state_dict and state_dict['unit'] == 'ms':
                        self.synapse_states[group_name][state].append(data[state]*ms)
                    else:
                        self.synapse_states[group_name][state].append(data[state])

        # Brian Network
        self.brian_net = brian2.Network()
        self.input_groups = dict()
        self.neuron_groups = dict()
        self.synapse_groups = dict()
        self.spike_monitors = dict()
        self.state_monitors = dict()

        # Create input and neuron groups (and their spike monitors)
        for name, count in self.group_count.items():
            # Spike generator group
            if self.model_registry[name]['graph_type'] == 'input':
                self.input_groups[name] = SpikeGeneratorGroup(count, self.spikegen_index,
                                                              self.spikegen_times, sorted=True)
                self.spike_monitors[name] = SpikeMonitor(self.input_groups[name])
            # Regular neuron model group
            elif self.model_registry[name]['graph_type'] == 'neuron':
                self.neuron_groups[name] = NeuronGroup(count, model=self.model_registry[name]['model_eqs'],
                                                       threshold=self.model_registry[name]['threshold'],
                                                       reset=self.model_registry[name]['reset'],
                                                       refractory=self.model_registry[name]['refractory'],
                                                       method=self.model_registry[name]['method'],
                                                       events=dict(self.model_registry[name]['events']),
                                                       namespace=self.group_params[name])
                # These "run regularly" methods bypass the standard Brian integration step
                if 'run_regularly' in self.model_registry[name]:
                    for program in self.model_registry[name]['run_regularly']:
                        self.neuron_groups[name].run_regularly(program['eqs'], when=program['when'])
                # These "run on event" methods trigger when a custom event happens
                if 'run_on_event' in self.model_registry[name]:
                    for program in self.model_registry[name]['run_on_event']:
                        self.neuron_groups[name].run_on_event(program['event'], program['eqs'])
                # Copy over states
                for state in self.model_registry[name]['state']:
                    getattr(self.neuron_groups[name], f"{state}")[:] = self.neuron_states[name][state]
                self.spike_monitors[name] = SpikeMonitor(self.neuron_groups[name])

        # Create synapse groups
        for (name, source, target) in self.edge_groups.keys():
            full_name = f"{name}_{source}_{target}"
            if self.model_registry[source]['graph_type'] == 'input':
                self.synapse_groups[full_name] = Synapses(self.input_groups[source],
                                                          self.neuron_groups[target],
                                                          model=self.model_registry[name]['model_eqs'],
                                                          on_pre=self.model_registry[name]['on_pre'])
            else:
                self.synapse_groups[full_name] = Synapses(self.neuron_groups[source],
                                                          self.neuron_groups[target],
                                                          model=self.model_registry[name]['model_eqs'],
                                                          on_pre=self.model_registry[name]['on_pre'])
            # Copy over connections
            self.synapse_groups[full_name].connect(i=self.synapse_connections[full_name]['i'],
                                                   j=self.synapse_connections[full_name]['j'])
            # Copy over states
            for state in self.model_registry[name]['state']:
                getattr(self.synapse_groups[full_name], f"{state}")[:,:] = self.synapse_states[full_name][state]

        # Recording (state monitors)
        if self.record_spec is not None:
            for name, value in self.record_spec.items():
                group = self.neuron_groups.get(name, self.synapse_groups.get(name, None))
                if group is None:
                    raise KeyError(f"group name {name} not found")
                self.state_monitors[name] = StateMonitor(source=group, **value)

        # Add all the objects to the network
        for value in self.input_groups.values():
            self.brian_net.add(value)
        for value in self.neuron_groups.values():
            self.brian_net.add(value)
        for value in self.synapse_groups.values():
            self.brian_net.add(value)
        for value in self.spike_monitors.values():
            self.brian_net.add(value)
        for value in self.state_monitors.values():
            self.brian_net.add(value)

        # This is the scheduling of events needed for the synapse input not be discarded
        # (the default handling of synapses occurs between thresholds and resets)
        self.brian_net.schedule = ['start', 'groups', 'thresholds', 'resets', 'synapses', 'end']
        # brian 2 default: ['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']

    # Run simulation backend
    def run_backend(self, timesteps=1, **kwargs):
        self.timesteps = timesteps
        self.tsim = self.timesteps * self.tstep
        self.brian_net.run(self.tsim)
    
    # Read spikes from spike monitors (in group order)
    def read_spikes(self):
        self.spike_list = []
        offset = 0
        for name, monitor in self.spike_monitors.items():
            self.spike_list.extend([[] for _ in range(self.group_count[name])])
            for s in range(len(monitor)):
                self.spike_list[offset+monitor.i[s]].append(monitor.t[s]/ms)
            offset += self.group_count[name]

        return self.spike_list
