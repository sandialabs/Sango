try:
    import fugu
except ImportError:
    fugu = None

from collections import Counter, defaultdict

import numpy as np
import networkx as nx

from ..backend import Backend

# Fugu Simulation Backend
class SimFugu(Backend):
    def __init__(self, net, debug=False, verbose=False):
        if fugu is None:
            raise ImportError("fugu package is required for SimFugu")
        super().__init__(net, debug=debug, verbose=verbose)

        self.fugu_backend = None
        self.backend_args = None
        self.scaffold = None
        self.output_neurons = []
        self.fugu_result = None

    # Compile to Fugu backends
    def to_backend(self, **kwargs):
        if self.is_multigraph:
            raise TypeError("SimFugu does not currently support multigraphs.")

        backend = kwargs.get('backend', 'snn')
        self.backend_args = kwargs.get('backend_args', {'record': 'all'})

        if backend == 'snn':
            from fugu.backends import snn_Backend
            self.fugu_backend = snn_Backend()
        elif backend == 'stacs':
            from fugu.backends import stacs_Backend
            self.fugu_backend = stacs_Backend()
        else:
            raise NotImplementedError(f"Backend '{backend}' is not supported.")

        if self.backend_args['record'] != 'all':
            if type(self.backend_args['record']) is list:
                self.output_neurons = self.backend_args['record']
            else:
                raise TypeError(f"Recorded output neurons must be type list, "
                                f"not {type(self.backend_args['record']).__name__}")
        if self.debug:
            self.backend_args['debug_mode'] = True

        # Build Fugu scaffold
        self.scaffold = self.to_scaffold()

        # Compile scaffold in Fugu backend
        self.fugu_backend.compile(self.scaffold, self.backend_args)

    # Run the network
    def run_backend(self, timesteps=1, **kwargs):
        self.fugu_result = self.fugu_backend.run(n_steps=timesteps)

    # Dummy brick object for Fugu scaffold
    class DummyBrick():
        def __init__(self, name='dummy'):
            self.name = name
            self.vector = []
            self.index = 0
            self.is_built = True
        # This method is mainly used for managing Fugu inputs
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
        def set_properties(self, properties=None):
            properties = properties or {}
            if "spike_vector" in properties:
                self.vector = properties["spike_vector"]
            return

    # Convert Sango network to Fugu Scaffold
    def to_scaffold(self):
        from fugu import Scaffold
        from fugu.scaffold import PortData, PortSpec
        from fugu.scaffold import ChannelData, ChannelSpec

        # Set up a dummy scaffold in Fugu
        scaffold = Scaffold()
        scaffold_circuit = nx.DiGraph()
        # Input and main Sango bricks
        input_brick = {'tag': 'input',
                       'name': 'InputBrick',
                       'brick': self.DummyBrick('input'),
                       'layer': 'input',
                       'ports': {'output': None},
                       'is_built': True}
        sango_brick = {'tag': 'sango',
                       'name': 'SangoBrick',
                       'brick': self.DummyBrick('sango'),
                       'layer': 'output',
                       'ports': {'input': None, 'output': None},
                       'is_built': True}
        # Connect dummy circuit
        scaffold_circuit.add_node(0, **input_brick)
        scaffold_circuit.add_node(1, **sango_brick)
        scaffold_circuit.add_edge(0, 1, bind={'input': 'output'})
        scaffold.tag_to_name = {'input': 'InputBrick', 'sango': 'SangoBrick'}
        scaffold.name_to_tag = {'InputBrick': 'input', 'SangoBrick': 'sango'}
        scaffold.brick_to_number = {'InputBrick': 0, 'SangoBrick': 1}

        # Set up underlying graph
        scaffold_graph = nx.DiGraph()

        # Attach spike times to input brick
        spike_times = defaultdict(list)
        for n, (neuron, times) in enumerate(self.input_data.items()):
            for t in times:
                spike_times[t].append(neuron)
        if spike_times:
            spike_vector = [spike_times[key] for key in range(max(spike_times.keys()) + 1)]
        else:
            spike_vector = []
        scaffold_circuit.nodes[0]['brick'].vector = spike_vector

        # Remove potential input neurons from output list
        input_set = set(self.input_data.keys())
        self.output_neurons = [n for n in self.output_neurons if n not in input_set]

        # Set up ports
        scaffold_circuit.nodes[0]['ports']['output'] = PortData(
            spec=PortSpec(name='output'),
            channels={'data': ChannelData(
                spec=ChannelSpec(name='data', coding='Raster',
                                 shape=(len(self.input_data),)),
                neurons=list(self.input_data.keys()))})
        scaffold_circuit.nodes[1]['ports']['input'] = PortData(
            spec=PortSpec(name='input'),
            channels={'data': ChannelData(
                spec=ChannelSpec(name='data', coding='Raster', shape=None),
                neurons=[])})
        scaffold_circuit.nodes[1]['ports']['output'] = PortData(
            spec=PortSpec(name='output'),
            channels={'data': ChannelData(
                spec=ChannelSpec(name='data', coding='Raster',
                                 shape=(len(self.output_neurons),)),
                neurons=self.output_neurons)})
        scaffold.circuit = scaffold_circuit

        # Insert nodes by group
        for n, (node, nidx) in enumerate(self.node_map.items()):
            node_data = self.node_data[n]
            model_name = node_data['model']
            node_data = {key: node_data[key] for key in self.model_registry[model_name]['state']}
            if self.model_registry[model_name]['graph_type'] == 'input':
                scaffold_graph.add_node(node, neuron_number=nidx, brick='InputBrick',
                                        model=model_name, **node_data)
            else:
                scaffold_graph.add_node(node, neuron_number=nidx, brick='SangoBrick',
                                        model=model_name, **node_data)

        # Global edge data
        for source, target, data in self.ref_graph.edges(data=True):
            edge_data = self.edge_data[self.node_index[source]][self.node_index[target]]
            model_name = edge_data['model']
            edge_data = {key: edge_data[key] for key in self.model_registry[model_name]['state']}
            scaffold_graph.add_edge(source, target, model=model_name, **edge_data)

        # Attach the built graph to the scaffold and mark as built
        scaffold.graph = scaffold_graph
        scaffold.is_built = True

        return scaffold

    # Collect any output from the simulation
    def read_spikes(self):
        spike_dict = defaultdict(list)
        for spike in self.fugu_result.itertuples(index=False):
            spike_dict[int(spike.neuron_number)].append(spike.time)
        self.spike_list = [spike_dict[key] for key in range(len(self.node_map))]
        return self.spike_list
