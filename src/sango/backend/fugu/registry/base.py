# Base model registry
model_registry = {
    'LIF': {'graph_type': 'node',
            'node_class': 'LIFNeuron',
            'state': {'threshold': {'dsl': 'threshold', 'default': 0.0},
                      'reset_voltage': {'dsl': 'reset', 'default': 0.0},
                      'decay': {'dsl': 'leak', 'default': 1.0},
                      'voltage': {'dsl': 'voltage', 'default': 0.0},
                      'bias': {'dsl': 'bias', 'default': 0.0}}},
    'IN': {'graph_type': 'input',
           'node_class': 'InputNeuron',
           'state': {'threshold': {'dsl': None, 'default': 0.1},
                     'voltage': {'dsl': None, 'default': 0.0}}},
    'PSP': {'graph_type': 'edge',
            'edge_class': 'Synapse',
            'state': {'weight': {'dsl': 'weight', 'default': 1.0},
                      'delay': {'dsl': 'delay', 'default': 1.0}}}
}
