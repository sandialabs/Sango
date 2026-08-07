# Base model registry
model_registry = {
    'LIF': {'node_class': 'LIFNeuron',
            'param': {'threshold': {'dsl': 'threshold', 'default': 0.0},
                      'reset_voltage': {'dsl': 'reset', 'default': 0.0},
                      'decay': {'dsl': 'leak', 'default': 1.0},
                      'voltage': {'dsl': 'voltage', 'default': 0.0},
                      'bias': {'dsl': 'bias', 'default': 0.0}}},
    'IN': {'node_class': 'InputNeuron',
           'param': {'threshold': {'dsl': None, 'default': 0.1},
                     'voltage': {'dsl': None, 'default': 0.0}}},
    'PSP': {'edge_class': 'Synapse',
            'param': {'weight': {'dsl': 'weight', 'default': 1.0},
                      'delay': {'dsl': 'delay', 'default': 1.0}}}
}
