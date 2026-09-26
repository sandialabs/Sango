# Base model registry
model_registry = {
    'LIF': {'graph_type': 'node',
            'node_class': 'LIFNeuron',
            'state': {'threshold': {'mapfrom': 'threshold', 'default': 0.0},
                      'reset_voltage': {'mapfrom': 'reset', 'default': 0.0},
                      'decay': {'mapfrom': 'leak', 'default': 1.0},
                      'voltage': {'mapfrom': 'voltage', 'default': 0.0},
                      'bias': {'mapfrom': 'bias', 'default': 0.0}}},
    'IN': {'graph_type': 'input',
           'node_class': 'InputNeuron',
           'state': {'threshold': {'mapfrom': None, 'default': 0.1},
                     'voltage': {'mapfrom': None, 'default': 0.0}}},
    'PSP': {'graph_type': 'edge',
            'edge_class': 'Synapse',
            'state': {'weight': {'mapfrom': 'weight', 'default': 1.0},
                      'delay': {'mapfrom': 'delay', 'default': 1.0}}}
}
