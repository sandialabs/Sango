# Prob model registry
model_registry = {
    'pLIF': {'graph_type': 'node',
             'node_class': 'LIFNeuron',
             'state': {'threshold': {'mapfrom': 'threshold', 'default': 0.0},
                       'reset_voltage': {'mapfrom': 'reset', 'default': 0.0},
                       'decay': {'mapfrom': 'leak', 'default': 1.0},
                       'voltage': {'mapfrom': 'voltage', 'default': 0.0},
                       'p': {'mapfrom': 'prob', 'default': 1.0},
                       'bias': {'mapfrom': 'bias', 'default': 0.0}}}
}
