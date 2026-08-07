# Prob model registry
model_registry = {
    'pLIF': {'node_class': 'LIFNeuron',
             'param': {'threshold': {'dsl': 'threshold', 'default': 0.0},
                       'reset_voltage': {'dsl': 'reset', 'default': 0.0},
                       'decay': {'dsl': 'leak', 'default': 1.0},
                       'voltage': {'dsl': 'voltage', 'default': 0.0},
                       'p': {'dsl': 'prob', 'default': 1.0},
                       'bias': {'dsl': 'bias', 'default': 0.0}}}
}
