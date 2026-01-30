# Base model registry
model_registry = {'SI':  {'graph_type': 'stream', 'model_type': 'spike_input',
                          'state': {}},
                  'SC':  {'graph_type': 'edge',   'model_type': 'spike_clamp',
                          'state': {'delay':    {'dsl': 'delay',     'default': 1.0, 'rep': 'tick'}}},
                  'IN':  {'graph_type': 'vertex', 'model_type': 'simple_input',
                          'state': {'I_clamp':  {'dsl': None,        'default': 0.0}}},
                  'LIF': {'graph_type': 'vertex', 'model_type': 'simple_neuron',
                          'state': {'v':        {'dsl': 'voltage',   'default': 0.0},
                                    'v_thresh': {'dsl': 'threshold', 'default': 1.0},
                                    'v_reset':  {'dsl': 'reset',     'default': 0.0},
                                    'v_bias':   {'dsl': 'bias',      'default': 0.0},
                                    'v_leak':   {'dsl': 'leak',      'default': 1.0},
                                    'I_syn':    {'dsl': None,        'default': 0.0}}},
                  'PSP': {'graph_type': 'edge',   'model_type': 'simple_synapse',
                          'state': {'delay':    {'dsl': 'delay',     'default': 1.0, 'rep': 'tick'},
                                    'weight':   {'dsl': 'weight',    'default': 1.0}}}
                 }