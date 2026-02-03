# Model registry
model_registry = {'pLIF': {'graph_type': 'vertex', 'model_type': 'fugu_neuron',
                           'param': {},
                           'state': {'v':        {'dsl': 'voltage',   'default': 0.0},
                                     'v_thresh': {'dsl': 'threshold', 'default': 1.0},
                                     'v_reset':  {'dsl': 'reset',     'default': 0.0},
                                     'v_bias':   {'dsl': 'bias',      'default': 0.0},
                                     'v_leak':   {'dsl': 'leak',      'default': 1.0},
                                     'p_spike':  {'dsl': 'prob',      'default': 1.0},
                                     'I_syn':    {'dsl': None,        'default': 0.0},
                                     'I_clamp':  {'dsl': None,        'default': 0.0}}}
                  }