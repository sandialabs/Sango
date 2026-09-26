# Model registry
model_registry = {'pLIF': {'graph_type': 'node',
                           'model_type': 'fugu_neuron',
                           'param': {},
                           'state': {'v':        {'mapfrom': 'voltage',   'default': 0.0},
                                     'v_thresh': {'mapfrom': 'threshold', 'default': 1.0},
                                     'v_reset':  {'mapfrom': 'reset',     'default': 0.0},
                                     'v_bias':   {'mapfrom': 'bias',      'default': 0.0},
                                     'v_leak':   {'mapfrom': 'leak',      'default': 1.0},
                                     'p_spike':  {'mapfrom': 'prob',      'default': 1.0},
                                     'I_syn':    {'mapfrom': None,        'default': 0.0},
                                     'I_clamp':  {'mapfrom': None,        'default': 0.0}}}
                  }