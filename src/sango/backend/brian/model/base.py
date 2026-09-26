# Base model registry
model_registry = {'IN':  {'graph_type': 'input', 'state': {}},
                  'LIF': {'graph_type': 'neuron',
                          'model_eqs' : '''v : 1
                                           v_thresh : 1
                                           v_reset : 1
                                           v_bias : 1
                                           v_leak : 1
                                        ''',
                          'method'    : 'exact',
                          'threshold' : 'v>v_thresh',
                          'reset'     : 'v=v_reset',
                          'refractory': False,
                          'events'    : {},
                          'run_regularly' : [{'eqs': 'v*=(1.0-v_leak)', 'when': 'resets'},
                                             {'eqs': 'v+=v_bias',       'when': 'groups'}],
                          'state': {'v':        {'mapfrom': 'voltage',   'default': 0.0},
                                    'v_thresh': {'mapfrom': 'threshold', 'default': 1.0},
                                    'v_reset':  {'mapfrom': 'reset',     'default': 0.0},
                                    'v_bias':   {'mapfrom': 'bias',      'default': 0.0},
                                    'v_leak':   {'mapfrom': 'leak',      'default': 1.0}}},
                  'PSP': {'graph_type': 'synapse',
                          'model_eqs' : 'weight : 1',
                          'on_pre'    : 'v+=weight',
                          'state': {'delay':    {'mapfrom': 'delay',     'default': 1.0, 'unit': 'ms'},
                                    'weight':   {'mapfrom': 'weight',    'default': 1.0}}}
                 }