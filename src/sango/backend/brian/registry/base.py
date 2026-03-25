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
                          'events'    : {},
                          'run_regularly' : [{'eqs': 'v*=(1.0-v_leak)', 'when': 'resets'},
                                             {'eqs': 'v+=v_bias',       'when': 'groups'}],
                          'state': {'v':        {'dsl': 'voltage',   'default': 0.0},
                                    'v_thresh': {'dsl': 'threshold', 'default': 1.0},
                                    'v_reset':  {'dsl': 'reset',     'default': 0.0},
                                    'v_bias':   {'dsl': 'bias',      'default': 0.0},
                                    'v_leak':   {'dsl': 'leak',      'default': 1.0}}},
                  'PSP': {'graph_type': 'synapse',
                          'model_eqs' : 'weight : 1',
                          'on_pre'    : 'v+=weight',
                          'state': {'delay':    {'dsl': 'delay',     'default': 1.0, 'unit': 'ms'},
                                    'weight':   {'dsl': 'weight',    'default': 1.0}}}
                 }