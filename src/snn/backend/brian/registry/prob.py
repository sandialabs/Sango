# Model registry
model_registry = {'pLIF': {'graph_type': 'neuron',
                           'model_eqs' : '''v : 1
                                            v_thresh : 1
                                            v_reset : 1
                                            v_bias : 1
                                            v_leak : 1
                                            p_spike : 1
                                         ''',
                           'method'    : 'exact',
                           'threshold' : '(v>v_thresh) and (rand()<=p_spike)',
                           'reset'     : '', # probabilistic spiking requires custom event
                           'events'    : {'pass_thresh': 'v>v_thresh'},
                           'run_regularly' : [{'eqs': 'v*=(1.0-v_leak)', 'when': 'resets'},
                                              {'eqs': 'v+=v_bias',       'when': 'groups'}],
                           'run_on_event'  : [{'event': 'pass_thresh', 'eqs': 'v=v_reset'}],
                           'state': {'v':        {'dsl': 'voltage',   'default': 0.0},
                                     'v_thresh': {'dsl': 'threshold', 'default': 1.0},
                                     'v_reset':  {'dsl': 'reset',     'default': 0.0},
                                     'v_bias':   {'dsl': 'bias',      'default': 0.0},
                                     'v_leak':   {'dsl': 'leak',      'default': 1.0},
                                     'p_spike':  {'dsl': 'prob',      'default': 1.0}}}
                 }