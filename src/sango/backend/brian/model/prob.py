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
                           'refractory': False,
                           'events'    : {'pass_thresh': 'v>v_thresh'},
                           'run_regularly' : [{'eqs': 'v*=(1.0-v_leak)', 'when': 'resets'},
                                              {'eqs': 'v+=v_bias',       'when': 'groups'}],
                           'run_on_event'  : [{'event': 'pass_thresh', 'eqs': 'v=v_reset'}],
                           'state': {'v':        {'mapfrom': 'voltage',   'default': 0.0},
                                     'v_thresh': {'mapfrom': 'threshold', 'default': 1.0},
                                     'v_reset':  {'mapfrom': 'reset',     'default': 0.0},
                                     'v_bias':   {'mapfrom': 'bias',      'default': 0.0},
                                     'v_leak':   {'mapfrom': 'leak',      'default': 1.0},
                                     'p_spike':  {'mapfrom': 'prob',      'default': 1.0}}}
                 }