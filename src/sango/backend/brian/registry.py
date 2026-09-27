# Package Imports
from ..registry import ModelSpec, ModelBuilder, ModelRegistry


# ========================================================================
# Brian model builder
# ========================================================================

class BrianModelBuilder(ModelBuilder):
    def __init__(self, model_cls, graph_type):
        super().__init__(model_cls, graph_type=graph_type)

        self._model_eqs = ''
        self._method = 'exact'
        self._threshold = ''
        self._reset = ''
        self._refractory = False
        self._events = {}
        self._on_pre = ''
        self._extra_backend = {}

    def model_eqs(self, model_eqs):
        self._model_eqs = model_eqs
        return self

    def method(self, method):
        self._method = method
        return self

    def threshold(self, threshold):
        self._threshold = threshold
        return self

    def reset(self, reset):
        self._reset = reset
        return self

    def refractory(self, refractory):
        self._refractory = refractory
        return self

    def event(self, name, value):
        self._events[name] = value
        return self

    def events(self, events):
        self._events.update(events)
        return self
    
    def on_pre(self, on_pre):
        self._on_pre = on_pre
        return self

    def clocked(self, frontend_names):
        if isinstance(frontend_names, (list, set)):
            for frontend_name in frontend_names:
                self._units[frontend_name] = 'ms'
        else:
            self._units[frontend_names] = 'ms'
        return self

    def backend_key(self, key, value):
        self._extra_backend[key] = value
        return self

    def backend_keys(self, mapping):
        self._extra_backend.update(mapping)
        return self

    def build_backend_spec(self):
        spec = self.build_model_spec()

        if spec.graph_type == 'neuron':
            spec.backend = {
                'model_eqs': self._model_eqs,
                'method': self._method,
                'threshold': self._threshold,
                'reset': self._reset,
                'refractory': self._refractory,
                'events': self._events,
            }
        elif spec.graph_type == 'synapse':
            spec.backend = {
                'model_eqs': self._model_eqs,
                'on_pre': self._on_pre,
            }
        spec.backend.update(self._extra_backend)

        return spec

    def load_backend_from_dict(self, spec):
        """Load Brian-specific keys from a Brian model dictionary."""
        self._model_eqs = spec.get('model_eqs', self._model_eqs)
        self._method = spec.get('method', self._method)
        self._threshold = spec.get('threshold', self._threshold)
        self._reset = spec.get('reset', self._reset)
        self._refractory = spec.get('refractory', self._refractory)
        self._on_pre = spec.get('on_pre', self._on_pre)

        # Handle events dictionary
        events = spec.get('events')
        if events is not None:
            self._events = dict(events)

        known_keys = {"graph_type", "state", "param",
                      "model_eqs", "method", "threshold",
                      "reset", "refractory", "events",
                      "on_pre"}

        for key, value in spec.items():
            if key not in known_keys:
                self._extra_backend[key] = value

        return self


# ========================================================================
# Brian model registry
# ========================================================================

class BrianRegistry(ModelRegistry):
    builder_cls = BrianModelBuilder

    def register(self, model_cls, graph_type=None):
        return super().register(model_cls, graph_type=graph_type)