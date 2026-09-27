# Package Imports
from ..registry import ModelSpec, ModelBuilder, ModelRegistry
from sango.model import Neuron, Synapse


# ========================================================================
# STACS model builder
# ========================================================================

class StacsModelBuilder(ModelBuilder):
    def __init__(self, model_cls, graph_type):
        super().__init__(model_cls, graph_type=graph_type)

        self._model_type = ""
        self._extra_backend = {}

    def model_type(self, model_type):
        self._model_type = model_type
        return self

    def clocked(self, frontend_names):
        if isinstance(frontend_names, (list, set)):
            for frontend_name in frontend_names:
                self._units[frontend_name] = 'tick'
        else:
            self._units[frontend_names] = 'tick'
        return self

    def backend_key(self, key, value):
        self._extra_backend[key] = value
        return self

    def backend_keys(self, mapping):
        self._extra_backend.update(mapping)
        return self

    def build_backend_spec(self):
        spec = self.build_model_spec()

        spec.backend = {
            "model_type": self._model_type,
        }
        spec.backend.update(self._extra_backend)
        
        return spec

    def load_backend_from_dict(self, spec):
        """Load STACS-specific keys from a STACS model dictionary."""
        self._model_type = spec.get("model_type", self._model_type)

        known_keys = {"graph_type", "state", "param",
                      "model_type"}

        for key, value in spec.items():
            if key not in known_keys:
                self._extra_backend[key] = value

        return self

# ========================================================================
# STACS model registry
# ========================================================================

class StacsRegistry(ModelRegistry):
    builder_cls = StacsModelBuilder

    def __init__(self, graph_type_map=None):
        # STACS has a slightly different graph type mapping
        stacs_graph_type_map = {Neuron: "node",
                                Synapse: "edge"}
        if graph_type_map:
            stacs_graph_type_map.update(graph_type_map)
        super().__init__(graph_type_map=stacs_graph_type_map)

    def register(self, model_cls, graph_type=None):
        return super().register(model_cls, graph_type=graph_type)