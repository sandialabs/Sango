# General Imports
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import Any, Optional
import warnings

# Package Imports
from sango.model import InputModel, Neuron, Synapse, get_shared_params


# ========================================================================
# Model specifications for builder
# ========================================================================

@dataclass
class FieldSpec:
    mapfrom: Optional[str]
    default: Any
    unit: Optional[str] = None

    def to_dict(self):
        d = {"mapfrom": self.mapfrom,
             "default": self.default}
        if self.unit is not None:
            d["unit"] = self.unit
        return d

@dataclass
class ModelSpec:
    graph_type: str
    state: dict[str, FieldSpec]
    param: dict[str, FieldSpec]
    # Backend-specific keys
    backend: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = {"graph_type": self.graph_type}
        d.update(self.backend) # backend-specific keys ordered first
        d["state"] = {name: spec.to_dict()
                      for name, spec in self.state.items()}
        d["param"] = {name: spec.to_dict()
                      for name, spec in self.param.items()}
        return d


# ========================================================================
# Model builder
# ========================================================================

class ModelBuilder:
    def __init__(self, model_cls, graph_type):
        if not is_dataclass(model_cls):
            raise TypeError(f"{model_cls!r} must be a dataclass")
        self.model_cls = model_cls
        self._name = getattr(model_cls, "model")
        self._graph_type = graph_type

        self._renames = {} # {frontend_name: backend_name}
        self._defaults = {}
        self._units = {}

        # Override which fields get placed in state/param
        self._state_fields = set()
        self._param_fields = set()
        # Fields that get omitted by backend
        self._omit_fields = set(["model"])
        
        # Backend-only fields. These do not exist in the frontend
        # dataclass. By default their mapfrom field is None.
        self._extra_state = {} # {backend_name: FieldSpec}
        self._extra_param = {}

    @property
    def model_name(self):
        return self._name

    # ---------------------------------------------------------------
    # Fluent API interface
    # ---------------------------------------------------------------
    
    def name(self, name):
        self._name = name
        return self

    def graph_type(self, graph_type):
        self._graph_type = graph_type
        return self

    def rename(self, frontend_name, backend_name):
        self._renames[frontend_name] = backend_name
        return self

    def renames(self, mapping):
        self._renames.update(mapping)
        return self

    def default(self, frontend_name, value):
        self._defaults[frontend_name] = value
        return self

    def defaults(self, mapping):
        self._defaults.update(mapping)
        return self

    def unit(self, frontend_name, unit):
        self._units[frontend_name] = unit
        return self

    def units(self, mapping):
        self._units.update(mapping)
        return self

    def omit(self, frontend_name):
        """Ignore a frontend dataclass field."""
        self._omit_fields.add(frontend_name)
        return self
    
    def omits(self, frontend_names):
        """Ignore multiple frontend dataclass fields."""
        self._omit_fields.update(frontend_names)
        return self

    def switch_to_state(self, frontend_name):
        """Force an original frontend dataclass field to a backend state."""
        self._state_fields.add(frontend_name)
        self._param_fields.discard(frontend_name)
        return self

    def switch_to_param(self, frontend_name):
        """Force an original frontend dataclass field to a backend param."""
        self._param_fields.add(frontend_name)
        self._state_fields.discard(frontend_name)
        return self
    
    def extra_state(self, backend_name, default=0.0, unit=None):
        """Add a backend-only state variable."""
        self._extra_state[backend_name] = FieldSpec(
            mapfrom=None, default=default, unit=unit)
        return self

    def extra_states(self, mapping):
        """Add multiple backend-only states."""
        if isinstance(mapping, (list, set)):
            # add with defaults only
            for backend_name in mapping:
                self._extra_state[backend_name] = FieldSpec(
                    mapfrom=None, default=0.0, unit=None)
        else:
            for backend_name, value in mapping.items():
                default = value.get("default", 0.0)
                unit = value.get("unit", None)
                self._extra_state[backend_name] = FieldSpec(
                    mapfrom=None, default=default, unit=unit)
        return self

    def extra_param(self, backend_name, default=0.0, unit=None):
        """Add a backend-only param variable."""
        self._extra_param[backend_name] = FieldSpec(
            mapfrom=None, default=default, unit=unit)
        return self

    def extra_params(self, mapping):
        """Add multiple backend-only params."""
        if isinstance(mapping, (list, set)):
            # add with defaults only
            for backend_name in mapping:
                self._extra_param[backend_name] = FieldSpec(
                    mapfrom=None, default=0.0, unit=None)
        else:
            for backend_name, value in mapping.items():
                default = value.get("default", 0.0)
                unit = value.get("unit", None)
                self._extra_param[backend_name] = FieldSpec(
                    mapfrom=None, default=default, unit=unit)
        return self

    # ---------------------------------------------------------------
    # Build/export
    # ---------------------------------------------------------------

    @staticmethod
    def _field_default(f):
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:  # type: ignore[attr-defined]
            return f.default_factory()        # type: ignore[misc]
        return MISSING

    def build_model_spec(self):
        state = {} # {backend_name: FieldSpec}
        param = {} # 

        # Loop through model dataclass
        for f in fields(self.model_cls):
            frontend_name = f.name

            if frontend_name in self._omit_fields:
                continue

            # Get default field
            field_default = self._field_default(f)
            if field_default is MISSING and frontend_name not in self._defaults:
                raise ValueError(
                    f"Field {frontend_name!r} has no dataclass default. "
                    "Provide one with .default(...) or .defaults(...)."
                )

            # Generate spec
            backend_name = self._renames.get(frontend_name, frontend_name)
            spec = FieldSpec(
                mapfrom=frontend_name,
                default=self._defaults.get(frontend_name, field_default),
                unit=self._units.get(frontend_name),
            )

            # Populate based on shared params
            shared_params = get_shared_params(self.model_cls)
            if frontend_name in self._param_fields:
                param[backend_name] = spec
            elif frontend_name in self._state_fields:
                state[backend_name] = spec
            elif frontend_name in shared_params:
                param[backend_name] = spec
            else:
                state[backend_name] = spec

        # Backend-only fields are applied last so they can also
        # intentionally override generated entries if needed.
        state.update(self._extra_state)
        param.update(self._extra_param)

        return ModelSpec(graph_type=self._graph_type,
                         state=state, param=param)

    # Method to be overridden in backend builder
    def build_backend_spec(self):
        return self.build_model_spec()

    def to_dict(self):
        return self.build_backend_spec().to_dict()
  

    # ---------------------------------------------------------------
    # Load from dictionary
    # ---------------------------------------------------------------

    def load_from_dict(self, spec):
        """Seed this builder from a model dictionary."""
        self.load_model_from_dict(spec)
        self.load_backend_from_dict(spec)
        return self

    def load_model_from_dict(self, spec):
        """Load model registry information from a model dictionary."""
        if "graph_type" in spec:
            self._graph_type = spec["graph_type"]

        frontend_fields = {f.name for f in fields(self.model_cls)}

        for backend_name, entry in spec.get("state", {}).items():
            frontend_name =entry.get("mapfrom", None)
            default=entry.get("default", 0.0)
            unit=entry.get("unit", None)
            if frontend_name is not None and frontend_name in frontend_fields:
                if backend_name != frontend_name:
                    self._renames[frontend_name] = backend_name
                self._defaults[frontend_name] = default
                if unit is not None:
                    self._units[frontend_name] = unit
                self._state_fields.add(frontend_name)
                self._param_fields.discard(frontend_name)
            else: # backend only state
                self._extra_state[backend_name] = FieldSpec(
                    mapfrom=None, default=default, unit=unit)
        
        for backend_name, entry in spec.get("param", {}).items():
            frontend_name =entry.get("mapfrom", None)
            default=entry.get("default", 0.0)
            unit=entry.get("unit", None)
            if frontend_name is not None and frontend_name in frontend_fields:
                if backend_name != frontend_name:
                    self._renames[frontend_name] = backend_name
                self._defaults[frontend_name] = default
                if unit is not None:
                    self._units[frontend_name] = unit
                self._param_fields.add(frontend_name)
                self._state_fields.discard(frontend_name)
            else: # backend only param
                self._extra_param[backend_name] = FieldSpec(
                    mapfrom=None, default=default, unit=unit)

        return self

    # Method to be overridden in backend builder
    def load_backend_from_dict(self, spec):
        """Load backend registry information from a model dictionary."""
        return self


# ========================================================================
# Model Registry (contains multiple builders)
# ========================================================================

class ModelRegistry:
    builder_cls = ModelBuilder

    def __init__(self, graph_type_map=None):
        # Map model types to graph types
        self.graph_type_map = {InputModel: "input",
                               Neuron: "neuron",
                               Synapse: "synapse"}
        if graph_type_map:
            self.graph_type_map.update(graph_type_map)
        self.default_graph_type = "unknown"

        # Dictionary of model builders
        self._builders = {} # {model_name: ModelBuilder}

    def register(self, model_cls, graph_type=None):
        if graph_type is None:
            graph_type = self.infer_graph_type(model_cls)

        builder = self.builder_cls(model_cls, graph_type=graph_type)
        self._builders[builder.model_name] = builder
        
        return builder

    def infer_graph_type(self, model_cls):
        for cls in model_cls.__mro__:
            if cls in self.graph_type_map:
                return self.graph_type_map[cls]
        return self.default_graph_type

    def add_graph_type(self, cls, graph_type):
        self.graph_type_map[cls] = graph_type

    def to_dict(self):
        return {name: builder.to_dict()
                for name, builder in self._builders.items()}

    def __getitem__(self, name):
        return self._builders[name]

    def merge(self, *registries, overwrite=True):
        """Merge one or more registry instances into this registry.

        This mutates the current registry, similar to dict.update(...).
        """
        for registry in registries:
            if not isinstance(registry, ModelRegistry):
                raise TypeError(
                    f"Can only merge ModelRegistry instances, got "
                    f"{type(registry).__name__}."
                )

            for model_name, builder in registry._builders.items():
                if not overwrite and model_name in self._builders:
                    raise ValueError(
                        f"Registry already contains model {model_name!r}. "
                        "Pass overwrite=True to replace it."
                    )

                # Rebuild through this registry from dictionary
                self.register_from_dict(builder.model_cls, builder.to_dict())

        return self

    
    # ---------------------------------------------------------------
    # Load from dictionary
    # ---------------------------------------------------------------
    
    def register_from_dict(self, model_cls, spec_or_entry, graph_type=None):
        """Register a model builder seeded from a model dictionary.
        
        Returns the fluent builder so users can continue modifying it.
        """
        model_name = getattr(model_cls, "model")
        # Extract model spec directly
        if any(key in spec_or_entry for key in ("graph_type", "state", "param")):
            model_spec = spec_or_entry
        # Otherwise expect a one-model registry dict:
        elif len(spec_or_entry) != 1:
            raise ValueError(
                "Expected either a single model spec containing keys like "
                "'graph_type', 'state', or 'param', or a one-model registry "
                "dictionary like {'LIF': {...}}. For multi-model loading, "
                "use load_from_dict(...)."
            )
        else:
            entry_model_name, model_spec = next(iter(spec_or_entry.items()))
            if model_name != entry_model_name:
                warnings.warn(
                    f"Mismatch: the dataclass model name {model_name}",
                    f"does not match the entry model name {entry_model_name}",
                    category=UserWarning)

        # Infer graph type
        graph_type = (
            graph_type
            or model_spec.get("graph_type")
            or self.infer_graph_type(model_cls)
        )

        # Create builder
        builder = self.register(model_cls, graph_type=graph_type)
        builder.load_from_dict(model_spec)

        return builder

    def load_from_dict(self, model_classes, model_registry):
        """Convenience method for loading many models."""
        for name, spec in model_registry.items():
            if name not in model_classes:
                raise KeyError(f"Model class not provided for model entry {name!r}")
            self.register_from_dict(model_classes[name], spec)

        return self
