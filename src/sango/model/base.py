# General Imports
from dataclasses import dataclass, field, fields

# Shared parameters across group
def shared(default):
    return field(default=default, metadata={'shared': True})

def get_shared_params(model):
    shared_set = {f.name for f in fields(model) if f.metadata.get('shared', False)}
    # str-typed fields are implicitly shared parameters
    shared_set |= {k for k, v in vars(model).items() if isinstance(v, str)}
    return shared_set
    
# Base model classes
@dataclass
class NodeModel:
    model: str = 'NodeModel'

@dataclass
class EdgeModel:
    model: str = 'EdgeModel'

@dataclass
class InputModel(NodeModel):
    model: str = 'InputModel'
    
@dataclass
class OutputModel(NodeModel):
    model: str = 'OutputModel'

# SNN generic classes
@dataclass
class Neuron(NodeModel):
    model: str = 'Neuron'

@dataclass
class Synapse(EdgeModel):
    model: str = 'Synapse'

@dataclass
class SpikeGen(InputModel):
    model: str = 'SpikeGen'

# SNN substrate models
@dataclass
class LIF(Neuron):
    model: str = 'LIF'
    voltage: float = 0.0
    threshold: float = 1.0
    reset: float = 0.0
    bias: float = 0.0
    leak: float = 1.0 # full leak

@dataclass
class PSP(Synapse):
    model: str = 'PSP'
    delay: float = 1.0
    weight: float = 1.0

@dataclass
class IN(SpikeGen):
    model: str = 'IN'
    times: list[float] = field(default_factory=list)