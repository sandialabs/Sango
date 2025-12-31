# General Imports
from dataclasses import dataclass, field

# Model classes
@dataclass
class Neuron:
    model: str = 'Neuron'

@dataclass
class Synapse:
    model: str = 'Synapse'

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
class IN(Neuron):
    model: str = 'IN'
    #times: object = None # supposed to be list
    times: list[float] = field(default_factory=list)