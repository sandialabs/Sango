# General Imports
from dataclasses import dataclass, field

from .base import LIF

@dataclass
class pLIF(LIF):
    model: str = 'pLIF'
    prob: float = 1.0 # probability of spiking
