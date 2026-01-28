# General Imports
from dataclasses import dataclass, field

from .base import LIF, PSP

@dataclass
class FuguLIF(LIF):
    model: str = 'FuguLIF'
    prob: float = 1.0 # probability of spiking

@dataclass
class FuguSyn(PSP):
    model: str = 'FuguSyn'