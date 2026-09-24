"""
Tests for the core primitives: node/edge model dataclasses, shared-parameter
machinery, Node, Edge, Link.
"""
import pytest
import numpy as np

pytestmark = pytest.mark.core

from sango.model.base import LIF, PSP, IN, shared, get_shared_params
from sango.model.base import NodeModel
from sango.core import Node, Edge, Link
from dataclasses import dataclass


# ========================================================================
# Basic graph elements
# ========================================================================

class TestNode:
    def test_default_creation(self):
        n = Node()
        assert n.index is None
        assert n.name is None
        assert n.data == {}

    def test_getattr_missing_raises(self):
        n = Node()
        with pytest.raises(AttributeError):
            _ = n.nonexistent


class TestEdge:
    def test_default_creation(self):
        e = Edge()
        assert e.source_index is None
        assert e.target_index is None
        assert e.source_name is None
        assert e.target_name is None
        assert e.data == {}

    def test_getattr_missing_raises(self):
        e = Edge()
        with pytest.raises(AttributeError):
            _ = e.missing


class TestLink:
    def test_default_creation(self):
        lnk = Link()
        assert lnk.index is None
        assert lnk.link is None


# ========================================================================
# Model dataclasses
# ========================================================================

class TestModelDefaults:
    def test_lif_defaults(self, lif_model):
        assert lif_model.model == "LIF"
        assert lif_model.voltage == 0.0
        assert lif_model.threshold == 1.0
        assert lif_model.reset == 0.0
        assert lif_model.bias == 0.0
        assert lif_model.leak == 1.0

    def test_lif_custom_values(self):
        m = LIF(voltage=0.5, threshold=2.0, reset=-0.1, bias=0.1, leak=0.5)
        assert m.voltage == 0.5
        assert m.threshold == 2.0
        assert m.reset == -0.1
        assert m.bias == 0.1
        assert m.leak == 0.5

    def test_in_defaults(self, in_model):
        assert in_model.model == "IN"
        assert in_model.times == []
        
    def test_psp_defaults(self, psp_model):
        assert psp_model.model == "PSP"
        assert psp_model.delay == 1.0
        assert psp_model.weight == 1.0


# ========================================================================
# Shared parameters
# ========================================================================

class TestSharedParams:
    """The ``shared()`` marker and implicit string-field sharing via
    ``get_shared_params``."""

    def test_shared_field_metadata(self):
        @dataclass
        class TestModel(NodeModel):
            model: str = "TestModel"
            x: float = 1.0
            y: float = shared(2.0)

        m = TestModel()
        sp = get_shared_params(m)
        # 'model' is str-typed -> implicitly shared
        assert "model" in sp
        # 'y' is explicitly shared
        assert "y" in sp
        # 'x' is NOT shared
        assert "x" not in sp
