"""
Tests for the core primitives: node/edge model dataclasses, shared-parameter
machinery, Node (proxy), Edge (proxy), Link (proxy).
"""
import pytest
import numpy as np

pytestmark = pytest.mark.core

from sango.model.base import LIF, PSP, IN, shared, get_shared_params
from sango.model.base import NodeModel
from sango.core import Node, Edge, Link, NodeGroup, EdgeGroup, NodePort
from dataclasses import dataclass


# ========================================================================
# Basic graph elements (proxy classes)
# ========================================================================

class TestNode:
    def test_default_creation(self):
        n = Node()  # detached proxy (group=None)
        assert n._group is None
        assert n.index is None

    def test_getattr_missing_raises(self):
        ng = NodeGroup(LIF(), 2)
        n = ng[0]
        with pytest.raises(AttributeError):
            _ = n.missing

    def test_proxy_from_group(self):
        ng = NodeGroup(LIF(), 2)
        n = ng[1]
        assert isinstance(n, Node)
        assert n.index == 1
        assert n.data != {}  # has model keys


class TestEdge:
    def test_default_creation(self):
        e = Edge()  # detached proxy (group=None)
        assert e._group is None
        assert e._index is None

    def test_getattr_missing_raises(self):
        ng = NodeGroup(LIF(), 2)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1)])
        e = eg[0]
        with pytest.raises(AttributeError):
            _ = e.missing

    def test_proxy_from_group(self):
        ng = NodeGroup(LIF(), 2)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1)])
        e = eg[0]
        assert isinstance(e, Edge)
        assert e.source_index == 0
        assert e.target_index == 1
        assert e.source_name is None  # no path set yet


class TestLink:
    def test_default_creation(self):
        lnk = Link()  # detached proxy (port=None)
        assert lnk._port is None
        assert lnk.index is None
        assert lnk.link is None

    def test_proxy_from_port(self):
        p = NodePort(2)
        lnk = p[0]
        assert isinstance(lnk, Link)
        assert lnk.index == 0
        assert lnk.link is None  # no link set yet


# ========================================================================
# Proxy equality and hashing
# ========================================================================

class TestProxyEquality:
    """NodeProxy, EdgeProxy, LinkProxy __eq__ and __hash__."""

    def test_node_proxy_eq(self):
        ng = NodeGroup(LIF(), 3)
        assert ng[0] == ng[0]
        assert ng[0] != ng[1]
        assert ng[0] is not ng[0]  # different objects

    def test_node_proxy_hash(self):
        ng = NodeGroup(LIF(), 3)
        assert hash(ng[0]) == hash(ng[0])
        s = {ng[0], ng[0], ng[1]}
        assert len(s) == 2

    def test_edge_proxy_eq(self):
        ng = NodeGroup(LIF(), 2)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1)])
        assert eg[0] == eg[0]

    def test_edge_proxy_hash(self):
        ng = NodeGroup(LIF(), 2)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1)])
        assert hash(eg[0]) == hash(eg[0])

    def test_link_proxy_eq(self):
        p = NodePort(3)
        assert p[0] == p[0]
        assert p[0] != p[1]

    def test_link_proxy_hash(self):
        p = NodePort(3)
        assert hash(p[0]) == hash(p[0])

    def test_node_proxy_read_write(self):
        ng = NodeGroup(LIF(), 3, voltage=[1.0, 2.0, 3.0])
        assert ng[1].voltage == 2.0
        ng[1].voltage = 5.0
        assert ng[1].voltage == 5.0
        assert ng.voltage[1] == 5.0

    def test_node_proxy_hasattr_link(self):
        ng = NodeGroup(LIF(), 2)
        # Node proxies should NOT have a 'link' attribute
        assert not hasattr(ng[0], 'link')


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
