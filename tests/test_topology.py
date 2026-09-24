"""
Tests for the Topology data structure and TempPath helper.
"""
import itertools
import warnings
import pytest
import networkx as nx

pytestmark = pytest.mark.network

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList, Node
from sango.network import Network, Topology, TempPath
from .conftest import Linear


# ========================================================================
# Topology
# ========================================================================

class TestTopologyBasic:
    def test_empty_topology(self):
        t = Topology()
        assert vars(t) == {}

    def test_add_entries(self):
        t = Topology()
        ng = NodeGroup(LIF(), 2)
        t.add(layer=ng)
        assert t.layer is ng

    def test_nested_dict_becomes_topology(self):
        t = Topology(sub={"inner": NodeGroup(LIF(), 1)})
        assert isinstance(t.sub, Topology)
        assert isinstance(t.sub.inner, NodeGroup)

    def test_list_of_dicts(self):
        t = Topology(items=[{"a": NodeGroup(LIF(), 1)}, {"b": NodeGroup(LIF(), 2)}])
        assert isinstance(t.items[0], Topology)
        assert isinstance(t.items[1], Topology)

    def test_str_with_nodegroup(self, simple_net):
        """Topology __str__ should include built node/edge paths."""
        result = str(simple_net._topology)
        assert "inp" in result
        assert "layer" in result


# ========================================================================
# Topology path expansion
# ========================================================================

class TestTopologyPathExpansion:
    """Topology.expand_path() tokenisation of dotted and indexed paths."""

    def test_simple_path(self):
        assert Topology.expand_path("a.b.c") == ["a", "b", "c"]

    def test_indexed_path(self):
        assert Topology.expand_path("layer[0].neurons") == ["layer", 0, "neurons"]

    def test_multiple_indexes(self):
        assert Topology.expand_path("x[1][2]") == ["x", 1, 2]


# ========================================================================
# Topology attribute access
# ========================================================================

class TestTopologyAccess:
    def test_access_nodegroup(self):
        t = Topology()
        ng = NodeGroup(LIF(), 3)
        t.layer = ng
        assert t.access("layer") is ng

    def test_access_nested(self):
        t = Topology()
        sub = Topology()
        ng = NodeGroup(LIF(), 2)
        sub.neurons = ng
        t.sub = sub
        assert t.access("sub.neurons") is ng

    def test_access_indexed_list(self):
        t = Topology()
        ngs = [NodeGroup(LIF(), i + 1) for i in range(3)]
        t.layers = ngs
        assert t.access("layers[1]") is ngs[1]

    def test_access_invalid_path_raises(self):
        """Topology.access on a nonexistent attribute raises AttributeError."""
        t = Topology()
        t.layer = NodeGroup(LIF(), 3)
        with pytest.raises(AttributeError):
            t.access("nonexistent")

    def test_access_invalid_nested_path_raises(self):
        """Topology.access on a valid root but invalid leaf raises AttributeError."""
        t = Topology()
        sub = Topology()
        sub.neurons = NodeGroup(LIF(), 2)
        t.sub = sub
        with pytest.raises(AttributeError):
            t.access("sub.nonexistent")


# ========================================================================
# Topology bind
# ========================================================================

class TestTopologyBind:
    """Direct port binding through the Topology object. This is
    actually bypassed by the network bind."""

    def test_bind_port(self):
        ng = NodeGroup(LIF(), 3)
        ng.set_path("src")
        port = NodePort(3)
        t = Topology()
        t.src = ng
        t.port = port
        t.bind(ng, port)
        assert port.link is ng


# ========================================================================
# Topology → NetworkX conversion
# ========================================================================

class TestTopologyToNx:
    def test_graph_structure(self, simple_net):
        g = simple_net._topology.to_nx()
        assert isinstance(g, nx.DiGraph)
        # 2 input + 3 layer nodes
        assert g.number_of_nodes() == 5
        # 2 * 3 edges
        assert g.number_of_edges() == 6

    def test_node_data_present(self, simple_net):
        g = simple_net._topology.to_nx()
        for _, data in g.nodes(data=True):
            assert "model" in data

    def test_edge_data_present(self, simple_net):
        g = simple_net._topology.to_nx()
        for _, _, data in g.edges(data=True):
            assert "model" in data


# ========================================================================
# TempPath
# ========================================================================

class TestTempPath:
    """TempPath attribute chaining, indexing, and string representation."""

    def test_temppath_getattr(self):
        net = Network()
        tp = TempPath(net, net._topology, "foo")
        tp2 = tp.bar
        assert isinstance(tp2, TempPath)
        assert tp2.path == "foo.bar"

    def test_temppath_getitem(self):
        net = Network()
        tp = TempPath(net, net._topology, "foo")
        tp2 = tp[0]
        assert isinstance(tp2, TempPath)
        assert tp2.path == "foo[0]"


# ========================================================================
# TempPath delegation to child topology
# ========================================================================

class TestTempPathDelegation:
    """Before build, TempPath should delegate attribute access to a child
    network's already-defined topology attributes."""

    def test_temppath_resolves_child_topology_attr(self):
        """Accessing net.child.inp before build should return the actual
        NodePort (not another TempPath) because the child has it in _topology."""
        net = Network()
        child = Linear(4)
        net.ff = child
        result = net.ff.inp
        assert isinstance(result, NodePort)

    def test_temppath_deep_child_attr_falls_back_to_temppath(self):
        """Accessing a non-existent attribute on a child through TempPath
        should return a TempPath (not raise)."""
        net = Network()
        child = Linear(4)
        net.ff = child
        result = net.ff.nonexistent
        assert isinstance(result, TempPath)

    def test_temppath_child_delegation_with_list(self):
        """Child network lists: the children should have accessible ports."""
        net = Network()
        net.ff = [Linear(4), Linear(2)]
        child_0 = net._children["ff"][0]
        assert isinstance(child_0.inp, NodePort)
