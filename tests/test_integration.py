"""
Tests for integration-level network topologies: hierarchical networks,
cyclic networks, deeply nested networks, and complex multi-port topologies.
"""
import itertools
import warnings
import pytest
import networkx as nx

pytestmark = pytest.mark.network

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList, Node
from sango.core import Link
from sango.network import Network, Topology, TempPath

# Import helper networks from conftest
from .conftest import (
    Linear, PassThrough,
    build_deeply_nested_net, build_complex_net, build_cyclic_net,
)


# ========================================================================
# Hierarchical Networks
# ========================================================================

class TestHierarchicalNetwork:
    """Two-level hierarchical network: build flag, child registration,
    graph node/edge counts, structure output, and hierarchical name paths."""

    def test_build_succeeds(self, hierarchical_net):
        assert hierarchical_net._built is True

    def test_children_registered(self, hierarchical_net):
        assert "ff" in hierarchical_net._children

    def test_child_networks_built(self, hierarchical_net):
        for child in hierarchical_net._children["ff"]:
            assert child._built is True

    def test_graph_has_all_nodes(self, hierarchical_net):
        g = hierarchical_net.graph()
        # 3 input + 4 layer[0] + 2 layer[1]
        assert g.number_of_nodes() == 9

    def test_graph_edges(self, hierarchical_net):
        g = hierarchical_net.graph()
        # 3*4 + 4*2 = 20
        assert g.number_of_edges() == 20

    def test_structure_has_children(self, hierarchical_net):
        s = hierarchical_net.structure()
        assert "child" in s
        assert "ff" in s["child"]
        assert isinstance(s["child"]["ff"], list)
        assert len(s["child"]["ff"]) == 2

    def test_node_names_hierarchical(self, hierarchical_net):
        g = hierarchical_net.graph()
        names = list(g.nodes)
        # Should contain paths like ff[0].layer[0], ff[1].layer[1]
        assert any("ff[0].layer" in n for n in names)
        assert any("ff[1].layer" in n for n in names)

    def test_child_net_path(self, hierarchical_net):
        child = hierarchical_net._children["ff"][0]
        assert "ff[0]" in child.net_path()


# ========================================================================
# Cyclic Networks
# ========================================================================

class TestCyclicNetwork:
    """Tests for the cyclic-binding machinery (set_portsize, feedback loops)."""

    # -- build correctness -------------------------------------------------

    def test_build_succeeds(self, cyclic_net):
        assert cyclic_net._built is True

    def test_children_built(self, cyclic_net):
        assert cyclic_net._children["a"]._built is True
        assert cyclic_net._children["b"]._built is True

    def test_no_unresolved_bindings(self, cyclic_net):
        assert len(cyclic_net._bindings) == 0

    def test_no_unresolved_dependencies(self, cyclic_net):
        assert not cyclic_net._children["a"]._dependencies
        assert not cyclic_net._children["b"]._dependencies

    def test_unresolvable_cycle_prints_error(self, capsys):
        """Without set_portsize the dependency cycle cannot be resolved."""
        net = Network()
        net.a = PassThrough()
        net.b = PassThrough()
        net.bind(net.a.layer, net.b.inp)
        net.bind(net.b.layer, net.a.inp)
        net.build()
        captured = capsys.readouterr()
        assert "unable to resolve dependencies" in captured.out.lower()

    def test_unresolvable_cycle_leaves_children_unbuilt(self, capsys):
        """Without set_portsize, the parent marks itself built but children
        remain unbuilt because the dependency cycle cannot be broken."""
        net = Network()
        net.a = PassThrough()
        net.b = PassThrough()
        net.bind(net.a.layer, net.b.inp)
        net.bind(net.b.layer, net.a.inp)
        net.build()
        _ = capsys.readouterr()
        assert net._built is True
        assert net._children["a"]._built is False
        assert net._children["b"]._built is False

    # -- graph structure ---------------------------------------------------

    def test_graph_node_count(self):
        """Each child has *size* nodes -> 2 * size total."""
        net = build_cyclic_net(size=3)
        g = net.graph()
        assert g.number_of_nodes() == 6

    def test_graph_edge_count(self):
        """Each child has *size* 1-to-1 edges -> 2 * size total."""
        net = build_cyclic_net(size=3)
        g = net.graph()
        assert g.number_of_edges() == 6

    def test_graph_contains_cycle(self, cyclic_net):
        """The nx DiGraph must contain at least one directed cycle."""
        g = cyclic_net.graph()
        assert nx.is_directed_acyclic_graph(g) is False

    # -- port wiring -------------------------------------------------------

    def test_port_sizes_resolved(self):
        """Both ports should acquire the size given to set_portsize."""
        net = build_cyclic_net(size=4)
        assert net._children["a"].inp.size == 4
        assert net._children["b"].inp.size == 4

    def test_port_links_are_cross_bound(self, cyclic_net):
        """a.inp should link to b.layer and vice-versa."""
        a_link = cyclic_net._children["a"].inp.link
        b_link = cyclic_net._children["b"].inp.link
        assert a_link is not None
        assert b_link is not None
        assert "b.layer" in (a_link.path or "")
        assert "a.layer" in (b_link.path or "")

    # -- parameterised size ------------------------------------------------

    @pytest.mark.parametrize("size", [1, 5, 10])
    def test_various_port_sizes(self, size):
        """Cycles should work for different port widths."""
        net = build_cyclic_net(size=size)
        g = net.graph()
        assert g.number_of_nodes() == 2 * size
        assert g.number_of_edges() == 2 * size


# ========================================================================
# Deeply nested networks (3+ levels)
# ========================================================================

class TestDeeplyNestedNetwork:
    """Three-level nesting: root -> Middle -> Inner, where Inner has a NodePort."""

    def test_build_succeeds(self, deeply_nested_net):
        assert deeply_nested_net._built is True

    def test_all_levels_built(self, deeply_nested_net):
        mid = deeply_nested_net._children["mid"]
        inner = mid._children["inner"]
        assert mid._built is True
        assert inner._built is True

    def test_net_paths(self, deeply_nested_net):
        mid = deeply_nested_net._children["mid"]
        inner = mid._children["inner"]
        assert "mid." in mid.net_path()
        assert "mid.inner." in inner.net_path()

    def test_graph_node_names(self, deeply_nested_net):
        g = deeply_nested_net.graph()
        names = list(g.nodes)
        assert any("mid.inner.layer" in n for n in names)

    def test_graph_structure(self, deeply_nested_net):
        g = deeply_nested_net.graph()
        assert g.number_of_nodes() == 5   # 2 src + 3 inner layer
        assert g.number_of_edges() == 6   # 2 * 3


# ========================================================================
# Complex Network (dual-port hierarchical with NodeList)
# ========================================================================

class TestComplexNetwork:
    """Tests for a network with hierarchical children, multiple NodePorts,
    and a NodeList mixing port-links and direct nodes."""

    def test_build_succeeds(self, complex_net):
        assert complex_net._built is True

    def test_child_built(self, complex_net):
        assert complex_net._children["sub"]._built is True

    def test_graph_node_count(self, complex_net):
        g = complex_net.graph()
        # 2 src_a + 3 src_b + 4 sub.layer = 9
        assert g.number_of_nodes() == 9

    def test_graph_edge_count(self, complex_net):
        g = complex_net.graph()
        # 2*4 + 3*4 = 20
        assert g.number_of_edges() == 20

    def test_flattened_nodelist(self, complex_net):
        # NodeList should have 4 entries
        out = complex_net._topology.out
        assert out.size == 4
        # Every entry should be resolved to a Node
        for item in out:
            assert isinstance(item, Node), f"Expected Node, got {type(item)}"

    def test_structure_has_child(self, complex_net):
        s = complex_net.structure()
        assert "child" in s
        assert "sub" in s["child"]
