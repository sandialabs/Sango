"""
Tests for the lightweight LightDiGraph / LightMultiDiGraph classes
and the Topology.to_nx() method that produces them.

These mirror the structure of the existing TestTopologyToNx tests in
test_topology.py and the TestProcessGraph tests in test_backend.py,
ensuring the simple-graph pathway is a drop-in replacement for the
networkx pathway when used by the backend.
"""
import itertools
import warnings
import pytest

pytestmark = pytest.mark.network

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList
from sango.network import Network, Topology
from sango.lightgraph import LightDiGraph, LightMultiDiGraph

from .conftest import (
    Linear, build_simple_net, build_hierarchical_net,
    build_deeply_nested_net, build_complex_net, build_cyclic_net,
)


# ========================================================================
# LightDiGraph
# ========================================================================

class TestLightDiGraph:
    def test_empty_graph(self):
        g = LightDiGraph()
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0
        assert g.is_multigraph() is False

    def test_add_node(self):
        g = LightDiGraph()
        g.add_node("a", model="LIF", voltage=0.0)
        assert g.number_of_nodes() == 1
        assert "a" in g.nodes
        assert g.nodes["a"]["model"] == "LIF"
        
    def test_add_edge_overwrites(self):
        """Adding the same edge twice replaces data."""
        g = LightDiGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("a", "b", weight=2.0)
        assert g.number_of_edges() == 1

    def test_nodes_data_true(self):
        g = LightDiGraph()
        g.add_node("a", x=1)
        g.add_node("b", x=2)
        pairs = list(g.nodes(data=True))
        assert len(pairs) == 2
        names = [n for n, _ in pairs]
        assert "a" in names
        assert "b" in names

    def test_edges_data_true(self):
        g = LightDiGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=2.0)
        triples = list(g.edges(data=True))
        assert len(triples) == 2
        for src, tgt, data in triples:
            assert "weight" in data

    def test_edges_plain_iteration(self):
        g = LightDiGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        pairs = list(g.edges())
        assert pairs == [("a", "b"), ("b", "c")]


# ========================================================================
# LightMultiDiGraph
# ========================================================================

class TestLightMultiDiGraph:
    def test_empty_multigraph(self):
        g = LightMultiDiGraph()
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0
        assert g.is_multigraph() is True

    def test_parallel_edges(self):
        g = LightMultiDiGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("a", "b", weight=2.0)
        assert g.number_of_edges() == 2

    def test_edges_data_and_keys(self):
        g = LightMultiDiGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("a", "b", weight=2.0)
        quads = list(g.edges(data=True, keys=True))
        assert len(quads) == 2
        keys_seen = set()
        for src, tgt, key, data in quads:
            assert src == "a"
            assert tgt == "b"
            assert "weight" in data
            keys_seen.add(key)
        # Keys should be 0 and 1
        assert keys_seen == {0, 1}

    def test_edges_data_only(self):
        g = LightMultiDiGraph()
        g.add_edge("x", "y", w=3.0)
        triples = list(g.edges(data=True))
        assert len(triples) == 1
        src, tgt, data = triples[0]
        assert data["w"] == 3.0

    def test_edges_keys_only(self):
        g = LightMultiDiGraph()
        g.add_edge("x", "y")
        g.add_edge("x", "y")
        triples = list(g.edges(keys=True))
        assert len(triples) == 2
        for src, tgt, key in triples:
            assert isinstance(key, int)


# ========================================================================
# Topology.to_nx — flat networks
# ========================================================================

class TestLightToNxSimple:
    def test_graph_type(self, simple_net):
        g = simple_net._topology.to_nx(light=True)
        assert isinstance(g, LightDiGraph)

    def test_graph_structure(self, simple_net):
        g = simple_net._topology.to_nx(light=True)
        # 2 input + 3 layer = 5 nodes
        assert g.number_of_nodes() == 5
        # 2 * 3 = 6 edges
        assert g.number_of_edges() == 6

    def test_node_data_present(self, simple_net):
        g = simple_net._topology.to_nx(light=True)
        for _, data in g.nodes(data=True):
            assert "model" in data

    def test_edge_data_present(self, simple_net):
        g = simple_net._topology.to_nx(light=True)
        for _, _, data in g.edges(data=True):
            assert "model" in data

    def test_matches_to_nx(self, simple_net):
        """to_nx should produce the same nodes, edges, and data
        as the full networkx to_nx."""
        g_nx = simple_net._topology.to_nx()
        g_light = simple_net._topology.to_nx(light=True)
        assert g_light.number_of_nodes() == g_nx.number_of_nodes()
        assert g_light.number_of_edges() == g_nx.number_of_edges()
        # Check node names and data match
        nx_nodes = dict(g_nx.nodes(data=True))
        for name, data in g_light.nodes(data=True):
            assert name in nx_nodes
            assert data == nx_nodes[name]
        # Check edge data matches
        nx_edges = list(g_nx.edges(data=True))
        light_edges = list(g_light.edges(data=True))
        assert len(light_edges) == len(nx_edges)
        for (s1, t1, d1), (s2, t2, d2) in zip(nx_edges, light_edges):
            assert s1 == s2
            assert t1 == t2
            assert d1 == d2


# ========================================================================
# Topology.to_nx — hierarchical networks
# ========================================================================

class TestLightToNxHierarchical:
    def test_hierarchical_structure(self, hierarchical_net):
        g = hierarchical_net._topology.to_nx(light=True)
        # 3 input + 4 ff[0] + 2 ff[1] = 9
        assert g.number_of_nodes() == 9
        # 3*4 + 4*2 = 20
        assert g.number_of_edges() == 20

    def test_deeply_nested_structure(self, deeply_nested_net):
        g = deeply_nested_net._topology.to_nx(light=True)
        # 2 src + 3 inner = 5
        assert g.number_of_nodes() == 5
        # 2*3 = 6
        assert g.number_of_edges() == 6

    def test_hierarchical_matches_to_nx(self, hierarchical_net):
        g_nx = hierarchical_net._topology.to_nx()
        g_light = hierarchical_net._topology.to_nx(light=True)
        assert g_light.number_of_nodes() == g_nx.number_of_nodes()
        assert g_light.number_of_edges() == g_nx.number_of_edges()


# ========================================================================
# Topology.to_nx — multigraph handling
# ========================================================================

class TestLightToNxMulti:
    def test_parallel_edges_auto_upgrade(self):
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(
            net.layer, net.layer, PSP(),
            edges=[(0, 1), (0, 1)],
            weight=[1.0, 2.0],
        )
        net.build()
        with pytest.warns(UserWarning, match="Multi/parallel edges detected"):
            g = net._topology.to_nx(light=True)
        assert isinstance(g, LightMultiDiGraph)
        assert g.number_of_edges() == 2

    def test_multi_true_skips_warning(self):
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(
            net.layer, net.layer, PSP(),
            edges=[(0, 1), (0, 1)],
            weight=[1.0, 2.0],
        )
        net.build()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            g = net._topology.to_nx(multi=True, light=True)
        assert isinstance(g, LightMultiDiGraph)
        assert g.number_of_edges() == 2

    def test_multi_false_forces_digraph(self):
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(
            net.layer, net.layer, PSP(),
            edges=[(0, 1), (0, 1)],
            weight=[1.0, 2.0],
        )
        net.build()
        g = net._topology.to_nx(multi=False, light=True)
        assert isinstance(g, LightDiGraph)
        assert g.number_of_edges() == 1

    def test_separate_edgegroups_same_pair(self):
        net = Network()
        net.a = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(net.a, net.a, PSP(), edges=[(0, 1)])
        net.e2 = EdgeGroup(net.a, net.a, PSP(), edges=[(0, 1)])
        net.build()
        with pytest.warns(UserWarning, match="Multi/parallel edges detected"):
            g = net._topology.to_nx(light=True)
        assert isinstance(g, LightMultiDiGraph)
        assert g.number_of_edges() == 2

    def test_multi_edges_data_and_keys(self):
        """MultiDiGraph edges with data=True, keys=True should return quads."""
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(
            net.layer, net.layer, PSP(),
            edges=[(0, 1), (0, 1)],
            weight=[1.0, 2.0],
        )
        net.build()
        g = net._topology.to_nx(multi=True, light=True)
        quads = list(g.edges(data=True, keys=True))
        assert len(quads) == 2
        for src, tgt, key, data in quads:
            assert "model" in data
            assert isinstance(key, int)
