"""
Tests for the Network post-build behaviour: graph generation, multigraph
support, NodeList resolution, flatten operations, and relink_ports.
"""
import warnings
import pytest
import networkx as nx

pytestmark = pytest.mark.network

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList, Node
from sango.core import Link
from sango.network import Network, Topology, TempPath

# Import helper networks from conftest
from .conftest import Linear, PassThrough


# ========================================================================
# Simple Network (post-build behaviour)
# ========================================================================

class TestSimpleNetwork:
    """Basic post-build smoke tests: built flag, node/edge names, graph
    caching, graph refresh, and structure output."""

    def test_build_sets_built(self, simple_net):
        assert simple_net._built is True

    def test_node_names_populated(self, simple_net):
        g = simple_net.graph()
        for node_name in g.nodes:
            assert "[" in node_name  # e.g. "inp[0]", "layer[2]"

    def test_edge_names_populated(self, simple_net):
        g = simple_net.graph()
        for src, tgt in g.edges:
            assert src is not None
            assert tgt is not None

    def test_graph_cached(self, simple_net):
        g1 = simple_net.graph()
        g2 = simple_net.graph()
        assert g1 is g2

    def test_graph_update(self, simple_net):
        g1 = simple_net.graph()
        g2 = simple_net.graph(update=True)
        assert g1 is not g2

    def test_structure_contains_class_and_param(self, simple_net):
        s = simple_net.structure()
        assert "class" in s
        assert "param" in s


# ========================================================================
# MultiGraph Networks (parallel / duplicate edges)
# ========================================================================

class TestMultiGraphNetwork:
    """Verify automatic MultiDiGraph upgrade when parallel edges are present,
    the ``multi`` kwarg override, and the upgrade warning behaviour."""

    def test_parallel_edges_create_multigraph(self):
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(
            net.layer, net.layer, PSP(),
            edges=[(0, 1), (0, 1)],  # parallel
            weight=[1.0, 2.0],
        )
        net.build()
        with pytest.warns(UserWarning, match="Multi/parallel edges detected"):
            g = net.graph()
        assert isinstance(g, nx.MultiDiGraph)
        assert g.number_of_edges() == 2

    def test_separate_edgegroups_same_pair(self):
        net = Network()
        net.a = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(net.a, net.a, PSP(), edges=[(0, 1)])
        net.e2 = EdgeGroup(net.a, net.a, PSP(), edges=[(0, 1)])
        net.build()
        with pytest.warns(UserWarning, match="Multi/parallel edges detected"):
            g = net.graph()
        assert isinstance(g, nx.MultiDiGraph)
        assert g.number_of_edges() == 2

    def test_multi_true_skips_upgrade_warning(self):
        """Passing multi=True to to_nx() should create a MultiDiGraph
        directly without triggering the upgrade warning."""
        import warnings as _warnings
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(
            net.layer, net.layer, PSP(),
            edges=[(0, 1), (0, 1)],  # parallel
            weight=[1.0, 2.0],
        )
        net.build()
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            g = net._topology.to_nx(multi=True)
        assert isinstance(g, nx.MultiDiGraph)
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 2

    def test_multi_false_forces_digraph(self):
        """Passing multi=False on a topology with parallel edges should
        force a plain DiGraph (collapsing parallel edges)."""
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(
            net.layer, net.layer, PSP(),
            edges=[(0, 1), (0, 1)],
            weight=[1.0, 2.0],
        )
        net.build()
        g = net._topology.to_nx(multi=False)
        assert isinstance(g, nx.DiGraph)
        assert not isinstance(g, nx.MultiDiGraph)


# ========================================================================
# Network NodeList resolution
# ========================================================================

class TestNetworkNodeList:
    """NodeLists built from NodeGroup entries, NodePort links, other
    NodeLists, or mixed sources should resolve to concrete Nodes after
    build."""

    def test_nodelist_size(self):
        net = Network()
        net.layer = NodeGroup(LIF(), 4)
        net.out = NodeList([net.layer[0], net.layer[3]])
        net.build()
        assert net.out.size == 2

    def test_nodelist_from_port_links(self):
        """Links from a pre-sized NodePort should resolve to Nodes after build."""
        net = Network()
        net.src = NodeGroup(LIF(), 3)
        net.sub = Linear(3)
        net.bind(net.src, net.sub.inp)
        # set_portsize so the port is indexable before build
        net.set_portsize(net.sub.inp, 3)
        net.out = NodeList([net.sub.inp[0], net.sub.inp[2]])
        net.build()
        assert net.out.size == 2
        # After flattening, every entry should be a Node, not a Link
        for item in net.out:
            assert isinstance(item, Node)

    def test_nodelist_from_another_nodelist(self):
        """A NodeList built from entries of another NodeList should flatten."""
        net = Network()
        net.layer = NodeGroup(LIF(), 5)
        net.list_a = NodeList([net.layer[0], net.layer[1], net.layer[2]])
        net.list_b = NodeList([net.list_a[0], net.list_a[2]])
        net.build()
        assert net.list_a.size == 3
        assert net.list_b.size == 2
        for item in net.list_b:
            assert isinstance(item, Node)

    def test_nodelist_mixed_sources(self):
        """NodeList mixing NodeGroup nodes and NodePort links."""
        net = Network()
        net.src = NodeGroup(LIF(), 3)
        net.sub = Linear(3)
        net.bind(net.src, net.sub.inp)
        net.set_portsize(net.sub.inp, 3)
        # One entry from a NodeGroup, one from a NodePort
        net.out = NodeList([net.src[1], net.sub.inp[2]])
        net.build()
        assert net.out.size == 2
        for item in net.out:
            assert isinstance(item, Node)

    def test_nodelist_port_links_resolve_to_correct_nodes(self):
        """Port links should resolve to the bound source nodes."""
        net = Network()
        net.src = NodeGroup(LIF(), 4)
        net.sub = Linear(4)
        net.bind(net.src, net.sub.inp)
        net.set_portsize(net.sub.inp, 4)
        net.out = NodeList([net.sub.inp[0], net.sub.inp[3]])
        net.build()
        # The port was bound to net.src, so the resolved nodes should
        # be the same objects that back net.src[0] and net.src[3].
        assert net.out[0] == net.src[0]
        assert net.out[1] == net.src[3]


# ========================================================================
# Flatten EdgeGroups (TempPath → resolved source/target)
# ========================================================================

class TestFlattenEdgeGroups:
    """After build, EdgeGroup source/target should be resolved objects
    (NodeGroup/NodePort/NodeList), not TempPaths."""

    def test_edge_source_is_correct_type(self, simple_net):
        """In a flat network, edge source/target should remain NodeGroups."""
        eg = simple_net._topology.dense
        assert isinstance(eg.source, NodeGroup)
        assert isinstance(eg.target, NodeGroup)

    def test_hierarchical_edge_sources_resolved(self, hierarchical_net):
        """In a hierarchical net, child EdgeGroup sources come from TempPath
        bindings.  After build they should be resolved."""
        topo = hierarchical_net._topology
        eg = topo.ff[0].dense
        assert isinstance(eg, EdgeGroup)
        assert not isinstance(eg.source, TempPath), "EdgeGroup source is still a TempPath"
        assert not isinstance(eg.target, TempPath), "EdgeGroup target is still a TempPath"

    def test_deeply_nested_edge_sources_resolved(self, deeply_nested_net):
        """Three-level nesting: edge sources in Inner should be resolved."""
        topo = deeply_nested_net._topology
        eg = topo.mid.inner.dense
        assert isinstance(eg, EdgeGroup)
        assert not isinstance(eg.source, TempPath)
        assert not isinstance(eg.target, TempPath)

    def test_edge_source_names_populated(self, hierarchical_net):
        """After flattening, individual Edge objects should have source/target names."""
        topo = hierarchical_net._topology
        eg = topo.ff[0].dense
        for edge in eg:
            assert edge.source_name is not None
            assert edge.target_name is not None
            assert "[" in edge.source_name
            assert "[" in edge.target_name


# ========================================================================
# Flatten NodeLists (multi-level indirection)
# ========================================================================

class TestFlattenNodeLists:
    """NodeList -> NodeList -> Node chains require multiple flattening passes."""

    def test_three_level_chain_resolves_to_nodes(self):
        """list_c -> list_b -> list_a -> NodeGroup entries."""
        net = Network()
        net.layer = NodeGroup(LIF(), 5)
        net.list_a = NodeList([net.layer[0], net.layer[1], net.layer[2]])
        net.list_b = NodeList([net.list_a[0], net.list_a[2]])
        net.list_c = NodeList([net.list_b[0], net.list_b[1]])
        net.build()
        for item in net.list_c:
            assert isinstance(item, Node), f"Expected Node, got {type(item)}"

    def test_three_level_resolves_correct_nodes(self):
        """Verify the chain resolves to the correct underlying nodes."""
        net = Network()
        net.layer = NodeGroup(LIF(), 5)
        net.list_a = NodeList([net.layer[0], net.layer[1], net.layer[2]])
        net.list_b = NodeList([net.list_a[0], net.list_a[2]])
        net.list_c = NodeList([net.list_b[0], net.list_b[1]])
        net.build()
        # list_c[0] -> list_b[0] -> list_a[0] -> layer[0]
        assert net.list_c[0] == net.layer[0]
        # list_c[1] -> list_b[1] -> list_a[2] -> layer[2]
        assert net.list_c[1] == net.layer[2]

    def test_four_level_chain_resolves_to_nodes(self):
        """Four levels of indirection."""
        net = Network()
        net.layer = NodeGroup(LIF(), 4)
        net.la = NodeList([net.layer[0], net.layer[1], net.layer[2], net.layer[3]])
        net.lb = NodeList([net.la[3], net.la[1]])
        net.lc = NodeList([net.lb[0], net.lb[1]])
        net.ld = NodeList([net.lc[1]])
        net.build()
        # ld[0] -> lc[1] -> lb[1] -> la[1] -> layer[1]
        assert isinstance(net.ld[0], Node)
        assert net.ld[0] == net.layer[1]


# ========================================================================
# Relink Ports (Link targets after NodeList binding)
# ========================================================================

class TestRelinkPorts:
    """After build, a NodePort whose link is a NodeList should have its
    individual Link objects pointing to the resolved Nodes."""

    def test_port_links_resolve_through_nodelist(self):
        """Create a NodeList and bind it to a child's NodePort.
        After build, the port's Links should point to the correct Nodes."""
        net = Network()
        net.src = NodeGroup(LIF(), 4)
        net.selected = NodeList([net.src[1], net.src[3]])
        net.sub = PassThrough()
        net.bind(net.selected, net.sub.inp)
        net.build()
        port = net._children["sub"].inp
        assert port.size == 2
        for item in port:
            assert isinstance(item, Link)
            assert isinstance(item.link, Node)
        assert port[0].link == net.src[1]
        assert port[1].link == net.src[3]

    def test_relink_ports_after_complex_build(self, complex_net):
        """In the complex network, the DualPort's inp_a port is linked.
        Verify Link objects point to real Nodes."""
        sub_child = complex_net._children["sub"]
        port_a = sub_child.inp_a
        assert port_a.link is not None
        for item in port_a:
            assert isinstance(item.link, Node)
