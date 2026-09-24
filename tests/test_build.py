"""
Tests for the Network build process: recursive_build, dependency resolution,
port binding during build, double-binding errors, and orphan binding detection.
"""
import pytest

pytestmark = pytest.mark.network

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList
from sango.network import Network

# Import helper networks from conftest
from .conftest import Linear, PassThrough


# ========================================================================
# NodePort Resolution
# ========================================================================

class TestNetworkPorts:
    
    # -- basic port resolution ---------------------------------------------

    def test_port_binding_resolves_size(self):
        net = Network()
        net.src = NodeGroup(LIF(), 5)
        net.dst = Linear(3)
        net.bind(net.src, net.dst.inp)
        net.build()
        # The port should now know its size
        assert net.dst.inp.size == 5

    def test_set_portsize_on_nonexistent_path(self, capsys):
        """set_portsize on a bad path should print an error and not crash."""
        net = Network()
        net.set_portsize(net.nonexistent_port, 5)
        captured = capsys.readouterr()
        assert "error" in captured.out.lower() or "does not exist" in captured.out.lower()

    def test_set_portsize_on_non_port(self, capsys):
        """set_portsize on a NodeGroup (not a NodePort) should print an error."""
        net = Network()
        net.layer = NodeGroup(LIF(), 3)
        net.build()
        net.set_portsize(net.layer, 5)
        captured = capsys.readouterr()
        assert "not a nodeport" in captured.out.lower()


    # -- size mismatch -----------------------------------------------------

    def test_size_mismatch_prints_error(self, capsys):
        """Binding a source of size 5 to a port pre-sized to 3 prints an error."""
        net = Network()
        net.src = NodeGroup(LIF(), 5)
        net.dst = Linear(3)
        net.set_portsize(net.dst.inp, 3)
        net.bind(net.src, net.dst.inp)
        net.build()
        captured = capsys.readouterr()
        assert "size mismatch" in captured.out.lower()

    def test_size_mismatch_child_still_builds(self, capsys):
        """Despite the mismatch the child network builds with its original port size."""
        net = Network()
        net.src = NodeGroup(LIF(), 5)
        net.dst = Linear(3)
        net.set_portsize(net.dst.inp, 3)
        net.bind(net.src, net.dst.inp)
        net.build()
        _ = capsys.readouterr()
        assert net._children["dst"]._built is True
        assert net._children["dst"].inp.size == 3

    # -- Double-binding errors -------------------------------------------

    def test_already_linked_prints_error(self, capsys):
        """Binding a second source to an already-linked port should print an error."""
        net = Network()
        net.src1 = NodeGroup(LIF(), 3)
        net.src2 = NodeGroup(LIF(), 3)
        net.sub = PassThrough()
        net.bind(net.src1, net.sub.inp)
        net.bind(net.src2, net.sub.inp)
        net.build()
        captured = capsys.readouterr()
        assert "already linked" in captured.out.lower()

    def test_first_link_wins(self, capsys):
        """The port should keep the first binding, not the second."""
        net = Network()
        net.src1 = NodeGroup(LIF(), 3)
        net.src2 = NodeGroup(LIF(), 3)
        net.sub = PassThrough()
        net.bind(net.src1, net.sub.inp)
        net.bind(net.src2, net.sub.inp)
        net.build()
        _ = capsys.readouterr()
        assert net._children["sub"].inp.link is net.src1

    # -- Orphan bindings -------------------------------------------------

    def test_orphan_binding_raises_error(self, capsys):
        """Bindings that reference nonexistent paths should trigger an error."""
        net = Network()
        net.layer = NodeGroup(LIF(), 3)
        net.sub = PassThrough()
        # Bind to a port whose source path doesn't exist
        net.bind(net.nonexistent_source, net.sub.inp)
        # Give sub its size so it can actually build
        net.set_portsize(net.sub.inp, 3)
        with pytest.raises(ValueError):
            net.build()
        captured = capsys.readouterr()
        assert "bindings remaining" in captured.out.lower()


# ========================================================================
# List of NodePorts as dependencies
# ========================================================================

class TestNodePortList:
    @staticmethod
    def _build_multi_port():
        import itertools as _it

        class MultiPortNet(Network):
            def __init__(self, n_ports=3, layer_size=5):
                super().__init__()
                self._layer_size = layer_size
                self.ports = [NodePort() for _ in range(n_ports)]

            def build(self):
                self.layer = NodeGroup(LIF(), self._layer_size)
                for i, port in enumerate(self.ports):
                    edges = list(_it.product(range(port.size), range(self._layer_size)))
                    setattr(self, f"dense_{i}",
                            EdgeGroup(port, self.layer, PSP(), edges=edges))

        net = Network()
        net.sources = [NodeGroup(LIF(), 2), NodeGroup(LIF(), 3), NodeGroup(LIF(), 4)]
        net.sub = MultiPortNet(3, 5)
        for i, src in enumerate(net.sources):
            net.bind(src, net.sub.ports[i])
        net.build()
        return net

    def test_build_succeeds(self):
        net = self._build_multi_port()
        assert net._built is True
        assert net._children["sub"]._built is True

    def test_port_sizes_resolved(self):
        net = self._build_multi_port()
        child = net._children["sub"]
        assert child.ports[0].size == 2
        assert child.ports[1].size == 3
        assert child.ports[2].size == 4

    def test_graph_node_count(self):
        net = self._build_multi_port()
        g = net.graph()
        # 2 + 3 + 4 sources + 5 layer = 14
        assert g.number_of_nodes() == 14

    def test_graph_edge_count(self):
        net = self._build_multi_port()
        g = net.graph()
        # 2*5 + 3*5 + 4*5 = 45
        assert g.number_of_edges() == 45


# ========================================================================
# Recursive dependency resolution (3+ child chain)
# ========================================================================

class TestRecursiveBuild:
    """Build a network where child C depends on B, and B depends on A,
    requiring three separate passes through the build loop."""

    @staticmethod
    def _build_three_stage():
        """A -> B -> C chain, each a PassThrough that feeds the next."""
        net = Network()
        net.src = NodeGroup(LIF(), 3)
        net.a = PassThrough()
        net.b = PassThrough()
        net.c = PassThrough()
        net.bind(net.src, net.a.inp)
        net.bind(net.a.layer, net.b.inp)
        net.bind(net.b.layer, net.c.inp)
        net.build()
        return net

    def test_build_succeeds(self):
        net = self._build_three_stage()
        assert net._built is True

    def test_all_children_built(self):
        net = self._build_three_stage()
        for name in ("a", "b", "c"):
            assert net._children[name]._built is True, f"child '{name}' not built"

    def test_no_unresolved_dependencies(self):
        net = self._build_three_stage()
        for name in ("a", "b", "c"):
            assert not net._children[name]._dependencies

    def test_no_unresolved_bindings(self):
        net = self._build_three_stage()
        assert len(net._bindings) == 0

    def test_port_sizes_resolved(self):
        net = self._build_three_stage()
        assert net._children["a"].inp.size == 3
        assert net._children["b"].inp.size == 3
        assert net._children["c"].inp.size == 3

    def test_graph_node_count(self):
        net = self._build_three_stage()
        g = net.graph()
        # 3 src + 3 a.layer + 3 b.layer + 3 c.layer = 12
        assert g.number_of_nodes() == 12

    def test_graph_edge_count(self):
        net = self._build_three_stage()
        g = net.graph()
        # 3 1-to-1 edges per PassThrough * 3 = 9
        assert g.number_of_edges() == 9
