"""
Tests for the Network wiring mechanics: __setattr__ routing, __init_subclass__
build wrapping, __getattr__ TempPath fallback, empty-list stashing, and
Network.access() traversal.

These cover the Python meta-protocol and wiring that takes place
*before* the build process runs, or that is orthogonal to it.
"""
import itertools
import warnings
import pytest
import networkx as nx
from functools import wraps

pytestmark = pytest.mark.network

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList, Node
from sango.core import Link
from sango.network import Network, Topology, TempPath

# Import helper networks from conftest
from .conftest import Linear, PassThrough, Inner, Middle


# ========================================================================
# Network.__setattr__ routing: plain attributes survive as regular attrs
# ========================================================================

class TestSetAttrRouting:
    """Verify that Network.__setattr__ correctly routes DSL objects to
    topology/children while leaving plain Python values as regular attributes."""

    def test_plain_int_is_regular_attr(self):
        net = Network()
        net.count = 42
        assert net.count == 42
        assert "count" not in vars(net._topology)

    def test_plain_list_is_regular_attr(self):
        """A list of plain values (not DSL objects) should be a regular attr."""
        net = Network()
        net.sizes = [3, 4, 2]
        assert net.sizes == [3, 4, 2]
        assert "sizes" not in vars(net._topology)

    def test_nodegroup_goes_to_topology(self):
        net = Network()
        ng = NodeGroup(LIF(), 3)
        net.layer = ng
        assert "layer" in vars(net._topology)
        assert net._topology.layer is ng

    def test_nodeport_goes_to_topology_and_creates_dependency(self):
        net = Network()
        p = NodePort()
        net.inp = p
        assert "inp" in vars(net._topology)
        assert "inp" in net._dependencies

    def test_sized_nodeport_no_dependency(self):
        net = Network()
        p = NodePort(3)
        net.inp = p
        assert "inp" in vars(net._topology)
        assert "inp" not in net._dependencies

    def test_network_goes_to_children(self):
        net = Network()
        child = Linear(4)
        net.ff = child
        assert "ff" in net._children
        assert child._parent is net

    def test_edgegroup_goes_to_topology(self):
        net = Network()
        ng = NodeGroup(LIF(), 2)
        net.layer = ng
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1)])
        net.dense = eg
        assert "dense" in vars(net._topology)

    def test_nodelist_goes_to_topology(self):
        net = Network()
        ng = NodeGroup(LIF(), 3)
        net.layer = ng
        nl = NodeList([ng[0], ng[2]])
        net.out = nl
        assert "out" in vars(net._topology)


# ========================================================================
# __init_subclass__ wrapped-build verification
# ========================================================================

class TestBuildWrapping:
    """The __init_subclass__ hook should wrap a subclass's build() so that
    recursive_build() is called automatically after the user build()."""

    def test_base_network_build(self):
        """Plain Network.build() should work (no subclass)."""
        net = Network()
        net.layer = NodeGroup(LIF(), 2)
        net.build()
        assert net._built is True

    def test_subclass_build_calls_recursive(self):
        """Calling build() on a subclass should set _built True
        (proving recursive_build ran)."""
        class Tiny(Network):
            def __init__(self):
                super().__init__()

            def build(self):
                self.layer = NodeGroup(LIF(), 2)

        t = Tiny()
        t.build()
        assert t._built is True

    def test_subclass_build_flattens_paths(self):
        """When a subclass is built as a top-level net, paths should be
        flattened (proving recursive_build -> flatten_paths ran)."""
        class Tiny(Network):
            def __init__(self):
                super().__init__()

            def build(self):
                self.layer = NodeGroup(LIF(), 2)

        t = Tiny()
        t.build()
        assert t.layer.path is not None
        assert t.layer[0].name is not None

    def test_user_build_runs_before_recursive(self):
        """The user's build code should execute before recursive_build.
        Verify by checking call order with a monkey-patched recursive_build."""
        call_order = []

        class Tracked(Network):
            def __init__(self):
                super().__init__()

            def build(self):
                call_order.append("user_build")
                self.layer = NodeGroup(LIF(), 2)

        original_rb = Network.recursive_build

        def patched_rb(self_net):
            call_order.append("recursive_build")
            return original_rb(self_net)

        Network.recursive_build = patched_rb
        try:
            t = Tracked()
            t.build()
        finally:
            Network.recursive_build = original_rb

        assert call_order == ["user_build", "recursive_build"]


# ========================================================================
# Network.__getattr__ TempPath fallback
# ========================================================================

class TestNetworkGetAttr:
    """Accessing non-existent attributes on a Network should return TempPaths
    (for DSL path construction), not raise AttributeError."""

    def test_nonexistent_attr_returns_temppath(self):
        net = Network()
        result = net.nonexistent
        assert isinstance(result, TempPath)
        assert result.path == "nonexistent"

    def test_temppath_chaining(self):
        net = Network()
        result = net.a.b.c
        assert isinstance(result, TempPath)
        assert result.path == "a.b.c"

    def test_existing_topology_attr_returned(self):
        """An attribute that exists in topology should be returned directly."""
        net = Network()
        ng = NodeGroup(LIF(), 3)
        net.layer = ng
        assert net.layer is ng

    def test_empty_list_attr_returned(self):
        """An empty list in _emptylists should be returned, not a TempPath."""
        net = Network()
        net.items = []
        result = net.items
        assert isinstance(result, list)
        assert result == []

    def test_private_attr_raises_attribute_error(self):
        """Accessing a non-existent _private attr should raise AttributeError."""
        net = Network()
        with pytest.raises(AttributeError):
            _ = net._nonexistent_internal


# ========================================================================
# Empty list stashing (_emptylists mechanism)
# ========================================================================

class TestEmptyLists:
    """Networks that initialise an empty list: both the case where it is
    populated during build() and the case where it stays empty."""

    # -- Populated during build -------------------------------------------

    @staticmethod
    def _build_layered(sizes=(3, 4, 2)):
        """Network whose layers list starts empty and is filled in build()."""
        import itertools as _it

        class LayeredNet(Network):
            def __init__(self, sizes):
                super().__init__()
                self._sizes = sizes
                self.layers = []           # triggers _emptylists stashing

            def build(self):
                for s in self._sizes:
                    self.layers.append(NodeGroup(LIF(), s))
                for i in range(len(self.layers) - 1):
                    src, tgt = self.layers[i], self.layers[i + 1]
                    edges = list(_it.product(range(src.size), range(tgt.size)))
                    setattr(self, f"dense_{i}",
                            EdgeGroup(src, tgt, PSP(), edges=edges))

        net = LayeredNet(sizes)
        net.build()
        return net

    def test_populated_build_succeeds(self):
        net = self._build_layered()
        assert net._built is True

    def test_populated_node_count(self):
        net = self._build_layered((3, 4, 2))
        g = net.graph()
        assert g.number_of_nodes() == 9  # 3 + 4 + 2

    def test_populated_edge_count(self):
        net = self._build_layered((3, 4, 2))
        g = net.graph()
        assert g.number_of_edges() == 20  # 3*4 + 4*2

    def test_populated_node_names_contain_list_indexes(self):
        net = self._build_layered((2, 3))
        g = net.graph()
        names = list(g.nodes)
        assert any("layers[0]" in n for n in names)
        assert any("layers[1]" in n for n in names)

    # -- Stays empty through build ----------------------------------------

    def test_permanently_empty_list_warns(self, capsys):
        class EmptyListNet(Network):
            def __init__(self):
                super().__init__()
                self.items = []

            def build(self):
                self.layer = NodeGroup(LIF(), 2)

        net = EmptyListNet()
        net.build()
        captured = capsys.readouterr()
        assert "warning" in captured.out.lower() and "empty list" in captured.out.lower()

    def test_permanently_empty_list_is_accessible(self):
        class EmptyListNet(Network):
            def __init__(self):
                super().__init__()
                self.items = []

            def build(self):
                self.layer = NodeGroup(LIF(), 2)

        net = EmptyListNet()
        net.build()
        assert net.items == []

    def test_permanently_empty_build_still_succeeds(self):
        class EmptyListNet(Network):
            def __init__(self):
                super().__init__()
                self.items = []

            def build(self):
                self.layer = NodeGroup(LIF(), 2)

        net = EmptyListNet()
        net.build()
        assert net._built is True


# ========================================================================
# Network.access() traversal
# ========================================================================

class TestNetworkAccess:
    """Network.access() should be able to traverse into child networks
    even when they aren't fully built yet."""

    def test_access_child_before_build(self):
        """access('ff') on a network with an unbuilt child should return
        the child Network object."""
        net = Network()
        child = Linear(4)
        net.ff = child
        result = net.access("ff")
        assert result is child

    def test_access_child_attr_before_build(self):
        """access('ff.inp') should traverse into the child and return
        its NodePort."""
        net = Network()
        child = Linear(4)
        net.ff = child
        result = net.access("ff.inp")
        assert isinstance(result, NodePort)

    def test_access_child_list_before_build(self):
        """access('ff') on a list of children should return the child list."""
        net = Network()
        net.ff = [Linear(4), Linear(2)]
        result = net.access("ff")
        assert isinstance(result, list)

    def test_access_after_build(self, simple_net):
        """After build, access should still work for topology paths."""
        result = simple_net.access("layer")
        assert isinstance(result, NodeGroup)
        assert result.size == 3

    def test_access_deep_path_after_build(self, hierarchical_net):
        """access('ff[0].layer') should return the child's NodeGroup."""
        result = hierarchical_net.access("ff[0].layer")
        assert isinstance(result, NodeGroup)
        assert result.size == 4
