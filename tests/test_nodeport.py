"""
Tests for the NodePort container: creation, sizing, linking, and string
representation.
"""
import pytest

pytestmark = pytest.mark.core

from sango.model.base import LIF
from sango.core import NodeGroup, NodePort, NodeList, Node, Link


# ========================================================================
# NodePort
# ========================================================================

class TestNodePort:
    def test_default_unsized(self):
        p = NodePort()
        assert p.size is None
        assert len(p) == 0

    def test_sized(self):
        p = NodePort(3)
        assert p.size == 3
        assert len(p) == 3
        for item in p:
            assert isinstance(item, Link)

    def test_set_size(self):
        p = NodePort()
        p.set_size(5)
        assert p.size == 5
        assert len(p) == 5

    def test_resize_warns(self):
        p = NodePort(2)
        with pytest.warns(UserWarning, match="changing port size"):
            p.set_size(4)
        assert p.size == 4

    def test_set_path(self):
        p = NodePort(2)
        p.set_path("my_port")
        assert p.path == "my_port"

    def test_set_link(self):
        ng = NodeGroup(LIF(), 3)
        ng.set_path("neurons")
        p = NodePort(3)
        p.set_link(ng)
        assert p.link is ng
        for i, item in enumerate(p):
            assert item.link is ng[i]

    def test_set_link_resolves_node_data(self):
        """After set_link, each Link's .link should expose the source node's data."""
        ng = NodeGroup(LIF(), 3, voltage=[0.1, 0.2, 0.3])
        ng.set_path("neurons")
        p = NodePort(3)
        p.set_link(ng)
        for i in range(3):
            node = p[i].link
            assert isinstance(node, Node)
            assert node.voltage == pytest.approx([0.1, 0.2, 0.3][i])

    def test_set_link_with_nodelist_source(self):
        """NodePort can be linked to a NodeList (used during relink_ports)."""
        ng = NodeGroup(LIF(), 4)
        ng.set_path("layer")
        nl = NodeList([ng[0], ng[2], ng[3]])
        nl.set_path("selected")
        p = NodePort(3)
        p.set_link(nl)
        assert p.link is nl
        assert p[0].link is ng[0]
        assert p[1].link is ng[2]
        assert p[2].link is ng[3]

    def test_str_with_path_and_link(self):
        ng = NodeGroup(LIF(), 2)
        ng.set_path("neurons")
        p = NodePort(2)
        p.set_path("my_port")
        p.set_link(ng)
        result = str(p)
        assert "(port) my_port" in result
        assert "(no link)" not in result
