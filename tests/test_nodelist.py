"""
Tests for the NodeList container: creation from groups, slices, path
management, and string representation.
"""
import pytest

pytestmark = pytest.mark.core

from sango.model.base import LIF
from sango.core import NodeGroup, NodeList


# ========================================================================
# NodeList
# ========================================================================

class TestNodeList:
    def test_empty_list(self):
        nl = NodeList()
        assert nl.size == 0

    def test_from_node_group(self, small_nodegroup):
        small_nodegroup.set_path("layer")
        nl = NodeList([small_nodegroup[0], small_nodegroup[2]])
        assert nl.size == 2

    def test_from_group_slice(self, small_nodegroup):
        """NodeList from a full-group slice (ng[:]) should contain all nodes."""
        small_nodegroup.set_path("layer")
        nl = NodeList(small_nodegroup[:])
        assert nl.size == 3
        for i in range(3):
            assert nl[i] == small_nodegroup[i]

    def test_from_partial_slice(self, small_nodegroup):
        """NodeList from a partial slice should contain the correct subset."""
        small_nodegroup.set_path("layer")
        nl = NodeList(small_nodegroup[1:])
        assert nl.size == 2
        assert nl[0] == small_nodegroup[1]
        assert nl[1] == small_nodegroup[2]

    def test_set_path(self):
        nl = NodeList()
        nl.set_path("output")
        assert nl.path == "output"

    def test_getattr_unknown_raises(self):
        nl = NodeList()
        with pytest.raises(AttributeError):
            _ = nl.blah


