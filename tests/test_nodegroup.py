"""
Tests for the NodeGroup container: creation, defaults, set_values, setattr,
shared parameters, views, add_node, and path/property management.
"""
import pytest
import numpy as np

pytestmark = pytest.mark.core

from sango.model.base import LIF, PSP, IN, shared, get_shared_params
from sango.core import NodeGroup, Node
from dataclasses import dataclass


# ========================================================================
# NodeGroup Creation
# ========================================================================

class TestNodeGroupCreation:
    def test_default_size(self):
        ng = NodeGroup(LIF())
        assert len(ng) == 1
        assert ng.size == 1

    def test_explicit_size(self):
        ng = NodeGroup(LIF(), 5)
        assert len(ng) == 5
        assert ng.size == 5

    def test_invalid_model_raises(self):
        with pytest.raises(TypeError):
            NodeGroup("invalid", 3)

    def test_nodes_are_node_instances(self, small_nodegroup):
        for node in small_nodegroup:
            assert isinstance(node, Node)

    def test_indexes_are_sequential(self, small_nodegroup):
        for i, node in enumerate(small_nodegroup):
            assert node.index == i

    def test_constructor_kwargs(self):
        ng = NodeGroup(LIF(), 3, voltage=0.6, leak=[0.5, 0.4, 0.3])
        np.testing.assert_array_equal(ng.voltage, [0.6, 0.6, 0.6])
        np.testing.assert_array_almost_equal(ng.leak, [0.5, 0.4, 0.3])


# ========================================================================
# NodeGroup set_values
# ========================================================================

class TestNodeGroupSetValues:
    def test_bulk_scalar(self, small_nodegroup):
        small_nodegroup.set_values(voltage=0.5)
        np.testing.assert_array_equal(small_nodegroup.voltage, [0.5, 0.5, 0.5])

    def test_per_element_list(self, small_nodegroup):
        small_nodegroup.set_values(voltage=[0.1, 0.2, 0.3])
        np.testing.assert_array_equal(small_nodegroup.voltage, [0.1, 0.2, 0.3])

    def test_size_mismatch_raises(self, small_nodegroup):
        with pytest.raises(IndexError):
            small_nodegroup.set_values(voltage=[1.0, 2.0])  # wrong length

    def test_unknown_key_raises(self, small_nodegroup):
        with pytest.raises(KeyError):
            small_nodegroup.set_values(unknown=3)

    def test_setattr_broadcast(self):
        ng = NodeGroup(LIF(), 3)
        ng.voltage = 7.0
        np.testing.assert_array_equal(ng.voltage, [7.0, 7.0, 7.0])

    def test_setattr_list(self):
        ng = NodeGroup(LIF(), 2)
        ng.voltage = [1.0, 2.0]
        np.testing.assert_array_equal(ng.voltage, [1.0, 2.0])


# ========================================================================
# NodeGroup Shared Parameters
# ========================================================================

class TestNodeGroupSharedParams:
    def test_model_str_is_shared(self, small_nodegroup):
        # 'model' is str-typed so it's shared
        sp = small_nodegroup.shared_params
        assert "model" in sp

    def test_shared_field(self):
        @dataclass
        class Custom(LIF):
            model: str = "CustomLIF"
            extra: float = shared(0.1)

        ng = NodeGroup(Custom(), 4)
        sp = ng.shared_params
        assert "extra" in sp
        # All nodes reference the same underlying value
        assert ng[0].extra == ng[3].extra
        # Changing via set_values affects all
        ng.set_values(extra=0.5)
        assert ng[0].extra == 0.5


# ========================================================================
# NodeGroup Views
# ========================================================================

class TestNodeGroupViews:
    def test_node_view_reflects_group(self, small_nodegroup):
        small_nodegroup.set_values(voltage=[1.0, 2.0, 3.0])
        assert small_nodegroup[1].voltage == 2.0

    def test_node_write_reflects_group(self, small_nodegroup):
        small_nodegroup[0].voltage = 0.5
        assert small_nodegroup.voltage[0] == 0.5

    def test_node_write_threshold_reflects_group(self, small_nodegroup):
        small_nodegroup.set_values(threshold=[0.5, 0.6, 0.7])
        small_nodegroup[2].threshold = 0.9
        np.testing.assert_array_almost_equal(
            small_nodegroup.threshold, [0.5, 0.6, 0.9])


# ========================================================================
# NodeGroup add_node
# ========================================================================

class TestNodeGroupAddNode:
    def test_add_node_increases_size(self, small_nodegroup):
        small_nodegroup.add_node()
        assert len(small_nodegroup) == 4

    def test_add_node_with_kwargs(self):
        ng = NodeGroup(LIF(), 2)
        ng.add_node(voltage=5.0)
        assert ng[2].voltage == 5.0


# ========================================================================
# NodeGroup Path & Properties
# ========================================================================

class TestNodeGroupProperties:
    def test_set_path(self, small_nodegroup):
        small_nodegroup.set_path("neurons")
        assert small_nodegroup.path == "neurons"
        assert small_nodegroup[0].name == "neurons[0]"
        assert small_nodegroup[2].name == "neurons[2]"

    def test_getattr_unknown_raises(self, small_nodegroup):
        with pytest.raises(AttributeError):
            _ = small_nodegroup.nonexistent

