"""
Tests for the EdgeGroup container: creation, defaults, set_values, setattr,
shared parameters, views, access, edge_map, add_edge, and path/property
management.
"""
import pytest
import numpy as np

pytestmark = pytest.mark.core

from sango.model.base import LIF, PSP, shared
from sango.core import NodeGroup, EdgeGroup, Edge
from dataclasses import dataclass


# ========================================================================
# EdgeGroup Creation
# ========================================================================

class TestEdgeGroupCreation:
    def test_default_edge(self):
        ng = NodeGroup(LIF(), 2)
        eg = EdgeGroup(ng, ng, PSP())
        assert len(eg) == 1
        assert eg[0].source_index == 0
        assert eg[0].target_index == 0

    def test_explicit_edges(self, small_nodegroup):
        eg = EdgeGroup(
            small_nodegroup, small_nodegroup, PSP(),
            edges=[(0, 1), (1, 2), (2, 0)],
        )
        assert len(eg) == 3
        assert eg.edges == [(0, 1), (1, 2), (2, 0)]

    def test_invalid_model_raises(self, small_nodegroup):
        with pytest.raises(TypeError):
            EdgeGroup(small_nodegroup, small_nodegroup, "invalid", edges=[(0, 1)])

    def test_edges_are_edge_instances(self, small_edgegroup):
        for edge in small_edgegroup:
            assert isinstance(edge, Edge)

    def test_edge_typo_warning(self, small_nodegroup):
        """Passing 'edge' (singular) should warn, then raise KeyError."""
        with pytest.warns(SyntaxWarning, match="edge"):
            with pytest.raises(KeyError):
                EdgeGroup(small_nodegroup, small_nodegroup, PSP(),
                          edge=[(0, 1)])

    def test_constructor_kwargs(self):
        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1), (1, 2)], weight=[2.0, 4.0])
        np.testing.assert_array_almost_equal(eg.weight, [2.0, 4.0])


# ========================================================================
# EdgeGroup set_values
# ========================================================================

class TestEdgeGroupSetValues:
    def test_bulk_scalar(self, small_edgegroup):
        small_edgegroup.set_values(weight=2.5)
        np.testing.assert_array_equal(small_edgegroup.weight, [2.5, 2.5])

    def test_per_element(self, small_edgegroup):
        small_edgegroup.set_values(weight=[1.0, 3.0])
        np.testing.assert_array_almost_equal(small_edgegroup.weight, [1.0, 3.0])

    def test_size_mismatch_raises(self, small_edgegroup):
        with pytest.raises(IndexError):
            small_edgegroup.set_values(weight=[1.0])

    def test_unknown_key_raises(self, small_edgegroup):
        with pytest.raises(KeyError):
            small_edgegroup.set_values(unknown=3)

    def test_setattr_broadcast(self):
        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1), (1, 2)])
        eg.weight = 5.0
        np.testing.assert_array_equal(eg.weight, [5.0, 5.0])
        
    def test_setattr_list(self):
        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1), (1, 2)])
        eg.weight = [6.0, 7.0]
        np.testing.assert_array_equal(eg.weight, [6.0, 7.0])

# ========================================================================
# EdgeGroup Shared Parameters
# ========================================================================

class TestEdgeGroupSharedParams:
    def test_model_str_is_shared(self, small_edgegroup):
        # 'model' is str-typed so it's shared
        sp = small_edgegroup.shared_params
        assert "model" in sp

    def test_shared_field(self):
        @dataclass
        class CustomPSP(PSP):
            model: str = "CustomPSP"
            extra: float = shared(0.1)

        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, CustomPSP(), edges=[(0, 1), (1, 2)])
        sp = eg.shared_params
        assert "extra" in sp
        # All edges reference the same underlying value
        assert eg[0].extra == eg[1].extra
        # Changing via set_values affects all
        eg.set_values(extra=0.5)
        assert eg[0].extra == 0.5

# ========================================================================
# EdgeGroup Views
# ========================================================================

class TestEdgeGroupViews:
    def test_edge_view_reflects_group(self):
        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1), (1, 2)], weight=[1.0, 2.0])
        assert eg[0].weight == 1.0
        assert eg[1].weight == 2.0

    def test_edge_write_reflects_group(self):
        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1), (1, 2)], weight=[1.0, 2.0])
        eg[0].weight = 1.5
        np.testing.assert_array_almost_equal(eg.weight, [1.5, 2.0])

    def test_edge_write_delay_reflects_group(self):
        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1), (1, 2)], delay=[2.0, 3.0])
        eg[(1, 2)].delay = 5.0
        np.testing.assert_array_almost_equal(eg.delay, [2.0, 5.0])

# ========================================================================
# EdgeGroup Access
# ========================================================================

class TestEdgeGroupAccess:
    def test_tuple_access(self, small_edgegroup):
        edge = small_edgegroup[(0, 1)]
        assert edge.source_index == 0
        assert edge.target_index == 1

    def test_tuple_access_with_key(self, small_edgegroup):
        edge = small_edgegroup[(1, 2, 0)]
        assert edge.source_index == 1
        assert edge.target_index == 2

    def test_index_access(self, small_edgegroup):
        edge = small_edgegroup[0]
        assert isinstance(edge, Edge)

    def test_missing_tuple_raises(self, small_edgegroup):
        with pytest.raises(KeyError):
            _ = small_edgegroup[(9, 9)]

    def test_invalid_key_type_raises(self, small_edgegroup):
        with pytest.raises(TypeError):
            _ = small_edgegroup["invalid"]


# ========================================================================
# EdgeGroup edge_map
# ========================================================================

class TestEdgeGroupEdgeMap:
    def test_edge_map_keys(self, small_edgegroup):
        assert (0, 1, 0) in small_edgegroup.edge_map
        assert (1, 2, 0) in small_edgegroup.edge_map

    def test_parallel_edges_auto_key(self, small_nodegroup):
        eg = EdgeGroup(small_nodegroup, small_nodegroup, PSP(),
                       edges=[(0, 1), (0, 1)])
        assert (0, 1, 0) in eg.edge_map
        assert (0, 1, 1) in eg.edge_map

    def test_edge_map_values_correct(self, small_edgegroup):
        """edge_map values should be the linear index of each edge."""
        assert small_edgegroup.edge_map[(0, 1, 0)] == 0
        assert small_edgegroup.edge_map[(1, 2, 0)] == 1

    def test_edge_map_values_parallel(self, small_nodegroup):
        """Parallel edges should map to distinct consecutive indices."""
        eg = EdgeGroup(small_nodegroup, small_nodegroup, PSP(),
                       edges=[(0, 1), (0, 1), (0, 1)])
        assert eg.edge_map[(0, 1, 0)] == 0
        assert eg.edge_map[(0, 1, 1)] == 1
        assert eg.edge_map[(0, 1, 2)] == 2


# ========================================================================
# EdgeGroup add_edge
# ========================================================================

class TestEdgeGroupAddEdge:
    def test_add_edge_increases_size(self, small_edgegroup):
        small_edgegroup.add_edge(2, 0)
        assert len(small_edgegroup) == 3

    def test_add_edge_tuple(self, small_edgegroup):
        small_edgegroup.add_edge((2, 0))
        assert small_edgegroup[2].source_index == 2
        assert small_edgegroup[2].target_index == 0

    def test_add_edge_updates_edge_map(self, small_edgegroup):
        small_edgegroup.add_edge((2, 0))
        assert small_edgegroup.edge_map[(2, 0, 0)] == 2

    def test_add_edge_with_kwargs(self):
        ng = NodeGroup(LIF(), 3)
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1)])
        eg.add_edge(1, 2, weight=5.0)
        assert eg[1].weight == 5.0

    def test_add_edge_no_args_raises(self, small_edgegroup):
        with pytest.raises(TypeError):
            small_edgegroup.add_edge()

    def test_add_edge_too_many_args_raises(self, small_edgegroup):
        with pytest.raises(TypeError):
            small_edgegroup.add_edge(1, 2, 3)

# ========================================================================
# EdgeGroup Path & Properties
# ========================================================================

class TestEdgeGroupProperties:
    def test_set_path(self):
        ng = NodeGroup(LIF(), 3)
        ng.set_path("layer")
        eg = EdgeGroup(ng, ng, PSP(), edges=[(0, 1), (1, 2)])
        eg.set_path("synapse")
        assert eg.path == "synapse"

    def test_source_index_property(self, small_edgegroup):
        assert small_edgegroup.source_index == [0, 1]

    def test_target_index_property(self, small_edgegroup):
        assert small_edgegroup.target_index == [1, 2]

    def test_getattr_unknown_raises(self, small_edgegroup):
        with pytest.raises(AttributeError):
            _ = small_edgegroup.nonexistent

