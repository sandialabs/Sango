"""
Tests for round-trip serialization: topology/network to dict, JSON, and pickle.
"""
import json
import itertools
import tempfile
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.serialization

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList, Node
from sango.network import Network, Topology
from sango.serialization import (
    save, load,
    topology_to_dict, topology_from_dict,
    network_to_dict, network_from_dict,
    statedict_to_json, statedict_from_json,
    _numpy_to_python, _is_state_dict,
)

from .conftest import (
    Linear, SpikeInput,
    build_simple_net, build_hierarchical_net, build_deeply_nested_net,
    build_complex_net, build_cyclic_net
)
from .test_subtractor import _build_subtractor_net


# ========================================================================
# Helpers
# ========================================================================

def _build_simple_net_with_nodelist():
    """Simple net with an output NodeList (for serialization tests)."""
    return build_simple_net(node_list=True)


# ========================================================================
# _numpy_to_python
# ========================================================================

class TestNumpyToPython:
    """Convert numpy scalars, arrays, and bools to native Python types;
    non-numpy values should pass through unchanged."""

    def test_ndarray(self):
        assert _numpy_to_python(np.array([1, 2, 3])) == [1, 2, 3]

    def test_integer(self):
        assert _numpy_to_python(np.int64(5)) == 5
        assert isinstance(_numpy_to_python(np.int64(5)), int)

    def test_floating(self):
        assert _numpy_to_python(np.float32(1.5)) == pytest.approx(1.5)
        assert isinstance(_numpy_to_python(np.float64(1.5)), float)

    def test_bool(self):
        assert _numpy_to_python(np.bool_(True)) is True


# ========================================================================
# _is_state_dict
# ========================================================================

class TestIsStateDict:
    """Validate the ``_is_state_dict`` heuristic: dict with string keys
    and array/list values is valid; empty, non-dict, or wrong types are not."""

    def test_valid_state_dict(self):
        sd = {"a": np.array([1.0]), "b": np.array([2.0])}
        assert _is_state_dict(sd) is True

    def test_empty_dict_is_invalid(self):
        assert _is_state_dict({}) is False

    def test_non_dict_is_invalid(self):
        assert _is_state_dict([1, 2]) is False

    def test_non_string_keys_invalid(self):
        assert _is_state_dict({1: np.array([0.0])}) is False

    def test_non_array_values_invalid(self):
        assert _is_state_dict({"a": 10}) is False

    def test_list_values_accepted(self):
        assert _is_state_dict({"a": [1, 2, 3]}) is True


# ========================================================================
# state_dict JSON round-trip
# ========================================================================

class TestStateDictJson:
    """JSON serialisation round-trip for state dicts, including dtype
    preservation."""

    def test_round_trip(self):
        sd = {"layer.voltage": np.array([0.1, 0.2, 0.3]),
              "layer.threshold": np.array([1.0, 1.0, 1.0])}
        j = statedict_to_json(sd)
        assert j["__type__"] == "state_dict"
        sd2 = statedict_from_json(j)
        for k in sd:
            np.testing.assert_array_almost_equal(sd[k], sd2[k])

    def test_dtype_preserved(self):
        sd = {"x": np.array([1, 2], dtype=np.int32)}
        j = statedict_to_json(sd)
        sd2 = statedict_from_json(j)
        assert sd2["x"].dtype == np.int32


# ========================================================================
# state_dict round-trip (load / strict / shape / connectivity)
# ========================================================================

class TestTopologyStateDict:
    """state_dict extraction, load_state_dict round-trip, strict/non-strict
    mode, shape-mismatch, connectivity-mismatch, and hierarchical key
    coverage."""

    def test_state_dict_round_trip(self, simple_net):
        sd = simple_net.state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0
        # All values should be numpy arrays
        for v in sd.values():
            assert isinstance(v, np.ndarray)
        # Modify a parameter, save, reload, and confirm round-trip
        param_key = next(k for k in sd if not k.endswith(("._source_index", "._target_index")))
        original = sd[param_key].copy()
        sd[param_key] = original + 1.0
        simple_net.load_state_dict(sd)
        sd2 = simple_net.state_dict()
        np.testing.assert_array_almost_equal(sd2[param_key], original + 1.0)

    def test_load_state_dict_strict(self, simple_net):
        sd = simple_net.state_dict()
        # Modify a value
        first_key = next(k for k in sd if not k.endswith(("._source_index", "._target_index")))
        sd[first_key] = sd[first_key] + 1.0
        simple_net.load_state_dict(sd)
        new_sd = simple_net.state_dict()
        np.testing.assert_array_equal(new_sd[first_key], sd[first_key])

    def test_load_state_dict_missing_key_raises(self, simple_net):
        sd = simple_net.state_dict()
        del sd[list(sd.keys())[0]]
        with pytest.raises(KeyError, match="Missing"):
            simple_net.load_state_dict(sd, strict=True)

    def test_load_state_dict_unexpected_key_raises(self, simple_net):
        sd = simple_net.state_dict()
        sd["extra_key"] = np.array([0.0])
        with pytest.raises(KeyError, match="Unexpected"):
            simple_net.load_state_dict(sd, strict=True)

    def test_load_state_dict_non_strict(self, simple_net):
        sd = simple_net.state_dict()
        sd["extra_key"] = np.array([0.0])
        # Should not raise
        simple_net.load_state_dict(sd, strict=False)

    def test_load_state_dict_shape_mismatch(self, simple_net):
        sd = simple_net.state_dict()
        first_key = next(k for k in sd if not k.endswith(("._source_index", "._target_index")))
        sd[first_key] = np.array([0.0])  # wrong shape
        with pytest.raises(ValueError, match="Shape mismatch"):
            simple_net.load_state_dict(sd)

    def test_load_state_dict_connectivity_mismatch(self, simple_net):
        sd = simple_net.state_dict()
        idx_key = next(k for k in sd if k.endswith("._source_index"))
        sd[idx_key] = sd[idx_key] + 100
        with pytest.raises(ValueError, match="Structure mismatch"):
            simple_net.load_state_dict(sd)

    def test_hierarchical_contains_child_edge_keys(self):
        """state_dict of a hierarchical network should include edge params
        from child sub-networks (e.g. ff[0].dense.weight)."""
        net = build_hierarchical_net()
        sd = net.state_dict()
        edge_keys = [k for k in sd if "dense" in k and not k.endswith(("._source_index", "._target_index"))]
        assert len(edge_keys) > 0, "No edge parameter keys found from child networks"

    def test_hierarchical_contains_child_connectivity_keys(self):
        """state_dict should include _source_index/_target_index for child edges."""
        net = build_hierarchical_net()
        sd = net.state_dict()
        src_keys = [k for k in sd if k.endswith("._source_index")]
        tgt_keys = [k for k in sd if k.endswith("._target_index")]
        # Two child networks (ff[0] and ff[1]) each with a dense EdgeGroup
        assert len(src_keys) >= 2
        assert len(tgt_keys) >= 2


# ========================================================================
# topology_to_dict / topology_from_dict
# ========================================================================

class TestTopologySerialization:
    """topology_to_dict / topology_from_dict round-trip: NodeGroup,
    EdgeGroup, NodePort, NodeList, and parameter-value preservation."""

    def test_nodegroup_preserved(self):
        net = _build_simple_net_with_nodelist()
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        assert isinstance(top2.layer, NodeGroup)
        assert top2.layer.size == 3

    def test_edgegroup_preserved(self):
        net = _build_simple_net_with_nodelist()
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        assert isinstance(top2.dense, EdgeGroup)
        assert len(top2.dense) == 6

    def test_nodeport_preserved(self):
        """NodePorts should survive serialization (even if no link)."""
        net = Network()
        net.port = NodePort(3)
        net.layer = NodeGroup(LIF(), 3)
        net.build()
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        assert isinstance(top2.port, NodePort)
        assert top2.port.size == 3

    def test_nodelist_preserved(self):
        net = _build_simple_net_with_nodelist()
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        assert isinstance(top2.out, NodeList)
        assert top2.out.size == 2
        for item in top2.out:
            assert isinstance(item, Node)

    def test_parameter_values_preserved(self):
        net = _build_simple_net_with_nodelist()
        net.layer.set_values(voltage=[0.1, 0.2, 0.3])
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        np.testing.assert_array_almost_equal(
            top2.layer.voltage, [0.1, 0.2, 0.3]
        )

    def test_edge_parameter_values_preserved(self):
        net = _build_simple_net_with_nodelist()
        net.dense.set_values(weight=2.5)
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        np.testing.assert_array_almost_equal(
            top2.dense.weight, np.full(6, 2.5)
        )


# ========================================================================
# Hierarchical topology serialization
# ========================================================================

class TestHierarchicalSerialization:
    """Round-trip for hierarchical topologies with nested Topology lists
    and child NodeGroups."""

    def test_round_trip_hierarchical(self):
        net = build_hierarchical_net()
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        assert isinstance(top2, Topology)
        # ff is a list of Topology objects
        assert isinstance(top2.ff, list)
        assert len(top2.ff) == 2
        for item in top2.ff:
            assert isinstance(item, Topology)

    def test_nested_nodegroups(self):
        net = build_hierarchical_net()
        d = topology_to_dict(net._topology)
        top2 = topology_from_dict(d)
        assert isinstance(top2.ff[0].layer, NodeGroup)
        assert top2.ff[0].layer.size == 4
        assert isinstance(top2.ff[1].layer, NodeGroup)
        assert top2.ff[1].layer.size == 2


# ========================================================================
# network_to_dict / network_from_dict
# ========================================================================

class TestNetworkSerialization:
    """network_to_dict / network_from_dict round-trip: graph equivalence
    and structure preservation."""

    def test_graph_equivalence(self):
        net = _build_simple_net_with_nodelist()
        g1 = net.graph()
        d = network_to_dict(net)
        net2 = network_from_dict(d)
        g2 = net2.graph()
        assert set(g1.nodes) == set(g2.nodes)
        assert set(g1.edges) == set(g2.edges)

    def test_structure_preserved(self):
        net = build_hierarchical_net()
        s1 = net.structure()
        d = network_to_dict(net)
        net2 = network_from_dict(d)
        s2 = net2.structure()
        # Class names should match
        assert s1["class"] == s2["class"]


# ========================================================================
# File-level save / load
# ========================================================================

class TestSaveLoadJson:
    """File-level save/load using the JSON format."""

    def test_network_json_round_trip(self, simple_net, tmp_path):
        p = tmp_path / "net.json"
        save(simple_net, p)
        net2 = load(p)
        assert isinstance(net2, Network)
        g1 = simple_net.graph()
        g2 = net2.graph()
        assert set(g1.nodes) == set(g2.nodes)

    def test_topology_json_round_trip(self, simple_net, tmp_path):
        p = tmp_path / "topo.json"
        save(simple_net._topology, p)
        top2 = load(p)
        assert isinstance(top2, Topology)

    def test_state_dict_json_round_trip(self, simple_net, tmp_path):
        sd = simple_net.state_dict()
        p = tmp_path / "sd.json"
        save(sd, p)
        sd2 = load(p)
        assert isinstance(sd2, dict)
        for k in sd:
            np.testing.assert_array_almost_equal(sd[k], sd2[k])

    def test_explicit_format(self, simple_net, tmp_path):
        p = tmp_path / "net.dat"
        save(simple_net, p, format="json")
        net2 = load(p, format="json")
        assert isinstance(net2, Network)


# ========================================================================
# File-level save / load (pickle)
# ========================================================================

class TestSaveLoadPickle:
    """File-level save/load using the pickle (.pkl/.pickle/.pt) format."""

    def test_network_pickle_round_trip(self, simple_net, tmp_path):
        p = tmp_path / "net.pkl"
        save(simple_net, p)
        net2 = load(p)
        assert isinstance(net2, Network)

    def test_topology_pickle_round_trip(self, simple_net, tmp_path):
        p = tmp_path / "topo.pickle"
        save(simple_net._topology, p)
        top2 = load(p)
        assert isinstance(top2, Topology)

    def test_pt_extension_uses_pickle(self, simple_net, tmp_path):
        p = tmp_path / "net.pt"
        save(simple_net, p)
        net2 = load(p)
        assert isinstance(net2, Network)


# ========================================================================
# Save / load error handling
# ========================================================================

class TestSaveLoadErrors:
    """Unknown format errors, unknown model class during deserialisation,
    and unexpected topology-dict types."""

    def test_unknown_format_save(self, simple_net, tmp_path):
        p = tmp_path / "net.xyz"
        with pytest.raises(ValueError, match="Unknown format"):
            save(simple_net, p, format="xml")

    def test_unknown_format_load(self, tmp_path):
        p = tmp_path / "net.xyz"
        with pytest.raises(ValueError, match="Unknown format"):
            load(p, format="xml")

    def test_unknown_model_class_raises(self):
        """Deserializing a topology with an unknown model class should raise."""
        net = build_simple_net(node_list=True)
        d = topology_to_dict(net._topology)
        # Corrupt the model class name in one nodegroup
        d["layer"]["model"]["__class__"] = "UnknownModel"
        with pytest.raises(TypeError, match="Unable to find model class"):
            topology_from_dict(d)

    def test_unknown_topology_type_raises(self):
        """An unexpected __type__ in a topology dict should raise TypeError."""
        bad_dict = {
            "__type__": "Topology",
            "thing": {"__type__": "CompletelyUnknown", "data": 42},
        }
        with pytest.raises(TypeError, match="Unexpected item"):
            topology_from_dict(bad_dict)

    def test_non_dict_in_topology_list_raises(self):
        """A bare non-dict item in a topology list should raise TypeError."""
        bad_dict = {
            "__type__": "Topology",
            "items": ["not_a_dict"],
        }
        with pytest.raises(TypeError, match="Unexpected item"):
            topology_from_dict(bad_dict)


# ========================================================================
# Full round-trip: build -> serialize -> deserialize -> compare graphs
# ========================================================================

class TestFullRoundTrip:
    """Parametrised end-to-end: build → save (JSON) → load → compare
    graph nodes/edges and state-dict values across all network builders."""

    @pytest.mark.parametrize("builder", [_build_simple_net_with_nodelist, build_hierarchical_net, build_deeply_nested_net, build_complex_net, build_cyclic_net, _build_subtractor_net])
    def test_json_round_trip_graph_identical(self, builder, tmp_path):
        net = builder()
        p = tmp_path / "net.json"
        save(net, p)
        net2 = load(p)
        g1 = net.graph()
        g2 = net2.graph()
        assert set(g1.nodes) == set(g2.nodes)
        assert g1.number_of_edges() == g2.number_of_edges()
        # Check node data values
        for node in g1.nodes:
            for k, v in g1.nodes[node].items():
                v2 = g2.nodes[node][k]
                if isinstance(v, (int, float)):
                    assert v == pytest.approx(v2), f"Mismatch at {node}.{k}"

    @pytest.mark.parametrize("builder", [_build_simple_net_with_nodelist, build_hierarchical_net, build_deeply_nested_net, build_complex_net, build_cyclic_net, _build_subtractor_net])
    def test_state_dict_round_trip(self, builder, tmp_path):
        net = builder()
        sd = net.state_dict()
        p = tmp_path / "sd.json"
        save(sd, p)
        sd2 = load(p)
        for k in sd:
            np.testing.assert_array_almost_equal(sd[k], sd2[k])
