"""
Tests for the shared Backend data structures: process_graph, node/edge
rekeying, group indexing, and multigraph detection.
"""
import pytest
import warnings

pytestmark = pytest.mark.backend

from sango.model.base import LIF, PSP, IN
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList
from sango.network import Network

from .conftest import Linear, SpikeInput, build_simple_net, build_hierarchical_net

# Use the STACS backend class for testing here
from sango.backend.stacs.stacs import SimSTACS


# ========================================================================
# Backend data structures (process_graph)
# ========================================================================

class TestProcessGraph:
    """
    Instantiate SimSTACS just to test the shared Backend.process_graph()
    data structures (no actual simulation).
    """

    # -- index and data population -----------------------------------------

    def test_node_index_populated(self):
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        assert isinstance(sim.node_index, dict)
        assert len(sim.node_index) == net.graph().number_of_nodes()

    def test_node_data_populated(self):
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        assert len(sim.node_data) == sim.num_nodes
        for d in sim.node_data:
            assert "model" in d

    def test_edge_data_populated(self):
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        assert len(sim.edge_data) == sim.num_nodes
        total_edges = sum(len(ed) for ed in sim.edge_data)
        assert total_edges > 0

    # -- group counting and ordering ---------------------------------------

    def test_group_count(self):
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        assert "IN" in sim.group_count
        assert "LIF" in sim.group_count

    def test_input_data_extracted(self):
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        assert len(sim.input_data) > 0
        for times in sim.input_data.values():
            assert isinstance(times, list)

    def test_group_index_ordering(self):
        """Input groups should precede non-input groups."""
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        in_idx = sim.group_index.get("IN")
        lif_idx = sim.group_index.get("LIF")
        assert in_idx is not None and lif_idx is not None
        assert in_idx < lif_idx

    def test_group_offset_prefix_sum(self):
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        for i in range(len(sim.group_total)):
            assert sim.group_offset[i + 1] == sim.group_offset[i] + sim.group_total[i]

    # -- multigraph detection ----------------------------------------------

    def test_multigraph_detection(self):
        """Multi-edges should flag is_multigraph."""
        net = Network()
        net.inp = NodeGroup(IN(), 1, times=[[1]])
        net.layer = NodeGroup(LIF(), 2)
        net.e1 = EdgeGroup(net.layer, net.layer, PSP(), edges=[(0, 1), (0, 1)])
        net.e_in = EdgeGroup(net.inp, net.layer, PSP(), edges=[(0, 0)])
        net.build()
        sim = SimSTACS(net)
        with pytest.warns(UserWarning, match="Multi/parallel edges detected"):
            sim.process_graph()
        assert sim.is_multigraph is True

    # -- hierarchical networks ---------------------------------------------

    def test_hierarchical_process_graph(self):
        net = build_hierarchical_net()
        sim = SimSTACS(net)
        sim.process_graph()
        # 3 input + 4 ff[0] + 2 ff[1] = 9
        assert sim.num_nodes == 9

    # -- DSL → backend rekeying --------------------------------------------

    def test_rekey_node_maps_dsl_to_backend(self):
        """rekey_model should replace DSL names (e.g. 'voltage') with
        backend names (e.g. 'v') and inject defaults for None-dsl entries."""
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        lif_nodes = [d for d in sim.node_data if d["model"] == "LIF"]
        for d in lif_nodes:
            # Mapped from DSL
            assert "v" in d
            assert "v_thresh" in d
            assert "v_reset" in d
            assert "v_bias" in d
            assert "v_leak" in d
            # Original DSL names should be gone
            assert "voltage" not in d
            assert "threshold" not in d

    def test_rekey_edge_maps_dsl_to_backend(self):
        """Edge data should have backend keys after rekeying."""
        net = build_simple_net()
        sim = SimSTACS(net)
        sim.process_graph()
        # Find an edge with data
        for s_edges in sim.edge_data:
            for key, data in s_edges.items():
                if data.get("model") == "PSP":
                    assert "weight" in data
                    assert "delay" in data
                    return
        pytest.fail("No PSP edge found after process_graph")

    def test_rekey_param_returns_keys_and_values(self):
        """rekey_param should return (keys_tuple, values_tuple)."""
        net = build_simple_net()
        sim = SimSTACS(net)
        # Use raw node data from the graph directly
        g = net.graph()
        node_name = list(g.nodes)[0]
        data = dict(g.nodes[node_name])
        keys, values = sim.rekey_param(data)
        assert isinstance(keys, tuple)
        assert isinstance(values, tuple)
        assert len(keys) == len(values)
