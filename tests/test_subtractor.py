"""
Regression test for the subtractor example notebook.

This replicates the full pipeline from the ``example_subtractor`` and
``example_simulator`` notebooks: network definition → build → graph structure
checks → Brian simulation → spike-decoded output verification.

Serialization and state-dict round-trip coverage for this network lives in
``test_serialization.py`` (via the parametrized ``TestFullRoundTrip`` suite).

The computation is:  101001₂ (41) − 010110₂ (22) = 010011₂ (19).
"""
import itertools
import pytest
import networkx as nx

pytestmark = pytest.mark.regression

from sango import Network, NodeGroup, EdgeGroup, NodePort, NodeList
from sango.model import IN, LIF, PSP

# Conditionally import Brian backend
try:
    from sango.backend.brian.brian import SimBrian
    _has_brian = True
except ImportError:
    _has_brian = False


# ========================================================================
# Network definitions (mirroring the notebook)
# ========================================================================

class Input(Network):
    """Spike-generator input layer: produces spike trains and a control pulse."""

    def __init__(self, spike_times, ctrl_times):
        super().__init__()
        self.spike_times = spike_times
        self.ctrl_times = ctrl_times

    def build(self):
        self.spikegen = NodeGroup(IN(), len(self.spike_times), times=self.spike_times)
        self.ctrl_out = NodeGroup(IN(), 1, times=self.ctrl_times)


class Adder(Network):
    """Carry-chain adder: sums binary spikes via threshold-coded carry nodes."""

    def __init__(self):
        super().__init__()
        self.input = NodePort()
        self.ctrl_in = NodePort(1)

    def build(self):
        num_input = len(self.input)
        num_carry = 2 * num_input - 1
        self.carry = NodeGroup(
            LIF(), num_carry,
            threshold=[0.9 + i for i in range(num_carry)],
        )
        self.output = NodeGroup(LIF(threshold=0.9), 1)

        # Input → Carry (all-to-all)
        self.ic = EdgeGroup(
            self.input, self.carry, PSP(),
            list(itertools.product(range(num_input), range(num_carry))),
        )

        # Carry → Carry
        edges = []
        i = 1
        for c1 in range(1, num_carry, 2):
            for c2 in range(c1 - i, num_carry):
                edges.append((c1, c2))
            i += 1
        self.cc = EdgeGroup(self.carry, self.carry, PSP(), edges)

        # Carry → Output (alternating +1 / −1 weights)
        self.co = EdgeGroup(
            self.carry, self.output, PSP(),
            edges=[(i, 0) for i in range(num_carry)],
            weight=[1.0 - (i % 2) * 2.0 for i in range(num_carry)],
        )


class Subtractor(Network):
    """Binary subtractor: inverts B, adds 1 (two's complement), then adds A."""

    def __init__(self, len_inputs=4):
        super().__init__()
        self.len_inputs = len_inputs
        self.input = NodePort(2)
        self.ctrl_in = NodePort(1)

    def build(self):
        self.a = NodeList([self.input[0]])
        self.b = NodeList([self.input[1]])

        # Inversion
        self.not_b = NodeGroup(LIF(threshold=0.9))
        self.b__nb = EdgeGroup(self.b, self.not_b, PSP(), weight=-1.0, delay=2.0)

        # Control relay
        self.ctrl_relay = NodeGroup(LIF(), self.len_inputs, threshold=0.9)
        self.ci__r = EdgeGroup(
            self.ctrl_in, self.ctrl_relay, PSP(),
            edges=[(0, i) for i in range(self.len_inputs)],
        )
        self.cr__nb = EdgeGroup(
            self.ctrl_relay, self.not_b, PSP(),
            edges=[(i, 0) for i in range(self.len_inputs)],
            delay=[1.0 + i for i in range(self.len_inputs)],
        )

        # Add-1 relay
        self.one_relay = NodeGroup(LIF(threshold=0.9))
        self.ci__or = EdgeGroup(self.ctrl_in, self.one_relay, PSP(), delay=2.0)

        # First Adder (−B)
        self.adder_b_input = NodeList([self.not_b[0], self.one_relay[0]])
        self.adder_b = Adder()
        self.bind(self.adder_b_input, self.adder_b.input)

        # A relay
        self.a_relay = NodeGroup(LIF(), 1, threshold=0.9)
        self.a__ar = EdgeGroup(self.a, self.a_relay, PSP(), delay=4.0)

        # Second Adder (A − B)
        self.adder_ab_input = NodeList([self.a_relay[0], self.adder_b.output[0]])
        self.adder_ab = Adder()
        self.bind(self.adder_ab_input, self.adder_ab.input)

        # Output
        self.output = NodeGroup(LIF(threshold=0.9), 1)
        self.ab__o = EdgeGroup(self.adder_ab.output, self.output, PSP())

        # Suppress final carry
        self.ci__o = EdgeGroup(
            self.ctrl_in, self.output, PSP(),
            weight=-1, delay=self.len_inputs + 7.0,
        )

        # Output-start timing
        self.ctrl_out = NodeGroup(LIF(threshold=0.9))
        self.ci__co = EdgeGroup(self.ctrl_in, self.ctrl_out, PSP(), delay=7.0)


# ========================================================================
# Helpers
# ========================================================================

def _build_subtractor_net():
    """Build the subtractor network from the example notebooks.

    Inputs:  A = 101001₂ (41),  B = 010110₂ (22),  ctrl at t=0.
    """
    input_vec = [[0, 3, 5], [1, 2, 4]]
    ctrl_vec = [[0]]

    net = Network()
    net.i = Input(spike_times=input_vec, ctrl_times=ctrl_vec)
    net.s = Subtractor(len_inputs=6)
    net.bind(net.i.spikegen, net.s.input)
    net.bind(net.i.ctrl_out, net.s.ctrl_in)
    net.build()
    return net


def _decode_output(spike_list, node_map, output_name, ctrl_name, num_bits):
    """Decode a spike-time list into a bit string (MSB-first)."""
    index_output = node_map[output_name]
    index_ctrl = node_map[ctrl_name]
    output_spikes = spike_list[index_output]
    ctrl_spikes = spike_list[index_ctrl]
    t0 = ctrl_spikes[0]
    bits = [0] * num_bits
    for t in output_spikes:
        step = int(t - t0)
        if 0 <= step < num_bits:
            bits[step] = 1
    return list(reversed(bits))


# ========================================================================
# Module-scoped fixture: build + simulate once for the whole file
# ========================================================================

@pytest.fixture(scope="module")
def subtractor_net():
    """Built (but not simulated) subtractor network."""
    return _build_subtractor_net()


@pytest.fixture(scope="module")
def subtractor_sim():
    """Compiled and run Brian simulation of the subtractor network."""
    if not _has_brian:
        pytest.skip("brian2 not installed")
    net = _build_subtractor_net()
    sim = SimBrian(net)
    sim.compile()
    sim.run(20)
    return sim, net


# ========================================================================
# Network construction
# ========================================================================

class TestSubtractorBuild:
    """Network construction: build flag, child registration, child and
    nested-adder build status."""

    def test_build_succeeds(self, subtractor_net):
        assert subtractor_net._built is True

    def test_children_registered(self, subtractor_net):
        assert "i" in subtractor_net._children
        assert "s" in subtractor_net._children

    def test_child_networks_built(self, subtractor_net):
        assert subtractor_net._children["i"]._built is True
        assert subtractor_net._children["s"]._built is True

    def test_nested_adders_built(self, subtractor_net):
        s = subtractor_net._children["s"]
        assert s._children["adder_b"]._built is True
        assert s._children["adder_ab"]._built is True


# ========================================================================
# Graph structure
# ========================================================================

class TestSubtractorGraph:
    """Graph-level assertions: node/edge counts, expected node names,
    ctrl-relay and adder-carry sizing, and structure hierarchy."""

    def test_node_count(self, subtractor_net):
        g = subtractor_net.graph()
        assert g.number_of_nodes() == 22

    def test_edge_count(self, subtractor_net):
        g = subtractor_net.graph()
        assert g.number_of_edges() == 42

    def test_expected_node_names(self, subtractor_net):
        g = subtractor_net.graph()
        names = set(g.nodes)
        # Input nodes
        assert "i.spikegen[0]" in names
        assert "i.spikegen[1]" in names
        assert "i.ctrl_out[0]" in names
        # Key internal nodes
        assert "s.not_b[0]" in names
        assert "s.one_relay[0]" in names
        assert "s.a_relay[0]" in names
        # Nested adder nodes
        assert "s.adder_b.output[0]" in names
        assert "s.adder_ab.output[0]" in names
        # Final output
        assert "s.output[0]" in names
        assert "s.ctrl_out[0]" in names

    def test_ctrl_relay_nodes(self, subtractor_net):
        g = subtractor_net.graph()
        relay_nodes = [n for n in g.nodes if n.startswith("s.ctrl_relay")]
        assert len(relay_nodes) == 6

    def test_adder_carry_nodes(self, subtractor_net):
        g = subtractor_net.graph()
        carry_b = [n for n in g.nodes if n.startswith("s.adder_b.carry")]
        carry_ab = [n for n in g.nodes if n.startswith("s.adder_ab.carry")]
        # Each adder has 2 inputs → 2*2 - 1 = 3 carry nodes
        assert len(carry_b) == 3
        assert len(carry_ab) == 3

    def test_structure_hierarchy(self, subtractor_net):
        s = subtractor_net.structure()
        assert "child" in s
        assert "i" in s["child"]
        assert "s" in s["child"]
        # Subtractor itself has nested children (the two adders)
        s_struct = s["child"]["s"]
        assert "child" in s_struct
        assert "adder_b" in s_struct["child"]
        assert "adder_ab" in s_struct["child"]


# ========================================================================
# Brian simulation — functional correctness
# ========================================================================

@pytest.mark.skipif(not _has_brian, reason="brian2 not installed")
class TestSubtractorSimulation:
    """End-to-end: simulate and verify the decoded binary subtraction."""

    def test_spike_list_length(self, subtractor_sim):
        sim, net = subtractor_sim
        spikes = sim.get_spikes()
        assert len(spikes) == 22

    def test_input_spikes_present(self, subtractor_sim):
        """Input spike generators should have fired."""
        sim, net = subtractor_sim
        spikes = sim.get_spikes()
        idx_a = sim.node_map["i.spikegen[0]"]
        idx_b = sim.node_map["i.spikegen[1]"]
        assert len(spikes[idx_a]) > 0
        assert len(spikes[idx_b]) > 0

    def test_ctrl_out_fires(self, subtractor_sim):
        """The output-start control node should fire exactly once."""
        sim, net = subtractor_sim
        spikes = sim.get_spikes()
        idx_ctrl = sim.node_map["s.ctrl_out[0]"]
        assert len(spikes[idx_ctrl]) == 1

    def test_output_fires(self, subtractor_sim):
        """The output node should fire (at least once)."""
        sim, net = subtractor_sim
        spikes = sim.get_spikes()
        idx_out = sim.node_map[net.s.output[0].name]
        assert len(spikes[idx_out]) > 0

    def test_decoded_result(self, subtractor_sim):
        """101001₂ (41) − 010110₂ (22) = 010011₂ (19)."""
        sim, net = subtractor_sim
        spikes = sim.get_spikes()
        result = _decode_output(
            spikes, sim.node_map,
            output_name=net.s.output[0].name,
            ctrl_name="s.ctrl_out[0]",
            num_bits=6,
        )
        assert result == [0, 1, 0, 0, 1, 1]

    def test_decoded_decimal(self, subtractor_sim):
        """Verify the decoded bit string equals 19 in decimal."""
        sim, net = subtractor_sim
        spikes = sim.get_spikes()
        bits = _decode_output(
            spikes, sim.node_map,
            output_name=net.s.output[0].name,
            ctrl_name="s.ctrl_out[0]",
            num_bits=6,
        )
        decimal = sum(b * 2 ** (len(bits) - 1 - i) for i, b in enumerate(bits))
        assert decimal == 19  # 41 - 22
