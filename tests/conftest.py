"""
Shared fixtures and collection hooks for the Sango test suite.
"""
import itertools
import pytest
import numpy as np
from dataclasses import dataclass, field

from sango.model.base import (
    NodeModel, EdgeModel, InputModel, OutputModel,
    Neuron, Synapse, SpikeGen,
    LIF, PSP, IN, shared, get_shared_params,
)
from sango.core import NodeGroup, EdgeGroup, NodePort, NodeList, Node, Edge, Link
from sango.network import Network, Topology, TempPath
from sango.serialization import (
    save, load,
    topology_to_dict, topology_from_dict,
    network_to_dict, network_from_dict,
    statedict_to_json, statedict_from_json,
)


# ========================================================================
# Reusable helper Networks
# ========================================================================

class SpikeInput(Network):
    """Wraps spike-generator inputs."""
    def __init__(self, spike_times):
        super().__init__()
        self.spike_times = spike_times

    def build(self):
        self.spikegen = NodeGroup(IN(), len(self.spike_times), times=self.spike_times)


class Linear(Network):
    """Feed-forward layer with dense connectivity."""
    def __init__(self, size=4):
        super().__init__()
        self.size = size
        self.inp = NodePort()  # unsized -> dependency

    def build(self):
        self.layer = NodeGroup(LIF(), self.size)
        edges = list(itertools.product(range(self.inp.size), range(self.layer.size)))
        self.dense = EdgeGroup(self.inp, self.layer, PSP(), edges=edges)


class PassThrough(Network):
    """Single-port sub-network: input -> output with 1-to-1 edges."""
    def __init__(self):
        super().__init__()
        self.inp = NodePort()  # size resolved via binding

    def build(self):
        n = self.inp.size
        self.layer = NodeGroup(LIF(), n)
        self.dense = EdgeGroup(self.inp, self.layer, PSP(),
                               edges=[(i, i) for i in range(n)])


class Inner(Network):
    """Leaf sub-network used by deeply nested topologies."""
    def __init__(self, size=3):
        super().__init__()
        self._size = size
        self.inp = NodePort()

    def build(self):
        self.layer = NodeGroup(LIF(), self._size)
        edges = list(itertools.product(range(self.inp.size), range(self._size)))
        self.dense = EdgeGroup(self.inp, self.layer, PSP(), edges=edges)


class Middle(Network):
    """Mid-level sub-network that wraps an Inner child."""
    def __init__(self, inner_size=3):
        super().__init__()
        self._inner_size = inner_size
        self.inp = NodePort()

    def build(self):
        self.inner = Inner(self._inner_size)
        self.bind(self.inp, self.inner.inp)


class DualPort(Network):
    """Sub-network with two input ports and separate edge groups."""
    def __init__(self, size=4):
        super().__init__()
        self._size = size
        self.inp_a = NodePort()
        self.inp_b = NodePort()

    def build(self):
        self.layer = NodeGroup(LIF(), self._size)
        ea = list(itertools.product(range(self.inp_a.size), range(self._size)))
        eb = list(itertools.product(range(self.inp_b.size), range(self._size)))
        self.dense_a = EdgeGroup(self.inp_a, self.layer, PSP(), edges=ea, weight=1.0)
        self.dense_b = EdgeGroup(self.inp_b, self.layer, PSP(), edges=eb, weight=0.5)
        self.output = NodeList([self.inp_a[0], self.layer[2]]) 

# ========================================================================
# Builder functions (plain functions so they work with @parametrize too)
# ========================================================================

def build_simple_net(node_list=False):
    """Minimal flat network: input -> layer with dense edges."""
    net = Network()
    net.inp = NodeGroup(IN(), 2, times=[[1, 3], [2]])
    net.layer = NodeGroup(LIF(), 3)
    edges = list(itertools.product(range(2), range(3)))
    net.dense = EdgeGroup(net.inp, net.layer, PSP(), edges=edges)
    if node_list:
        net.out = NodeList([net.layer[0], net.layer[2]])
    net.build()
    return net


def build_hierarchical_net():
    """Two-layer hierarchical network using Linear sub-networks."""
    net = Network()
    net.inp = NodeGroup(IN(), 3, times=[[1], [2], [3]])
    net.ff = [Linear(4), Linear(2)]
    net.bind(net.inp, net.ff[0].inp)
    net.bind(net.ff[0].layer, net.ff[1].inp)
    net.build()
    return net


def build_deeply_nested_net(inner_size=3, src_size=2):
    """Three-level nesting: root -> Middle -> Inner, each with a NodePort."""
    net = Network()
    net.src = NodeGroup(LIF(), src_size)
    net.mid = Middle(inner_size)
    net.bind(net.src, net.mid.inp)
    net.build()
    return net


def build_complex_net():
    """Complex network with hierarchical children, multiple NodePorts, a
    NodeList mixing port-links and direct nodes, and multiple EdgeGroups
    with distinct weights."""
    net = Network()
    net.src_a = NodeGroup(IN(), 2, times=[[1], [2]])
    net.src_b = NodeGroup(LIF(), 3)
    net.sub = DualPort(4)
    #net.set_portsize(net.sub.inp_a, 2) # automatic
    net.set_portsize(net.sub.inp_b, 3)  # manual
    net.bind(net.src_a, net.sub.inp_a)
    net.bind(net.src_b, net.sub.inp_b)
    net.out = NodeList([net.sub.output[0], net.sub.output[1],
                        net.sub.inp_b[1], net.src_b[2]])
    net.build()
    return net


def build_cyclic_net(size=2):
    """Two PassThrough children wired in a cycle: A -> B -> A."""
    net = Network()
    net.a = PassThrough()
    net.b = PassThrough()
    # Break the dependency cycle by giving one port its size up front.
    net.set_portsize(net.a.inp, size)
    # A's output feeds B, and B's output feeds back into A.
    net.bind(net.a.layer, net.b.inp)
    net.bind(net.b.layer, net.a.inp)
    net.build()
    return net


# ========================================================================
# Fixtures (thin wrappers around the builder functions above)
# ========================================================================

@pytest.fixture
def lif_model():
    return LIF()

@pytest.fixture
def psp_model():
    return PSP()

@pytest.fixture
def in_model():
    return IN()

@pytest.fixture
def small_nodegroup():
    """3-node LIF group."""
    return NodeGroup(LIF(), 3)

@pytest.fixture
def small_edgegroup(small_nodegroup):
    """Chain edges 0->1, 1->2 between nodes of *small_nodegroup*."""
    return EdgeGroup(small_nodegroup, small_nodegroup, PSP(),
                     edges=[(0, 1), (1, 2)])

@pytest.fixture
def simple_net():
    return build_simple_net()

@pytest.fixture
def hierarchical_net():
    return build_hierarchical_net()

@pytest.fixture
def deeply_nested_net():
    return build_deeply_nested_net()

@pytest.fixture
def cyclic_net():
    return build_cyclic_net()

@pytest.fixture
def complex_net():
    return build_complex_net()


# ========================================================================
# Collection ordering — run test categories in dependency order
# ========================================================================

MARK_ORDER = ["core", "network", "serialization", "backend", "regression"]

# Within a mark category, enforce a stable file ordering so that
# lower-level modules are exercised before the things that depend on them.
_FILE_ORDER = [
    "test_core.py",
    "test_nodegroup.py",
    "test_edgegroup.py",
    "test_nodeport.py",
    "test_nodelist.py",
    "test_wiring.py",
    "test_topology.py",
    "test_build.py",
    "test_network.py",
    "test_integration.py",
    "test_serialization.py",
    "test_backend.py",
    "test_simulator.py",
    "test_subtractor.py",
]
_FILE_RANK = {name: i for i, name in enumerate(_FILE_ORDER)}


def pytest_collection_modifyitems(items):
    """Sort collected tests so that categories run in a logical order:
    core → network → serialization → backend → regression.
    Unmarked tests are appended at the end.

    Within each category, files are ordered according to ``_FILE_ORDER``
    so that lower-level modules are tested before higher-level ones.
    Tests inside a single file keep their original source order.
    """
    original_index = {id(item): idx for idx, item in enumerate(items)}

    def sort_key(item):
        # Determine mark bucket
        mark_rank = len(MARK_ORDER)
        for i, mark_name in enumerate(MARK_ORDER):
            if item.get_closest_marker(mark_name):
                mark_rank = i
                break

        # Determine file bucket
        filename = item.fspath.basename
        file_rank = _FILE_RANK.get(filename, len(_FILE_ORDER))

        return (mark_rank, file_rank, original_index[id(item)])

    items.sort(key=sort_key)
