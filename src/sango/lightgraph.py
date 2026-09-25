"""
Lightweight graph data structures that provide the subset of the
networkx DiGraph / MultiDiGraph API consumed by process_graph.

Supported operations
--------------------
* ``add_node(name, **data)``
* ``add_edge(source, target, **data)``
* ``add_nodes_from([(name, **data),])``
* ``add_edges_from([(source, target, **data),])``
* ``number_of_nodes()``
* ``number_of_edges()``
* ``is_multigraph()``
* ``nodes(data=False)`` – iterate node names, or ``(name, data_dict)`` pairs
* ``edges(data=False, keys=False)`` – iterate edge tuples

These classes intentionally do not depend on networkx and carry no
analysis or plotting functionality.
"""


# ======================================================================
# Node / edge view helpers (mimic networkx view iteration protocol)
# ======================================================================

class _NodeView:
    __slots__ = ('_nodes',)

    def __init__(self, nodes):
        # nodes: dict of {name: data_dict}
        self._nodes = nodes

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, item):
        return item in self._nodes

    def __getitem__(self, key):
        return self._nodes[key]

    def __call__(self, data=False):
        if data:
            return self._nodes.items()
        return self._nodes.keys()


class _EdgeView:
    __slots__ = ('_edges',)

    def __init__(self, edges):
        # edges: dict of {(src, tgt): data_dict}
        self._edges = edges

    def __iter__(self):
        for src, tgt, _ in self._edges:
            yield src, tgt

    def __len__(self):
        return len(self._edges)

    def __call__(self, data=False):
        if data:
            for (src, tgt), dat in self._edges.items():
                yield src, tgt, dat
        else:
            for src, tgt in self._edges:
                yield src, tgt


class _MultiEdgeView:
    __slots__ = ('_edges',)

    def __init__(self, edges):
        # edges: list of (src, tgt, key, data_dict)
        self._edges = edges

    def __iter__(self):
        for src, tgt, _key, _ in self._edges:
            yield src, tgt

    def __len__(self):
        return len(self._edges)

    def __call__(self, data=False, keys=False):
        if data and keys:
            for (src, tgt, key), dat in self._edges.items():
                yield src, tgt, key, dat
        elif data:
            for (src, tgt, _), dat in self._edges.items():
                yield src, tgt, dat
        elif keys:
            for src, tgt, key in self._edges:
                yield src, tgt, key
        else:
            for src, tgt, _ in self._edges:
                yield src, tgt


# ======================================================================
# LightDiGraph
# ======================================================================

class LightDiGraph:
    def __init__(self):
        self._nodes = {}              # {name: data_dict}
        self._edges = {}              # {(src, tgt): data_dict)}

    def add_node(self, name, **data):
        self._nodes[name] = data

    def add_edge(self, source, target, **data):
        edge_tuple = (source, target)
        self._edges[edge_tuple] = data
    
    def add_nodes_from(self, nodes_for_adding):
        _nodes = self._nodes
        for name, data in nodes_for_adding:
            _nodes[name] = data

    def add_edges_from(self, edges_for_adding):
        _edges = self._edges
        for source, target, data in edges_for_adding:
            _edges[(source, target)] = data
    
    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)

    def is_multigraph(self):
        return False

    @property
    def nodes(self):
        return _NodeView(self._nodes)

    def edges(self, data=False):
        return _EdgeView(self._edges)(data=data)


# ======================================================================
# LightMultiDiGraph
# ======================================================================

class LightMultiDiGraph:
    def __init__(self):
        self._nodes = {}             # {name: data_dict}
        self._edges = {}             # {(src, tgt, key): data_dict}
        self._edge_key_count = {}    # {(src, tgt): next_key}

    def add_node(self, name, **data):
        self._nodes[name] = data

    def add_edge(self, source, target, **data):
        key = self._edge_key_count.get((source, target), 0)
        self._edge_key_count[(source, target)] = key + 1
        self._edges[(source, target, key)] = data

    def add_nodes_from(self, nodes_for_adding):
        _nodes = self._nodes
        for name, data in nodes_for_adding:
            _nodes[name] = data

    def add_edges_from(self, edges_for_adding):
        _edges = self._edges
        _edge_key_count = self._edge_key_count
        for source, target, data in edges_for_adding:
            key = _edge_key_count.get((source, target), 0)
            _edge_key_count[(source, target)] = key + 1
            _edges[(source, target, key)] = data

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)

    def is_multigraph(self):
        return True

    @property
    def nodes(self):
        return _NodeView(self._nodes)

    def edges(self, data=False, keys=False):
        return _MultiEdgeView(self._edges)(data=data, keys=keys)
