# General Imports
import numpy as np
import warnings

# Package Imports
from .model.base import NodeModel, EdgeModel
from .model.base import get_shared_params


# ========================================================================
# Lightweight proxy classes (for network elements)
# ========================================================================

class Node:
    """Lightweight proxy for a single node inside a NodeGroup.

    Holds only a back-reference to the owning group and an integer index.
    Attribute reads/writes are forwarded to the group-level numpy arrays.
    """
    __slots__ = ('_group', '_index')

    def __init__(self, group=None, index=None):
        object.__setattr__(self, '_group', group)
        object.__setattr__(self, '_index', index)

    @property
    def index(self):
        return self._index

    @property
    def name(self):
        group = self._group
        if group is None or group.path is None:
            return None
        return f'{group.path}[{self._index}]'

    @property
    def data(self):
        """Computed dict view of model data."""
        group = self._group
        if group is None:
            return {}
        data = {}
        for key in vars(group.nodemodel):
            arr = group.__dict__.get(key)
            if arr is not None:
                if key in group.shared_params:
                    data[key] = arr[0:1]
                else:
                    data[key] = arr[self._index:self._index + 1]
        return data

    def __getattr__(self, name):
        # explicit __getattr__ bypass to get the slot values
        group = object.__getattribute__(self, '_group')
        index = object.__getattribute__(self, '_index')
        if group is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        data = group.__dict__
        if name in data:
            arr = data[name]
            if name in group.shared_params:
                return arr[0]
            else:
                return arr[index]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        if name in ('_group', '_index'):
            object.__setattr__(self, name, value)
            return
        group = object.__getattribute__(self, '_group')
        index = object.__getattribute__(self, '_index')
        data = group.__dict__
        if name in data:
            arr = data[name]
            if name in group.shared_params:
                arr[0] = value
            else:
                arr[index] = value
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __dir__(self):
        group = self._group
        base = ['index', 'name', 'data']
        if group is not None:
            base += list(vars(group.nodemodel).keys())
        return base

    def __str__(self):
        if self.name is None:
            return 'detached node'
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return (self._group is other._group) and (self._index == other._index)

    def __hash__(self):
        return hash((id(self._group), self._index))


class Edge:
    """Lightweight proxy for a single edge inside an EdgeGroup."""
    __slots__ = ('_group', '_index')

    def __init__(self, group=None, index=None):
        object.__setattr__(self, '_group', group)
        object.__setattr__(self, '_index', index)

    @property
    def source_index(self):
        return int(self._group._source_index[self._index])

    @property
    def target_index(self):
        return int(self._group._target_index[self._index])

    @property
    def source_name(self):
        if self._group._source_name is None:
            return None
        return self._group._source_name[self._index]

    @property
    def target_name(self):
        if self._group._target_name is None:
            return None
        return self._group._target_name[self._index]

    @property
    def data(self):
        """Computed dict view of model data."""
        group = self._group
        data = {}
        for key in vars(group.edgemodel):
            arr = group.__dict__.get(key)
            if arr is not None:
                if key in group.shared_params:
                    data[key] = arr[0:1]
                else:
                    data[key] = arr[self._index:self._index + 1]
        return data

    def __getattr__(self, name):
        # explicit __getattr__ bypass to get the slot values
        group = object.__getattribute__(self, '_group')
        index = object.__getattribute__(self, '_index')
        if group is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        data = group.__dict__
        if name in data:
            arr = data[name]
            if name in group.shared_params:
                return arr[0]
            else:
                return arr[index]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        if name in ('_group', '_index'):
            object.__setattr__(self, name, value)
            return
        group = object.__getattribute__(self, '_group')
        index = object.__getattribute__(self, '_index')
        data = group.__dict__
        if name in data:
            arr = data[name]
            if name in group.shared_params:
                arr[0] = value
            else:
                arr[index] = value
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __dir__(self):
        group = self._group
        base = ['source_index', 'target_index', 'source_name', 'target_name', 'data']
        if group is not None:
            base += list(vars(group.edgemodel).keys())
        return base

    def __str__(self):
        if self.source_name is None:
            return 'detached edge'
        return f"{self.source_name} -> {self.target_name}"

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return (self._group is other._group) and (self._index == other._index)

    def __hash__(self):
        return hash((id(self._group), self._index))


class Link:
    """Lightweight proxy for a single link inside a NodePort."""
    __slots__ = ('_port', '_index')

    def __init__(self, port=None, index=None):
        object.__setattr__(self, '_port', port)
        object.__setattr__(self, '_index', index)

    @property
    def index(self):
        return self._index

    @property
    def link(self):
        port = self._port
        if port is None or port.link is None:
            return None
        return port.link[self._index]

    def __str__(self):
        if self.link is None:
            return 'detached link'
        return f"{self.link}"

    def __eq__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return (self._port is other._port) and (self._index == other._index)

    def __hash__(self):
        return hash((id(self._port), self._index))


# ========================================================================
# Group of instantiated nodes sharing the same model (e.g. Neurons)
# ========================================================================

class NodeGroup:
    def __init__(self, model, size=None, **kwargs):
        if isinstance(model, NodeModel):
            self.nodemodel = model # defaults
        else:
            raise TypeError(f"{model} not NodeModel class")
        self.shared_params = get_shared_params(self.nodemodel)
        self.path = None # if None, not built
        self._size = 0
        self.set_size(size)
        self.set_values(**kwargs)

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += self._size
            if key < 0 or key >= self._size:
                raise IndexError(f"index {key} out of range for NodeGroup of size {self._size}")
            return Node(self, key)
        elif isinstance(key, slice):
            return [Node(self, i) for i in range(*key.indices(self._size))]
        else:
            raise TypeError(f"indices must be integers or slices, not {type(key).__name__}")

    def __iter__(self):
        for i in range(self._size):
            yield Node(self, i)

    def __contains__(self, item):
        if isinstance(item, Node) and item._group is self:
            return 0 <= item._index < self._size
        return False

    def __str__(self):
        if self.path is None:
            return 'detached nodegroup'
        return f"(node) {self.path}"

    def __setattr__(self, name, value):
        if name in ('path', 'nodemodel', 'shared_params', '_size'):
            super().__setattr__(name, value)
        elif name in vars(self.nodemodel).keys():
            set_dict = {name: value}
            self.set_values(**set_dict)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name == 'size':
            return self._size
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        return list(set(
            super().__dir__() + list(vars(self.nodemodel).keys()) + ['size']
        ))

    def add_node(self, **kwargs):
        self._size += 1
        # append to numpy arrays (very slow...)
        for key, value in vars(self.nodemodel).items():
            if key in self.shared_params:
                continue
            if key in kwargs:
                self.__dict__[key] = np.append(self.__dict__[key], kwargs[key])
            else:
                self.__dict__[key] = np.append(self.__dict__[key], value)

    def set_size(self, size):
        if size is None:
            self._size = 1
        else:
            self._size = size

        # instantiate model data
        for key, value in vars(self.nodemodel).items():
            if key in self.shared_params: # shared params
                self.__dict__[key] = np.empty(1, dtype=object)
                self.__dict__[key][0] = value
            elif isinstance(value, (int, float)):
                # set key directly due to overloaded __setattr__
                self.__dict__[key] = np.full((self._size,), value)
            else:
                # safer way to deal with objects (e.g. empty list)
                self.__dict__[key] = np.empty((self._size,), dtype=object)
                self.__dict__[key][...] = [value for _ in range(self._size)]

    def set_values(self, **kwargs):
        for key, value in kwargs.items():
            if key in vars(self.nodemodel).keys():
                if key in self.shared_params:
                    getattr(self, key)[0] = value
                elif hasattr(value, '__len__'):
                    if len(value) != len(getattr(self, key)):
                        raise IndexError(f"size mismatch for {key}, required {len(getattr(self, key))}, got {len(value)}")
                    else:
                        for i, item in enumerate(value):
                            getattr(self, key)[i] = item
                else: # single value
                    for i in range(len(getattr(self, key))):
                        getattr(self, key)[i] = value
            else:
                raise KeyError(f"'{key}' not found in node model {self.nodemodel}")

    # flatten
    def set_path(self, path):
        self.path = path


# ========================================================================
# Group of instantiated edges between two sets of nodes (e.g. Synapses)
# ========================================================================

class EdgeGroup:
    def __init__(self, source, target, model, edges=None, **kwargs):
        if isinstance(model, EdgeModel):
            self.edgemodel = model # defaults
        else:
            raise TypeError(f"{model} not EdgeModel class")
        self.shared_params = get_shared_params(self.edgemodel)
        self.source = source
        self.target = target
        self.path = None
        self.edge_map = dict() # tuple to index
        self._source_index = None
        self._target_index = None
        self._source_name = None
        self._target_name = None
        self.set_edges(edges)
        if 'edge' in kwargs.keys():
            warnings.warn("'edge' found in keyword arguments, did you mean 'edges'?",
                          category=SyntaxWarning, stacklevel=2)
        self.set_values(**kwargs)

    def __len__(self):
        return len(self.edge_map)

    def __getitem__(self, key):
        if isinstance(key, int):
            size = len(self.edge_map)
            if key < 0:
                key += size
            if key < 0 or key >= size:
                raise IndexError(f"index {key} out of range for EdgeGroup of size {size}")
            return Edge(self, key)
        elif isinstance(key, slice):
            return [Edge(self, i) for i in range(*key.indices(len(self)))]
        elif isinstance(key, tuple):
            if len(key) == 2:
                key = (key[0], key[1], 0)
            try:
                index = self.edge_map[key]
                return Edge(self, index)
            except KeyError:
                raise KeyError(f"No item found for key: {key}")
        else:
            raise TypeError(f"List indices must be integers or edge tuples, not {type(key).__name__}")

    def __iter__(self):
        for i in range(len(self)):
            yield Edge(self, i)

    def __contains__(self, item):
        if isinstance(item, Edge) and item._group is self:
            return 0 <= item._index < len(self)
        return False

    def __str__(self):
        if self.path is None:
            return 'detached edgegroup'
        return f"(edge) {self.path}: {self.source} -> {self.target}"

    def __setattr__(self, name, value):
        if name in ('path', 'edgemodel', 'source', 'target', 'edge_map',
                    'shared_params', '_source_index', '_target_index',
                    '_source_name', '_target_name'):
            super().__setattr__(name, value)
        elif name in vars(self.edgemodel).keys():
            set_dict = {name: value}
            self.set_values(**set_dict)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name == 'edges':
            return list(zip(self._source_index.tolist(), self._target_index.tolist()))
        elif name == 'source_index':
            return self._source_index.tolist()
        elif name == 'target_index':
            return self._target_index.tolist()
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        return list(set(
            super().__dir__() + list(vars(self.edgemodel).keys()) + ['edges', 'source_index', 'target_index']
        ))

    def add_edge(self, *args, **kwargs):
        if not args:
            raise TypeError(f"Edge indices are required")
        elif len(args) == 1:
            s, t = args[0]
        elif len(args) == 2:
            s, t = args
        else:
            raise TypeError(f"Edge indices must be integers or a tuple of integers")
        # auto-assign edge key (incrementing for parallel edges)
        edge_key = 0
        while (s, t, edge_key) in self.edge_map:
            edge_key += 1
        index = len(self)
        self._source_index = np.append(self._source_index, s)
        self._target_index = np.append(self._target_index, t)
        self.edge_map[(s, t, edge_key)] = index
        # append to numpy arrays (very slow...)
        for key, value in vars(self.edgemodel).items():
            if key in self.shared_params: # don't update shared params
                continue
            if key in kwargs:
                self.__dict__[key] = np.append(self.__dict__[key], kwargs[key])
            else: # defaults
                self.__dict__[key] = np.append(self.__dict__[key], value)

    def set_edges(self, edges):
        if edges is None:
            self._source_index = np.array([0], dtype=int)
            self._target_index = np.array([0], dtype=int)
            self.edge_map[(0, 0, 0)] = 0
        else:
            # track next available key per (s,t) pair for auto-assignment
            next_edge_key = dict()
            source_index = []
            target_index = []
            for i, (s, t) in enumerate(edges):
                source_index.append(s)
                target_index.append(t)
                edge_key = next_edge_key.get((s, t), 0)
                self.edge_map[(s, t, edge_key)] = i
                next_edge_key[(s, t)] = edge_key + 1
            self._source_index = np.array(source_index, dtype=int)
            self._target_index = np.array(target_index, dtype=int)

        # instantiate model data
        for key, value in vars(self.edgemodel).items():
            if key in self.shared_params:
                self.__dict__[key] = np.empty(1, dtype=object)
                self.__dict__[key][0] = value
            elif isinstance(value, (int, float)):
                self.__dict__[key] = np.full((len(self),), value)
            else:
                self.__dict__[key] = np.empty((len(self),), dtype=object)
                self.__dict__[key][...] = [value for _ in range(len(self))]

    def set_values(self, **kwargs):
        for key, value in kwargs.items():
            if key in vars(self.edgemodel).keys():
                if key in self.shared_params:
                    getattr(self, key)[0] = value
                elif hasattr(value, '__len__'):
                    if len(value) != len(getattr(self, key)):
                        raise IndexError(f"size mismatch for {key}, required {len(getattr(self, key))}, got {len(value)}")
                    else:
                        for i, item in enumerate(value):
                            getattr(self, key)[i] = item
                else: # single value
                    for i in range(len(getattr(self, key))):
                        getattr(self, key)[i] = value
            else:
                raise KeyError(f"'{key}' not found in edge model {self.edgemodel}")

    def set_path(self, path):
        def trace(root, index):
            if isinstance(root, NodeGroup):
                return root[index].name
            elif isinstance(root, NodeList):
                try:
                    node = root[index]
                    while hasattr(node, 'link'):
                        node = node.link
                    return node.name
                except AttributeError:
                    print(f"error tracing {root}[{index}]")
            elif isinstance(root, NodePort):
                return trace(root.link, index)
            else:
                print(f"error tracing {root} {index}")
                return None
        
        # follow the links through ports
        self.path = path
        self._source_name = [None] * len(self.edge_map)
        self._target_name = [None] * len(self.edge_map)
        for i in range(len(self.edge_map)):
            self._source_name[i] = trace(self.source, int(self._source_index[i]))
            if self._source_name[i] is None:
                raise ValueError(f"error at {self.path}: setting source path {self.source}")
            self._target_name[i] = trace(self.target, int(self._target_index[i]))
            if self._target_name[i] is None:
                raise ValueError(f"error at {self.path}: setting target path {self.target}")


# ========================================================================
# Alias class pointing to set of (external) nodes (e.g. Network Inputs)
# ========================================================================

class NodePort:
    def __init__(self, size=None):
        if size is not None:
            self._size = size
        else:
            self._size = 0
        self.size = size  # if no size, creates dependency
        # this is basically just a symlink
        self.path = None
        self.link = None

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        if isinstance(key, int):
            n = len(self)
            if key < 0:
                key += n
            if key < 0 or key >= n:
                raise IndexError(f"index {key} out of range for NodePort of size {n}")
            return Link(self, key)
        elif isinstance(key, slice):
            return [Link(self, i) for i in range(*key.indices(len(self)))]
        else:
            raise TypeError(f"indices must be integers or slices, not {type(key).__name__}")

    def __iter__(self):
        for i in range(len(self)):
            yield Link(self, i)

    def __contains__(self, item):
        if isinstance(item, Link) and item._port is self:
            return 0 <= item._index < len(self)
        return False

    def __str__(self):
        if self.path is None:
            return 'detached nodeport'
        if self.link is None:
            return f"(port) {self.path} <- (no link)"
        return f"(port) {self.path} <- {self.link}"

    def set_size(self, size):
        if self.size is not None:
            warnings.warn(
                f"changing port size from {self.size} to {size}",
                category=UserWarning,
                stacklevel=2)
        self.size = size
        self._size = size

    def set_path(self, path):
        self.path = path

    def set_link(self, link):
        self.link = link


# ======================================================================
# Alias class with a list of (pointers to) nodes (e.g. Network Outputs)
# ======================================================================

class NodeList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.path = None

    def __getattr__(self, name):
        if name == 'size':
            return len(self)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        return super().__dir__() + ['size']

    def __str__(self):
        if self.path is None:
            return 'detached nodelist'
        return f"(list) {self.path}"

    def set_path(self, path):
        self.path = path
