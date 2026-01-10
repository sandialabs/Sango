# General Imports
import numpy as np

# Package Imports
from .model import NodeModel, EdgeModel

# Base classes
class Node:
    __slots__ = ('name', 'index', 'data')
    
    def __init__(self, index=None):
        self.index = index
        self.data = dict()
        # set during path flattening
        self.name = None

    def __getattr__(self, name):
        if name in self.data:
            if hasattr(self.data[name], 'base'):
                # element of numpy array (could be more error checking)
                if (isinstance(self.data[name][0], tuple) and
                    len(self.data[name][0]) == 1):
                    return self.data[name][0][0]
                else:
                    return self.data[name][0]
            else:
                return self.data[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            if hasattr(self.data[name], 'base'):
                # element of numpy array
                self.data[name][0] = value
            else:
                self.data[name] = value

    def __dir__(self):
        return super().__dir__() + list(self.data.keys())
    
    def __str__(self):
        if self.name is None:
            return 'detatched node'
        else:
            return f"{self.name}"

class Edge:
    __slots__ = ('source_name', 'target_name', 'source_index', 'target_index',  'data')
    
    def __init__(self, source_index=None, target_index=None):
        self.source_index = source_index
        self.target_index = target_index
        self.data = dict()
        # set during path flattening
        self.source_name = None
        self.target_name = None
    
    def __getattr__(self, name):
        if name in self.data:
            if hasattr(self.data[name], 'base'):
                # element of numpy array
                if (isinstance(self.data[name][0], tuple) and
                    len(self.data[name][0]) == 1):
                    return self.data[name][0][0]
                else:
                    return self.data[name][0]
            else:
                return self.data[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            if hasattr(self.data[name], 'base'):
                # element of numpy array
                self.data[name][0] = value
            else:
                self.data[name] = value

    def __dir__(self):
        return super().__dir__() + list(self.data.keys())
    
    def __str__(self):
        if self.source_name is None:
            return 'detatched edge'
        else:
            return f"{self.source_name} -> {self.target_name}"

class Link:
    __slots__ = ('link', 'index')
    
    def __init__(self, index=None):
        self.index = index
        # set during path flattening
        self.link = None

    def __str__(self):
        if self.link is None:
            return 'detatched link'
        else:
            return f"{self.link}"

# Group of instantiated nodes sharing the same model (e.g. Neurons)
class NodeGroup(list):
    def __init__(self, model, size=None, **kwargs):
        if isinstance(model, NodeModel):
            self.nodemodel = model # defaults
        else:
            print(f"error: {model} not NodeModel class")
        self.path = None # if None, not built
        self.set_size(size)
        self.set_values(**kwargs)

    def __str__(self):
        if self.path is not None:
            return f"(node) {self.path}"
        else:
            return 'detached nodegroup'

    def __setattr__(self, name, value):
        if name in ('path', 'nodemodel'):
            super().__setattr__(name, value)
        elif name in vars(self.nodemodel).keys():
            set_dict = {name: value}
            self.set_values(**set_dict)
        else:
            super().__setattr__(name, value)
    
    def __getattr__(self, name):
        if name == 'size':
            return len(self)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        return super().__dir__() + list(vars(self.nodemodel).keys()) + ['size']
    
    def add_node(self, **kwargs):
        # append new node to regular list
        index = len(self)
        super().append(Node(index))
        # append to numpy arrays (very slow...)
        for key, value in vars(self.nodemodel).items():
            if key in kwargs:
                self.__dict__[key] = np.append(self.__dict__[key], kwargs[key])
            else: # defaults
                self.__dict__[key] = np.append(self.__dict__[key], value)
            # add dynamic view (sliced)
            for i, item in enumerate(self):
                item.data[key] = getattr(self, key)[i:i+1]

    def set_size(self, size):
        # node list
        if size is not None:
            super().__init__([Node(i) for i in range(size)])
        else:
            super().__init__()
        
        # instantiate model data
        for key, value in vars(self.nodemodel).items():
            if isinstance(value, (str, tuple)): # shared params
                self.__dict__[key] = np.empty(1,dtype=object)
                self.__dict__[key][0] = value
                for item in self:
                    item.data[key] = getattr(self, key)[0:1]
            elif isinstance(value, (int, float)):
                # set key directly due to overloaded __setattr__
                self.__dict__[key] = np.full((len(self)),value)
                for i, item in enumerate(self):
                    # dynamic view (sliced)
                    item.data[key] = getattr(self, key)[i:i+1]
            else:
                # safer way to deal with objects (e.g. empty list)
                self.__dict__[key] = np.empty((len(self)),dtype=object)
                self.__dict__[key][...] = [value for _ in range(len(self))]
                for i, item in enumerate(self):
                    item.data[key] = getattr(self, key)[i:i+1]

    def set_values(self, **kwargs):
        for key, value in kwargs.items():
            if key in vars(self.nodemodel).keys():
                if isinstance(value, (str, tuple)):
                    getattr(self, key)[0] = value
                elif (isinstance(value, (int, float)) and
                      isinstance(getattr(self, key)[0], tuple)):
                    getattr(self, key)[0] = (value,) # keep it as tuple
                elif hasattr(value, '__len__'):
                    if len(value) != len(getattr(self, key)):
                        print(f"error: size mismatch for {key}, required {len(getattr(self, key))}, got {len(value)}")
                    else:
                        for i, item in enumerate(value):
                            getattr(self, key)[i] = item
                else: # single value
                    for i in range(len(getattr(self, key))):
                        getattr(self, key)[i] = value

    # flatten
    def set_path(self, path):
        self.path = path
        for i in range(len(self)):
            self[i].name = f'{self.path}[{i}]'

# Group of instantiated edges between two sets of nodes (e.g. Synapses)
class EdgeGroup(list):
    def __init__(self, source, target, model, edges=None, **kwargs):
        if isinstance(model, EdgeModel):
            self.edgemodel = model # defaults
        else:
            print(f"error: {model} not EdgeModel class")
        self.source = source
        self.target = target
        self.path = None
        self.edge_map = dict() # tuple to index
        self.set_edges(edges)
        self.set_values(**kwargs)

    def __str__(self):
        if self.path is not None:
            return f"(edge) {self.path}: {self.source} -> {self.target}"
        else:
            return 'detached edgegroup'
    
    def __setattr__(self, name, value):
        if name in ('path', 'edgemodel', 'source', 'target', 'edge_map'):
            super().__setattr__(name, value)
        elif name in vars(self.edgemodel).keys():
            set_dict = {name: value}
            self.set_values(**set_dict)
        else:
            super().__setattr__(name, value)
            
    def __getattr__(self, name):
        if name == 'edges':
            return [(self[i].source_index, self[i].target_index) for i in range(len(self))]
        elif name == 'source_index':
            return [self[i].source_index for i in range(len(self))]
        elif name == 'target_index':
            return [self[i].target_index for i in range(len(self))]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        return super().__dir__() + list(vars(self.edgemodel).keys()) + ['edges', 'source_index', 'target_index']

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            # Handle standard index access
            return super().__getitem__(key)
        elif isinstance(key, tuple):
            # Handle custom access using a tuple key
            try:
                index = self.edge_map[key]
                return super().__getitem__(index)
            except KeyError:
                raise KeyError(f"No item found for key: {key}")
        else:
            raise TypeError(f"List indices must be integers or edge tuples, not {type(key).__name__}")

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            super().__setitem__(key, value)
        elif isinstance(key, tuple):
            try:
                index = self.edge_map[key]
                super().__setitem__(index, value)
            except KeyError:
                raise KeyError(f"No item found for key: {key}")
        else:
            raise TypeError(f"List indices must be integers or edge tuples, not {type(key).__name__}")
    
    def add_edge(self, *args, **kwargs):
        # get source and target indexes
        if not args:
            raise TypeError(f"Edge indices are required")
        elif len(args) == 1:
            s, t = args[0]
        elif len(args) == 2:
            s, t = args
        else:
            raise TypeError(f"Edge indices must be integers or a tuple of integers")
        if (s,t) in self.edge_map:
            raise ValueError(f"error: edge '({s},{t})' already exists (no duplicates allowed)")
        # append new node to regular list
        index = len(self)
        super().append(Edge(s,t))
        self.edge_map[(s,t)] = index
        # append to numpy arrays (very slow...)
        for key, value in vars(self.edgemodel).items():
            if key in kwargs:
                self.__dict__[key] = np.append(self.__dict__[key], kwargs[key])
            else: # defaults
                self.__dict__[key] = np.append(self.__dict__[key], value)
            # add dynamic view (sliced)
            for i, item in enumerate(self):
                item.data[key] = getattr(self, key)[i:i+1]
        
    def set_edges(self, edges):
        # edge list (from list of tuples)
        if edges is None:
            super().__init__([Edge(0,0)]) # default edge
            self.edge_map[(0,0)] = 0
        else:
            super().__init__([Edge(s,t) for s,t in edges])
            for i, (s,t) in enumerate(edges):
                # check for duplicates
                if (s,t) in self.edge_map:
                    raise ValueError(f"error: edge '({s},{t})' already exists (no duplicates allowed)")
                self.edge_map[(s,t)] = i
            
        # instantiate model data
        for key, value in vars(self.edgemodel).items():
            if isinstance(value, (str, tuple)): # shared params
                self.__dict__[key] = np.empty(1,dtype=object)
                self.__dict__[key][0] = value
                for item in self:
                    item.data[key] = getattr(self, key)[0:1]
            elif isinstance(value, (int, float)):
                # set key directly due to overloaded __setattr__
                self.__dict__[key] = np.full((len(self)),value)
                for i, item in enumerate(self):
                    # dynamic view (sliced)
                    item.data[key] = getattr(self, key)[i:i+1]
            else:
                # safer way to deal with objects (e.g. empty list)
                self.__dict__[key] = np.empty((len(self)),dtype=object)
                self.__dict__[key][...] = [value for _ in range(len(self))]
                for i, item in enumerate(self):
                    item.data[key] = getattr(self, key)[i:i+1]

    def set_values(self, **kwargs):
        for key, value in kwargs.items():
            if key in vars(self.edgemodel).keys():
                if isinstance(value, (str, tuple)):
                    getattr(self, key)[0] = value
                elif (isinstance(value, (int, float)) and
                      isinstance(getattr(self, key)[0], tuple)):
                    getattr(self, key)[0] = (value,) # keep it as tuple
                elif hasattr(value, '__len__'):
                    if len(value) != len(getattr(self, key)):
                        print(f"error: size mismatch for {key}, required {len(getattr(self, key))}, got {len(value)}")
                    else:
                        for i, item in enumerate(value):
                            getattr(self, key)[i] = item
                else: # single value
                    for i in range(len(getattr(self, key))):
                        getattr(self, key)[i] = value
            
    def set_path(self, path):
        def trace(root, index):
            if isinstance(root, NodeGroup):
                return root[index].name
            elif isinstance(root, NodeList):
                try:
                    return root[index].name
                except AttributeError:
                    print(f"error at {root}[{index}]")
            elif isinstance(root, NodePort):
                return trace(root.link, index)
            else:
                print('error')
                return None
        
        # follow the links through ports
        self.path = path
        for i in range(len(self)):
            self[i].source_name = trace(self.source, self[i].source_index)
            self[i].target_name = trace(self.target, self[i].target_index)

# Alias class pointing to set of (external) nodes (e.g. Network Inputs)
class NodePort(list):
    def __init__(self, size=None):
        if size is not None:
            super().__init__([Link(i) for i in range(size)])
            self.size = size
        else:
            super().__init__()
            self.size = None # if no size, creates dependency
        # this is basically just a symlink
        self.path = None
        self.link = None

    def __str__(self):
        if self.path is None:
            return 'detached nodeport'
        elif self.link is not None:
            return f"(port) {self.path} <- {self.link}"
        else:
            return f"(port) {self.path} <- (no link)"

    def set_size(self, size):
        if self.size is not None:
            print(f"warning: changing port size from {self.size} to {size}")
        super().__init__([Link(i) for i in range(size)])
        self.size = size
        
    def set_path(self, path):
        self.path = path
        
    def set_link(self, link):
        self.link = link
        for i, item in enumerate(self):
            item.link = self.link[i]

# Alias class with a list of (pointers to) nodes (e.g. Network Outputs)
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
        if self.path is not None:
            return f"(list) {self.path}"
        else:
            return 'detached nodelist'
        
    def set_path(self, path):
        self.path = path