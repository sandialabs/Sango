# General Imports
from types import SimpleNamespace
from functools import reduce, wraps
import networkx as nx
import re

# Package Imports
from .core import NodeGroup, EdgeGroup, NodePort, NodeList, Link

# Turns a non-existing path to a string
class TempPath:
    def __init__(self, net, root, path):
        self.net = net
        self.root = root
        self.path = path

    def __str__(self):
        return self.path + ' (temp)'
    
    def __getattr__(self, key):
        return TempPath(self.net, self.root, '.'.join((self.path, key)))
        
    def __getitem__(self, item):
        return TempPath(self.net, self.root, f"{self.path}[{item}]")

    def __dir__(self):
        net_dir = []
        # getting dir hints for tab completion (kinda cursed)
        if isinstance(self.net, Network):
            net_path = self.net.access(self.path)
            net_dir += list(vars(net_path).keys())
            if isinstance(net_path, Network):
                net_dir += (list(vars(net_path._topology).keys())
                            + list(net_path._children.keys())
                            + list(net_path._emptylists.keys()))
        return super().__dir__() + net_dir

# Directory structure of network topology
class Topology(SimpleNamespace):
    def __init__(self, net=None, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if type(value) == dict:
                setattr(self, key, Topology(**value))
            elif type(value) == list:
                setattr(self, key, list(map(self.map_entry, value)))
        # link back to parent network if provided
        if isinstance(net, Network):
            self._network = net
        else:
            self._network = None
        # unnamed edgegroups
        self.edgegroups = list()

    # Not really used anymore
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return Topology(**entry)
        return entry

    # Traverse the path tree
    @staticmethod
    def traverse(root, name):
        if root is None:
            return
        elif isinstance(root, TempPath):
            print(f"path not found: {root.path}")
        elif isinstance(name, str):
            return getattr(root, name)
        elif isinstance(name, int):
            return root[name]
        else:
            print(f"error accessing {root} {name}")
            return
            
    # Traverse the path tree (for node lists)
    @staticmethod
    def traverse_node(root, name):
        if root is None:
            return
        elif isinstance(root, TempPath):
            print(f"path to node not found: {root.path}")
        elif isinstance(root, NodePort):
            if isinstance(name, str):
                return getattr(root.link, name)
            elif isinstance(name, int):
                return root.link[name]
            else:
                print(f"error accessing {root} {name}")
                return
        elif isinstance(name, str):
            return getattr(root, name)
        elif isinstance(name, int):
            return root[name]
        else:
            print(f"error accessing {root} {name}")
            return
    
    # Convert path string to list of attributes and indexes
    @staticmethod
    def expand_path(path):
        def find_num(string):
            start = string.find('[')
            if start == -1:
                return string, []
            numbers = re.findall(r"\[(\d+)]", string)
            return string[:start], [int(num) for num in numbers]
            
        expanded = []
        paths = path.split('.')
        for p in paths:
            string, nums = find_num(p)
            expanded.append(string)
            for num in nums:
                expanded.append(num)
        return expanded

    # Helper function for traversing the path tree
    def access(self, path):
        resolved_path = reduce(self.traverse, self.expand_path(path), self)
        if resolved_path is None:
            print(f"error resolving path {path}")
        return resolved_path

    # Helper function for traversing the path tree (for node lists)
    def access_node(self, path):
        resolved_path = reduce(self.traverse_node, self.expand_path(path), self)
        if resolved_path is None:
            print(f"error resolving path {path}")
        return resolved_path

    # Temporary path if object doesn't exist
    def __getattr__(self, key):
        return TempPath(self._network, self, key)

    def __str__(self):
        def format_paths(top):
            paths = []
            for key, value in vars(top).items():
                if key.startswith('_'):
                    continue
                if isinstance(value, Topology):
                    paths.append(format_paths(value))
                elif isinstance(value, (NodeGroup, EdgeGroup, NodePort, NodeList)):
                    paths.append(f"{value}")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, Topology):
                            paths.append(format_paths(item))
                        elif isinstance(item, (NodeGroup, EdgeGroup, NodePort, NodeList)):
                            paths.append(f"{item}")
                        else:
                            paths.append(f"{key}[{i}] is pathless")
                else:
                    paths.append(f"{key} is pathless")
            return '\n'.join(paths)
            
        return format_paths(self)

    # Add new entries
    def add(self, **kwargs):
        for key, value in kwargs.items():
            if type(value) == dict:
                setattr(self, key, Topology(**value))
            elif type(value) == list:
                setattr(self, key, list(map(self.map_entry, value)))
            else:
                setattr(self, key, value)
    
    # Generate path structure (after built)
    def flatten_paths(self):
        def _flatten_nodegroups(top, parent_path=''):
            for key, value in vars(top).items():
                if key.startswith('_'):
                    continue
                current_path = f"{parent_path}.{key}" if parent_path else key
                # If the value is another topwork, recurse
                if isinstance(value, Topology):
                    _flatten_nodegroups(value, current_path)
                # Set path for nodegroups
                elif isinstance(value, (NodeGroup, NodePort)):
                    value.set_path(current_path)
                elif isinstance(value, (NodeList, EdgeGroup)):
                    pass
                # If the value is a list, loop through 
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        list_path = f"{current_path}[{i}]"
                        if isinstance(item, Topology):
                            _flatten_nodegroups(item, list_path)
                        elif isinstance(item, (NodeGroup, NodePort)):
                            item.set_path(list_path)
                        elif isinstance(item, (NodeList, EdgeGroup)):
                            pass
                        else:
                            print(f"error at {list_path}: cannot set path for {item}")
                else:
                    print(f"error at {current_path}: cannot set path for {value}")
        def _flatten_nodelists(top, parent_path=''):
            for key, value in vars(top).items():
                if key.startswith('_'):
                    continue
                current_path = f"{parent_path}.{key}" if parent_path else key
                # If the value is another topwork, recurse
                if isinstance(value, Topology):
                    _flatten_nodelists(value, current_path)
                # If the value is a nodelist, replace any temp paths
                elif isinstance(value, NodeList):
                    for i, item in enumerate(value):
                        if isinstance(item, TempPath):
                            node = top.access_node(item.path)
                            if isinstance(node, TempPath):
                                print(f"error at {current_path}: {node.path} does not exist")
                            else:
                                value[i] = node
                        elif isinstance(item, Link):
                            if isinstance(item.link, TempPath):
                                # top level search
                                node = self.access_node(item.link.path)
                                if isinstance(node, TempPath):
                                    print(f"error at {current_path}: {node.path} does not exist")
                                else:
                                    value[i] = node
                            else:
                                value[i] = item.link
                    value.set_path(current_path)
                elif isinstance(value, (NodeGroup, NodePort, EdgeGroup)):
                    pass
                # If the value is a list, loop through 
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        list_path = f"{current_path}[{i}]"
                        if isinstance(item, Topology):
                            _flatten_nodelists(item, list_path)
                        elif isinstance(item, NodeList):
                            for e, entry in enumerate(item):
                                if isinstance(entry, TempPath):
                                    node = top.access_node(entry.path)
                                    if isinstance(node, TempPath):
                                        print(f"error at {list_path}: {node.path} does not exist")
                                    else:
                                        item[e] = node
                                elif isinstance(entry, Link):
                                    if isinstance(entry.link, TempPath):
                                        # top level search
                                        node = self.access_node(entry.link.path)
                                        if isinstance(node, TempPath):
                                            print(f"error at {list_path}: {node.path} does not exist")
                                        else:
                                            item[e] = node
                                    else:
                                        item[e] = entry.link
                            item.set_path(list_path)
                        elif isinstance(item, (NodeGroup, NodePort, EdgeGroup)):
                            pass
                        else:
                            print(f"error at {list_path}: cannot set path for {item}")
                else:
                    print(f"error at {current_path}: cannot set path for {value}")
        def _flatten_edgegroups(top, parent_path=''):
            for key, value in vars(top).items():
                if key.startswith('_'):
                    continue
                current_path = f"{parent_path}.{key}" if parent_path else key
                # If the value is another topwork, recurse
                if isinstance(value, Topology):
                    _flatten_edgegroups(value, current_path)
                # If the value is an edgegroup, replace any temp paths
                elif isinstance(value, EdgeGroup):
                    if isinstance(value.source, TempPath):
                        value.source = top.access(value.source.path)
                        if isinstance(value.source, TempPath):
                            print(f"error at {current_path}: {value.source.path} does not exist")
                    if isinstance(value.target, TempPath):
                        value.target = top.access(value.target.path)
                        if isinstance(value.target, TempPath):
                            print(f"error at {current_path}: {value.target.path} does not exist")
                    value.set_path(current_path)
                elif isinstance(value, (NodeGroup, NodePort, NodeList)):
                    pass
                # If the value is a list, loop through 
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        list_path = f"{current_path}[{i}]"
                        if isinstance(item, Topology):
                            _flatten_edgegroups(item, list_path)
                        elif isinstance(item, EdgeGroup):
                            if isinstance(item.source, TempPath):
                                item.source = top.access(item.source.path)
                                if isinstance(item.source, TempPath):
                                    print(f"error at {list_path}: {item.source.path} does not exist")
                            if isinstance(item.target, TempPath):
                                item.target = top.access(item.target.path)
                                if isinstance(item.target, TempPath):
                                    print(f"error at {list_path}: {item.target.path} does not exist")
                            item.set_path(list_path)
                        elif isinstance(item, (NodeGroup, NodePort, NodeList)):
                            pass
                        else:
                            print(f"error at {list_path}: cannot set path for {item}")
                else:
                    print(f"error at {current_path}: cannot set path for {value}")
        # Flattening order is important for resolution
        _flatten_nodegroups(self) # first for node groups (and ports)
        _flatten_nodelists(self)  # then for node lists
        _flatten_edgegroups(self) # finally for edge groups
        
    # Connect network elements
    def connect(self, source, target, model=None, edges=None):
        # Convert any temporary paths to source/target object references
        if isinstance(source, TempPath):
            source = self.access(source.path)
            if isinstance(source, TempPath):
                print(f"connection error: {source.path} does not exist")
        if isinstance(target, TempPath):
            target = self.access(target.path)
            if isinstance(target, TempPath):
                print(f"connection error: {target.path} does not exist")
        # Link ports with object references
        if isinstance(target, NodePort):
            print('linking port as target')
            if (len(source) != len(target)):
                print(f"warning: port size mismatch {len(source)} -> {len(target)}")
            target.set_link(source) # by reference
        # Connect node groups and lists with edge groups
        if (isinstance(source, (NodeGroup, NodePort, NodeList)) and
            isinstance(target, (NodeGroup, NodeList))):
            #print('adding unnamed edgegroup')
            self.edgegroups.append(EdgeGroup(source, target, model, edges))
        # Connecting individual nodes?
        # Connecting multiple edgegroups?

    # Generate a networkx graph
    def to_nx(self):
        def flatten_data(data):
            flat_data = dict()
            for key, value in data.items():
                if hasattr(value, 'base'):
                    # element of numpy array (could be more error checking)
                    flat_data[key] = value[0]
                else:
                    flat_data[key] = value
            return flat_data

        def populate(top, graph):
            for key, value in vars(top).items():
                if key.startswith('_'):
                    continue
                if isinstance(value, Topology):
                    populate(value, graph)
                elif isinstance(value, NodeGroup):
                    for node in value:
                        graph.add_node(node.name, **flatten_data(node.data))
                elif isinstance(value, EdgeGroup):
                    for edge in value:
                        graph.add_edge(edge.source_name, edge.target_name, **flatten_data(edge.data))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, Topology):
                            populate(item, graph)
                        elif isinstance(item, NodeGroup):
                            for node in item:
                                graph.add_node(node.name, **flatten_data(node.data))
                        elif isinstance(item, EdgeGroup):
                            for edge in item:
                                graph.add_edge(edge.source_name, edge.target_name, **flatten_data(edge.data))

        # launch the recursive build
        graph = nx.DiGraph()
        populate(self, graph)
        return graph

# Wrapper around Topology
# Built hierarchically around dependencies
class Network:
    def __init__(self, parent=None):
        self._topology = Topology(self)
        self._built = False
        self._graph = None
        # Scaffolding
        self._name = ''
        self._parent = parent
        self._children = dict()
        self._connections = dict()
        self._dependencies = dict()
        self._emptylists = dict()
        self._netlists = dict()

    # Special attribute assignments
    def __setattr__(self, name, value):
        if name.startswith('_'): # special attributes
            super().__setattr__(name, value)
        elif isinstance(value, NodePort):
            # add dependency for ports
            if value.size is None:
                self._dependencies[name] = value
            self.add(name, value)
        elif isinstance(value, (Network, NodeGroup, EdgeGroup, NodeList)):
            self.add(name, value)
        elif type(value) is list:
            # stash empty lists for later
            # we expect these to be generated procedurally
            if not value:
                print(f"info: adding empty list {self.net_path()}{name}")
                self._emptylists[name] = value
            # all items in list should be the same type
            elif isinstance(value[0], NodePort):
                self._dependencies[name] = value
                self.add(name, value)
            elif isinstance(value[0], (Network, NodeGroup, EdgeGroup, NodeList)):
                self.add(name, value)
            else:
                # regular attributes
                super().__setattr__(name, value)
        else:
            # regular attributes
            super().__setattr__(name, value)

    # Automatically call recursive build
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # wrap the 'build' method
        sub_build = cls.build

        @wraps(sub_build)
        def wrapped_build(self, *args, **kwargs_):
            # run subclass build first
            sub_build(self, *args, **kwargs_)
            # base class recursive build
            self.recursive_build()
            return

        cls.build = wrapped_build

    # Expose the underlying Topology
    def __getattr__(self, name):
        if name in self._emptylists:
            return self._emptylists[name]
        else:
            return getattr(self._topology, name)
    
    def __dir__(self):
        return (super().__dir__()
                + list(vars(self._topology).keys())
                + list(self._children.keys())
                + list(self._emptylists.keys()))

    def __str__(self):
        return str(self._topology)

    # Adding network elements
    def add(self, name, value):
        # stash networks for future processing
        if isinstance(value, Network):
            print(f"info: adding network {self.net_path()}{name}")
            self._children[name] = value
            value._parent = self
            value._name = f"{name}"
        # also stash lists of networks
        elif (type(value) is list):
            if isinstance(value[0], Network):
                if not all(isinstance(item, Network) for item in value):
                    print(f"error adding {self.net_path}{name} list: not all elements are networks")
                print(f"info: adding list of networks {self.net_path()}{name}")
                self._children[name] = value
                # placeholder paths for network lists
                self._netlists[name] = [TempPath(self, self._topology, f"{name}[{i}]")
                                        for i in range(len(value))]
                for i, item in enumerate(value):
                    item._parent = self
                    item._name = f"{name}[{i}]"
            else:
                setattr(self._topology, name, value)
        else:
            setattr(self._topology, name, value)

    # Adding connections (links, unnamed edgegroups)
    def connect(self, source, target, model=None, edges=None):
        # stash connections for future processing
        key = str(len(self._connections))
        self._connections[key] = (source, target, model, edges)

    # Manually setting port sizes (for breaking dependency chains)
    def set_portsize(self, port, size):
        if isinstance(port, TempPath):
            port = self.access(port.path)
        if isinstance(port, TempPath):
            print(f"error: {port.path} does not exist")
            return
        if isinstance(port, NodePort):
            port.set_size(size)
        else:
            print(f"error: {port} is not a nodeport")
        return
    
    # Traverse the path tree
    @staticmethod
    def traverse(root, name):
        if root is None:
            return
        elif isinstance(root, TempPath):
            print(f"path not found: {root.path}")
        # initialized, but unbuilt networks
        elif isinstance(root, Network):
            try:
                return root._children[name]
            except KeyError:
                if isinstance(name, str):
                    return getattr(root, name)
                elif isinstance(name, int):
                    return root[name]
                else:
                    print(f"error accessing {root} {name}")
        elif isinstance(name, str):
            return getattr(root, name)
        elif isinstance(name, int):
            return root[name]
        else:
            print(f"error accessing {root} {name}")
    
    # Convert path string to list of attributes and indexes
    @staticmethod
    def expand_path(path):
        def find_num(string):
            start = string.find('[')
            if start == -1:
                return string, []
            numbers = re.findall(r"\[(\d+)]", string)
            return string[:start], [int(num) for num in numbers]

        expanded = []
        paths = path.split('.')
        for p in paths:
            string, nums = find_num(p)
            expanded.append(string)
            for num in nums:
                expanded.append(num)
        return expanded

    # Helper function for traversing the path tree
    def access(self, path):
        resolved_path = reduce(self.traverse, self.expand_path(path), self)
        if resolved_path is None:
            print(f"error resolving path {path}")
        return resolved_path

    # Traversing up the path tree to get full path
    def net_path(self):
        if self._parent is None:
            return '' # name should be ''
        else:
            return f"{self._parent.net_path()}{self._name}."

    # Passthrough method for build
    def build(self):
        self.recursive_build()

    # Post build tasks (e.g. node lists)
    def finalize(self):
        pass

    # Incrementally and recursively build children, add connections, resolve dependencies
    def recursive_build(self):
        # Go through any previously uninitialized lists
        for name, value in self._emptylists.items():
            if not value:
                print(f"warning: setting empty list {self.net_path()}{name}")
                super().__setattr__(name, value)
            # all items in list should be the same type
            elif isinstance(value[0], NodePort):
                self._dependencies[name] = value
                self.add(name, value)
            elif isinstance(value[0], (Network, NodeGroup, EdgeGroup, NodeList)):
                self.add(name, value)
            else:
                # regular attributes
                super().__setattr__(name, value)

        # Add any placeholder netlists to the topology
        for name, value in self._netlists.items():
            setattr(self._topology, name, value)

        # Main build process
        # Basically a topological sort at each network level
        still_building = True
        children_count = len(self._children)
        build_loops = 0
        max_build_loops = 3
        while (still_building):
            # Build any bricks (hierarchically, and if dependencies met)
            children = []
            for key, value in self._children.items():
                if type(value) is list:
                    for i, item in enumerate(value):
                        # build any item that can be built (that hasn't already been built)
                        if not item._dependencies and item._built == False:
                            print(f"info: building network {self.net_path()}{key}[{i}]")
                            item.build()
                            # update the placeholder list item
                            getattr(self._topology, key)[i] = item._topology
                            # reset build loop counter
                            build_loops = 0
                    # add the list if everything built (all dependencies met)
                    if not any(item._dependencies for item in value):
                        children.append(key)
                else:
                    if not value._dependencies:
                        print(f"info: building network {self.net_path()}{key}")
                        value.build()
                        setattr(self._topology, key, value._topology)
                        children.append(key)
            # cleanup any built networks from list
            for child in children:
                del self._children[child]
            #print(f"debug: children remaining: {len(self._children)}")
            
            # Add any connections
            connections = []
            for key, (source, target, model, edges) in self._connections.items():
                # try to resolve any temporary paths
                source_path = source.path
                if isinstance(source, TempPath):
                    source = self.access(source.path)
                    if isinstance(source, TempPath):
                        print(f"info: waiting on source {self.net_path()}{source_path}")
                target_path = target.path
                if isinstance(target, TempPath):
                    target = self.access(target.path)
                    if isinstance(target, TempPath):
                        print(f"info: waiting on target {self.net_path()}{target_path}")
                # connect ports (bypass topology methods)
                if (isinstance(target, NodePort) and
                    not isinstance(source, TempPath)):
                    print(f"info: linking port {self.net_path()}{target_path}")
                    if target.size is None:
                        target.set_size(source.size)
                    elif (len(source) != len(target)):
                        print(f"error: linking port {self.net_path()}{target_path} size mismatch {len(source)} -> {len(target)}")
                    if target.link is None:
                        target.set_link(source)
                    else:
                        print(f"error linking {target}: already linked")
                    connections.append(key)
                # connect unnamed edgegroups
                if (isinstance(source, (NodeGroup, NodePort, NodeList)) and
                    isinstance(target, (NodeGroup, NodeList))):
                    print('info: adding unnamed edgegroup')
                    self._topology.connect(source, target, model, edges)
                    connections.append(key)
            # cleanup any added connections from list
            for conn in connections:
                del self._connections[conn]
            #print(f"debug: connections remaining: {len(self._connections)}")
            # connections remaining after children are built should be 0
            if ((not self._children) and self._connections):
                print("error: connections remaining after children built")
            
            # Resolve any dependencies
            for key, value in self._children.items():
                if (type(value) is list):
                    for i, item in enumerate(value):
                        item_dep_keys = []
                        for item_dep_key, item_dep_value in item._dependencies.items():
                            if isinstance(item_dep_value, NodePort):
                                if item_dep_value.size is not None:
                                    print(f"info: resolving dependency for {item.net_path()}{item_dep_key}")
                                    item_dep_keys.append(item_dep_key)
                            elif type(item_dep_value) is list:
                                if all(item_dep_item.size is not None for item_dep_item in item_dep_value):
                                    print(f"info: resolving dependency for {item.net_path()}{item_dep_key}")
                                    item_dep_keys.append(item_dep_key)
                            else:
                                print(f"error: dependency value at {item_dep_value}")
                        for item_dep_key in item_dep_keys:
                            del item._dependencies[item_dep_key]
                else:
                    dep_keys = []
                    for dep_key, dep_value in value._dependencies.items():
                        if isinstance(dep_value, NodePort):
                            if dep_value.size is not None:
                                print(f"info: resolving dependency for {value.net_path()}{dep_key}")
                                dep_keys.append(dep_key)
                        elif type(dep_value) is list:
                            if all(dep_item.size is not None for dep_item in dep_value):
                                print(f"info: resolving dependency for {value.net_path()}{dep_key}")
                                dep_keys.append(dep_key)
                        else:
                            print(f"error: dependency value at {dep_value}")
                    for dep_key in dep_keys:
                        del value._dependencies[dep_key]
                        
            # Check if done building
            if not self._children:
                still_building = False
            elif len(self._children) < children_count:
                children_count = len(self._children)
                # reset loop counter
                build_loops = 0
            else:
                build_loops += 1
                if build_loops >= max_build_loops:
                    print("error: unable to resolve dependencies")
                    # exit out of potentially infinite loop
                    still_building = False
        
        # Finalize and cleanup
        self.finalize()
        self._built = True

        # Flatten networks at top level
        if (self._parent is None):
            print('info: flattening network topology')
            self._topology.flatten_paths()
            # should this be generated by default?
            self._graph = self._topology.to_nx()

        return