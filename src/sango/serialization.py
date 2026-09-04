# Serialization of network data structures
import sys
import json
import pickle
import warnings
import numpy as np
from dataclasses import fields
from pathlib import Path

from .core import NodeGroup, EdgeGroup, NodePort, NodeList, Node, Edge, Link
from .model.base import get_shared_params
from .network import Topology, Network

# Convert numpy scalars/arrays to plain Python for JSON
def _numpy_to_python(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    return value

# Serialize a topology object to dictionary
def topology_to_dict(top):
    # Convert node/edge dataclass models to dicts
    def _model_to_dict(model):
        model_dict = {}
        for fld in fields(model):
            val = getattr(model, fld.name)
            model_dict[fld.name] = _numpy_to_python(val)
        model_dict['__class__'] = type(model).__name__
        return model_dict

    # Convert nodegroup information to dict
    def _nodegroup_to_dict(ng):
        node_dict = {'__type__': 'NodeGroup',
                     'model': _model_to_dict(ng.nodemodel),
                     'size': len(ng),
                     'path': ng.path,
                     'params': {}}
        for param_name, arr in vars(ng).items():
            if param_name in ('path', 'nodemodel', 'shared_params'):
                continue
            node_dict['params'][param_name] = _numpy_to_python(arr)
        return node_dict

    # Convert edgegroup information to dict
    def _edgegroup_to_dict(eg):
        eg_dict = {'__type__': 'EdgeGroup',
                   'model': _model_to_dict(eg.edgemodel),
                   'edges': [(int(e.source_index), int(e.target_index)) for e in eg],
                   'source_path': eg.source.path if hasattr(eg.source, 'path') else str(eg.source),
                   'target_path': eg.target.path if hasattr(eg.target, 'path') else str(eg.target),
                   'path': eg.path,
                   'params': {}}
        for param_name, arr in vars(eg).items():
            if param_name in ('path', 'edgemodel', 'source', 'target', 'edge_map', 'shared_params'):
                continue
            eg_dict['params'][param_name] = _numpy_to_python(arr)
        return eg_dict

    # Convert nodeport information to dict
    def _nodeport_to_dict(port):
        port_dict = {'__type__': 'NodePort',
                     'size': port.size,
                     'path': port.path,
                     'link_path': None}
        if port.link is not None:
            port_dict['link_path'] = port.link.path
        return port_dict
    
    # Convert nodelist information to dict
    def _nodelist_to_dict(nl):
        nl_dict = {'__type__': 'NodeList',
                   'path': nl.path,
                   'nodes': []}
        # Go through the list
        for item in nl:
            if isinstance(item, Node):
                nl_dict['nodes'].append(item.name)
            else:
                raise TypeError(f"Unexpected item found in NodeList: '{item}'")
        return nl_dict
    
    # Convert topology information to dict
    def _topology_to_dict(top):
        # Initialize the topology dictionary
        top_dict = {'__type__': 'Topology'}
        # Recursively go through topology
        for key, value in vars(top).items():
            if key.startswith('_'):
                continue
            if isinstance(value, Topology):
                top_dict[key] = _topology_to_dict(value)
            elif isinstance(value, NodeGroup):
                top_dict[key] = _nodegroup_to_dict(value)
            elif isinstance(value, EdgeGroup):
                top_dict[key] = _edgegroup_to_dict(value)
            elif isinstance(value, NodePort):
                top_dict[key] = _nodeport_to_dict(value)
            elif isinstance(value, NodeList):
                top_dict[key] = _nodelist_to_dict(value)
            elif isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, Topology):
                        items.append(_topology_to_dict(item))
                    elif isinstance(item, NodeGroup):
                        items.append(_nodegroup_to_dict(item))
                    elif isinstance(item, EdgeGroup):
                        items.append(_edgegroup_to_dict(item))
                    elif isinstance(item, NodePort):
                        items.append(_nodeport_to_dict(item))
                    elif isinstance(item, NodeList):
                        items.append(_nodelist_to_dict(item))
                    else:
                        raise TypeError(f"Unexpected item found in Topology: '{item}'")
                top_dict[key] = items
            else:
                raise TypeError(f"Unexpected item found in Topology: '{value}'")
        return top_dict

    # Construct the topology dictionary from root
    top_dict = _topology_to_dict(top)
    return top_dict

# Reconstruct a Topology object from a dict
def topology_from_dict(top_dict):
    # Reconstruct node/edge dataclass models from dict
    def _model_from_dict(model_dict):
        from . import model as model_pkg
        cls_name = model_dict['__class__']
        # Search the built-in models and then __main__
        cls = getattr(model_pkg, cls_name, None)
        if cls is None:
            main_module = sys.modules.get('__main__')
            if main_module is not None:
                cls = getattr(main_module, cls_name, None)
        # Error if class is not found
        if cls is None:
            raise TypeError(
                f"Unable to find model class '{cls_name}'. "
                f"Make sure the class is defined or imported before deserializing."
            )
        # Construct the model class with its params
        valid_fields = {fld.name for fld in fields(cls)}
        filtered = {key: value for key, value in model_dict.items() if key in valid_fields}
        return cls(**filtered)

    # Reconstruct nodegroup information from dict
    def _nodegroup_from_dict(ng_dict):
        model = _model_from_dict(ng_dict['model'])
        size = ng_dict['size']
        params = ng_dict.get('params', {})
        # Shared fields are handled separately from per-instance values
        shared_fields = get_shared_params(model)
        unique_values = {}
        shared_values = {}
        for key, value in params.items():
            if key in shared_fields:
                # Shared params are serialized as single-element lists
                shared_values[key] = value[0]
            else:
                unique_values[key] = value
        # Construct node group with unique values
        ng = NodeGroup(model, size, **unique_values)
        # Restore any shared param values
        ng.set_values(**shared_values)
        # Set path
        ng.set_path(ng_dict.get('path'))
        return ng

    # Reconstruct edgegroup information from dict
    # Requires topology structure and nodes to be resolved
    def _edgegroup_from_dict(eg_dict, ref_top):
        model = _model_from_dict(eg_dict['model'].copy())
        edges = [tuple(edge) for edge in eg_dict['edges']]
        params = eg_dict.get('params', {})
        #  Resolve source/target from a topology reference
        source = ref_top.access(eg_dict.get('source_path'))
        target = ref_top.access(eg_dict.get('target_path'))
        # Shared fields are handled separately from per-instance values
        shared_fields = get_shared_params(model)
        unique_values = {}
        shared_values = {}
        for key, value in params.items():
            if key in shared_fields:
                # Shared params are serialized as single-element lists
                shared_values[key] = value[0]
            else:
                unique_values[key] = value
        # Construct edge group with unique values
        eg = EdgeGroup(source, target, model, edges=edges, **unique_values)
        # Restore any shared param values
        eg.set_values(**shared_values)
        # Set path
        eg.set_path(eg_dict.get('path'))
        return eg

    # Reconstruct nodeport information
    # Requires topology structure to be built
    def _resolve_nodeport(port, ref_top):
        # Set link path
        link_path = getattr(port, '_link_path')
        if link_path is not None:
            port.set_link(ref_top.access(link_path))
        # Clean up the temporary attribute
        del port._link_path

    # Reconstruct nodelist information
    # Requires topology structure to be built
    def _resolve_nodelist(nl, ref_top):
        for i, node_name in enumerate(nl):
            resolved = ref_top.access_node(node_name)
            if isinstance(resolved, Node):
                nl[i] = resolved
            else:
                raise TypeError(f"NodeList item not resolved: '{node_name}'")

    # Recursively construct topology (with placeholders)
    def _topology_from_dict(top_dict):
        top = Topology()
        for key, value in top_dict.items():
            if key.startswith('_'):
                continue
            if isinstance(value, dict):
                value_type = value.get('__type__')
                if value_type == 'Topology':
                    setattr(top, key, _topology_from_dict(value))
                elif value_type == 'NodeGroup':
                    setattr(top, key, _nodegroup_from_dict(value))
                elif value_type == 'EdgeGroup':
                    # Store the raw dict for later
                    setattr(top, key, value)
                elif value_type == 'NodePort':
                    port = NodePort(value.get('size'))
                    port.path = value.get('path')
                    port._link_path = value.get('link_path')
                    setattr(top, key, port)
                elif value_type == 'NodeList':
                    nl = NodeList()
                    nl.path = value.get('path')
                    for node_name in value.get('nodes', []):
                        nl.append(node_name) # raw names placeholder
                    setattr(top, key, nl)
                else:
                    raise TypeError(f"Unexpected item found in Topology: '{value}'")
            elif isinstance(value, list):
                items = []
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        item_type = item.get('__type__')
                        if item_type == 'Topology':
                            items.append(_topology_from_dict(item))
                        elif item_type == 'NodeGroup':
                            items.append(_nodegroup_from_dict(item))
                        elif item_type == 'EdgeGroup':
                            items.append(item) # raw dict placeholder
                        elif item_type == 'NodePort':
                            port = NodePort(item.get('size'))
                            port.path = item.get('path')
                            port._link_path = item.get('link_path')
                            items.append(port)
                        elif item_type == 'NodeList':
                            nl = NodeList()
                            nl.path = item.get('path')
                            for node_name in item.get('nodes', []):
                                nl.append(node_name) # raw names placeholder
                            items.append(nl)
                        else:
                            raise TypeError(f"Unexpected item found in Topology: '{item}'")
                    else:
                        raise TypeError(f"Unexpected item found in Topology: '{item}'")
                setattr(top, key, items)
            else:
                raise TypeError(f"Unexpected item found in Topology: '{value}'")
        return top

    # Resolve NodePort links and NodeList nodes
    def _resolve_node_references(top, ref_top):
        # Recursively go through topology
        for key, value in vars(top).items():
            if key.startswith('_'):
                continue
            if isinstance(value, Topology):
                _resolve_node_references(value, ref_top)
            elif isinstance(value, NodePort):
                _resolve_nodeport(value, ref_top)
            elif isinstance(value, NodeList):
                _resolve_nodelist(value, ref_top)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Topology):
                        _resolve_node_references(item, ref_top)
                    elif isinstance(item, NodePort):
                        _resolve_nodeport(item, ref_top)
                    elif isinstance(item, NodeList):
                        _resolve_nodelist(item, ref_top)

    # Reconstruct EdgeGroups from fully resolved nodes
    def _reconstruct_edges(top, ref_top):
        # Recursively go through topology
        for key, value in list(vars(top).items()):
            if key.startswith('_'):
                continue
            if isinstance(value, Topology):
                _reconstruct_edges(value, ref_top)
            elif isinstance(value, dict) and value.get('__type__') == 'EdgeGroup':
                setattr(top, key, _edgegroup_from_dict(value, ref_top))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Topology):
                        _reconstruct_edges(item, ref_top)
                    elif isinstance(item, dict) and item.get('__type__') == 'EdgeGroup':
                        value[i] = _edgegroup_from_dict(item, ref_top)
    
    # First pass, get overall topology structure of components
    top = _topology_from_dict(top_dict)
    # Second pass, resolve NodePort links and NodeList nodes
    _resolve_node_references(top, top)
    # Third pass, reconstruct EdgeGroups now that all references are resolved
    _reconstruct_edges(top, top)
    # Return the fully reconstructed topology object
    return top

# Check if an object looks like a state_dict: dict(str->array)
def _is_state_dict(obj):
    if not isinstance(obj, dict) or not obj:
        return False
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if not isinstance(v, (np.ndarray, list)):
            return False
    return True

# Convert a state_dict to JSON (with dtypes)
def statedict_to_json(sd):
    json_dict = {'__type__': 'state_dict'} # state_dict identifier
    for key, arr in sd.items():
        arr = np.asarray(arr)
        json_dict[key] = {'data': arr.tolist(),
                          'dtype': str(arr.dtype)}
    return json_dict

# Reconstruct a state_dict from JSON (with dtypes)
def statedict_from_json(json_dict):
    sd = {}
    for key, value in json_dict.items():
        if key == '__type__': # ignore the state_dict identifier
            continue
        sd[key] = np.asarray(value['data'], dtype=np.dtype(value['dtype']))
    return sd

# Convert network information to dict
def network_to_dict(net):
    net_dict = {'__type__': 'Network',
                'structure': net.structure(),
                'topology': topology_to_dict(net._topology)}
    return net_dict

# Reconstruct a Network object from a dictionary
def network_from_dict(net_dict):
    # Create the Network wrapper and attach Topology objects
    def _reconstruct_network(struct, top):
        # Try to resolve network (sub)class name
        module_name, cls_name = struct['class'].rsplit('.', 1)
        mod = sys.modules.get(module_name)
        if mod is not None:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                warnings.warn(
                    f"Could not resolve network class '{struct['class']}'; "
                    f"falling back to {Network.__name__}.",
                    category=UserWarning,
                    stacklevel=3)
                cls = Network
        # Instantiate without calling build (already built)
        net = Network.__new__(cls)
        # Initialize saffolding attributes from init
        net._topology = top
        net._built = True
        net._graph = None
        net._name = ''
        net._parent = None
        net._children = {}
        net._bindings = {}
        net._dependencies = {}
        net._emptylists = {}
        net._netlists = {}
        # Restore user-defined instance attributes
        for key, value in struct.get('param', {}).items():
            object.__setattr__(net, key, value)
        # Recurse into children
        child_structs = struct.get('child', {})
        for child_name, child_struct in child_structs.items():
            # Find the corresponding sub-topology
            sub_top = getattr(top, child_name)
            if isinstance(child_struct, list):
                # List of child networks
                child_list = []
                for i, cs in enumerate(child_struct):
                    child_net = _reconstruct_network(cs, sub_top[i])
                    child_net._name = f"{child_name}[{i}]"
                    child_net._parent = net
                    child_list.append(child_net)
                net._children[child_name] = child_list
            elif isinstance(child_struct, dict):
                # Standalone child network
                child_net = _reconstruct_network(child_struct, sub_top)
                child_net._name = child_name
                child_net._parent = net
                net._children[child_name] = child_net
        return net

    # Reconstruct topology first
    top = topology_from_dict(net_dict['topology'])
    # Reconstruct network from structure
    struct = net_dict['structure']
    net = _reconstruct_network(struct, top)
    # Return reconstructed network
    return net

# Automatically get file format from extension
def _format_from_ext(suffix):
    format_map = {'.json': 'json',
                  '.pkl': 'pickle',
                  '.pickle': 'pickle',
                  '.pt': 'pickle'}
    # JSON by default
    return format_map.get(suffix, 'json')

# Save network objects to disk
def save(obj, path, format=None, **kwargs):
    path = Path(path)
    # Guess file format if none provided
    if format is None:
        suffix = path.suffix.lower()
        format = _format_from_ext(suffix)
    
    if format == 'json':
        indent = kwargs.get('indent', 2)
        if isinstance(obj, Network):
            obj_dict = network_to_dict(obj)
        elif isinstance(obj, Topology):
            obj_dict = topology_to_dict(obj)
        elif _is_state_dict(obj):
            obj_dict = statedict_to_json(obj)
        else:
            # Try to dump to JSON directly
            obj_dict = obj
        with open(path, 'w') as file:
            json.dump(obj_dict, file, indent=indent, default=_numpy_to_python)
    elif format == 'pickle':
        protocol = kwargs.get('protocol', pickle.HIGHEST_PROTOCOL)
        with open(path, 'wb') as file:
            pickle.dump(obj, file, protocol=protocol)
    else:
        raise ValueError(f"Unknown format: {format!r}. "
                         "Use 'json' or 'pickle'.")

# Load network objects from disk
def load(path, format=None):
    path = Path(path)
    # Guess file format if none provided
    if format is None:
        suffix = path.suffix.lower()
        format = _format_from_ext(suffix)

    if format == 'json':
        with open(path, 'r') as file:
            json_dict = json.load(file)
        if json_dict.get('__type__') == 'Network':
            return network_from_dict(json_dict)
        elif json_dict.get('__type__') == 'Topology':
            return topology_from_dict(json_dict)
        elif json_dict.get('__type__') == 'state_dict':
            return statedict_from_json(json_dict)
        else:
            return json_dict
    elif format == 'pickle':
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError(f"Unknown format: {format!r}. "
                         "Use 'json' or 'pickle'.")
