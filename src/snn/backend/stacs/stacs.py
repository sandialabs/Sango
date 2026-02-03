import os
import sys
from pathlib import Path
import importlib.util

import time
import yaml
import subprocess
from collections import Counter, defaultdict
from contextlib import ExitStack

import numpy as np
import matplotlib.pyplot as plt

# STACS Simulation Backend
class SimSTACS:
    def __init__(self, net):
        self.net = net
        self.num_streams = 0
        self.num_nodes = 0
        self.num_edges = 0
        
        self.netwkdir = './dslnet'
        self.filebase = 'network'
        self.recordir = 'record'
        self.netparts = 1
        self.netfiles = 1
        
        self.spike_list = None
        
        self.ticks_per_ms = 1000000
        self.timesteps = None

        self.input_model = {'SI':  {'name': 'spike_input',
                                    'target': 'IN',
                                    'edge': 'SC',
                                    'port': 'input/spike_input.yml'}}
        self.model_registry = self.import_registry()

    # Dynamically import the model registry files
    def import_registry(self):
        registry = dict()
        registry_dir = Path(__file__).resolve().parent / 'registry'
        sys.path.insert(0, str(registry_dir.parent))
        for file_path in registry_dir.glob("*.py"):
            module_name = f"registry.{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            registry.update(module.model_registry)
        sys.path.pop(0)
        return registry

    # Convert between dsl model to stacs model parameters
    def rekey_param(self, data):
        param = []
        for key, value in self.model_registry[data['model']]['param'].items():
            if value['dsl'] is not None:
                param.append(data[value['dsl']][0])
            else:
                param.append(value['default'])
        return tuple(param)

    # Convert between dsl model to stacs model
    def rekey_model(self, data):
        for key, value in self.model_registry[data['model']]['state'].items():
            if value['dsl'] is not None:
                data[key] = data.pop(value['dsl'])
            else:
                data[key] = value['default']
        return data

    def compile(self, netparts=1, netfiles=1, input=True, prefix='./dslnet',
                write_fileinit=False, debug=False):
        self.netparts = netparts
        self.netfiles = netfiles
        self.has_input = input
        self.netwkdir = prefix
        self.fileinit = write_fileinit
        self.debug = debug

        # Convert network to stacs
        start_time = time.perf_counter()
        self.to_stacs()
        self.write_yaml()
        if self.fileinit:
            self.write_file()
        self.write_dcsr()
        end_time = time.perf_counter()
        self.compile_time = end_time - start_time
        if self.debug:
            print(f"Compile time: {self.compile_time}")

    # Run the network
    def run(self, timesteps=10.0, num_pe=None, runmode=None, stacsdir='~/stacs', verbose=False):
        # Path to STACS executables
        self.stacsdir = stacsdir
        self.charmrun = f"{self.stacsdir}/charmrun"
        self.stacsbin = f"{self.stacsdir}/stacs"
        
        # Simulation arguments
        runconf = f"{self.netwkdir}/{self.filebase}.yml"
        if num_pe is None:
            charm_pe = '+p' + str(self.netparts)
        else:
            charm_pe = '+p' + str(num_pe)
        charmrun = os.path.realpath(os.path.expanduser(self.charmrun))
        stacsbin = os.path.realpath(os.path.expanduser(self.stacsbin))
        self.timesteps = float(timesteps)
        self.verbose = verbose
        
        # Base command to run STACS
        runcmd = charmrun + ' ' + charm_pe + ' ' + stacsbin + ' ' + runconf
        if self.verbose:
            print(runcmd)
        
        # Modify the simulation configuration
        with open(runconf,"r") as file:
            config_yaml = yaml.safe_load(file)
        config_yaml['tmax'] = self.timesteps
        with open(runconf,"w") as file:
            yaml.dump(config_yaml,file,sort_keys=False)

        # Call STACS
        start_time = time.perf_counter()
        runlist = runcmd.split()
        if runmode is not None:
            runlist.append(runmode)
        if (self.verbose):
            subprocess.run(runlist)
        else:
            subprocess.run(runlist, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        end_time = time.perf_counter()
        self.run_time = end_time - start_time
        if self.verbose:
            print(f"Run time: {self.run_time}")

    # Convert the network topology to STACS
    def to_stacs(self):
        # Get a flattened graph object
        stacs_network = self.net._topology.to_nx()
        
        # Convert to SNN-dCSR
        if self.has_input:
            self.num_streams = len(self.input_model)
        self.num_nodes = stacs_network.number_of_nodes() + self.num_streams
        self.num_edges = 0 # undirected edges (computed later)
        
        self.node_map = dict()
        self.node_data = [dict() for _ in range(self.num_nodes)]
        self.edge_data = [dict() for _ in range(self.num_nodes)] # this will be dict of dicts
        self.group_count = Counter()
        self.local_index = [0 for _ in range(self.num_nodes)]
        self.node_set = defaultdict(dict)
        self.edge_set = defaultdict(dict)
        
        # Add special spike_input node (stream)
        input_targets = []
        input_source = dict()
        self.spike_input = dict()
        if self.has_input:
            for n, (name, value) in enumerate(self.input_model.items()):
                self.node_map[value['name']] = n
                self.node_data[n] = {'model': name, 'group_name': name}
                self.group_count.update([name])
                self.local_index[n] = self.group_count[name] - 1
                input_targets.append(value['target'])
                input_source[value['target']] = name
        
        # Nodes and indexes
        for n, (node, data) in enumerate(stacs_network.nodes(data=True)):
            index = n + self.num_streams
            self.node_map[node] = index
            self.node_data[index] = self.rekey_model(data)
            group_param = self.rekey_param(data)
            if group_param not in self.node_set[self.node_data[index]['model']]:
                self.node_set[self.node_data[index]['model']][group_param] = len(self.node_set[self.node_data[index]['model']])
            if self.node_set[self.node_data[index]['model']][group_param] == 0:
                group_name = f"{self.node_data[index]['model']}"
            else:
                group_name = f"{self.node_data[index]['model']}_{self.node_set[self.node_data[index]['model']][group_param]}"
            self.node_data[index]['group_name'] = group_name
            self.group_count.update([group_name])
            self.local_index[index] = self.group_count[group_name] - 1
            if self.has_input and data['model'] in input_targets:
                source_index = self.node_map[self.input_model[input_source[data['model']]]['name']]
                edge_model = self.input_model[input_source[data['model']]]['edge']
                model_name = {'model': edge_model}
                default_states = {key: item['default'] for key, item in self.model_registry[edge_model]['state'].items()}
                self.edge_data[index][source_index] = {**model_name, **default_states}
                self.edge_data[source_index][index] = None
                self.edge_set[(self.edge_data[index][source_index]['model'], self.node_data[source_index]['group_name'], self.node_data[index]['group_name'])][None] = 0
                edge_name = f"{self.edge_data[index][source_index]['model']}_{self.node_data[source_index]['group_name']}_{self.node_data[index]['group_name']}"
                self.edge_data[index][source_index]['group_name'] = edge_name
                self.spike_input[index] = data['times']
                
        # Get the model insertion order of our counter
        self.group_index = {key: index for index, key in enumerate(self.group_count.keys())}
        
        # Edges (to semi-undirected format)
        for source, target, data in stacs_network.edges(data=True):
            s = self.node_map[source]
            t = self.node_map[target]
            self.edge_data[t][s] = self.rekey_model(data)
            edge_tuple = (self.edge_data[t][s]['model'], self.node_data[s]['group_name'], self.node_data[t]['group_name'])
            group_param = self.rekey_param(data)
            if group_param not in self.edge_set[edge_tuple]:
                self.edge_set[edge_tuple][group_param] = len(self.edge_set[edge_tuple])
            edge_name = f"{self.edge_data[t][s]['model']}_{self.node_data[s]['group_name']}_{self.node_data[t]['group_name']}"
            if self.edge_set[edge_tuple][group_param] == 0:
                group_name = edge_name
            else:
                group_name = f"{edge_name}__{self.edge_set[edge_tuple][group_param]}"
            self.edge_data[t][s]['group_name'] = group_name
            if t not in self.edge_data[s]:
                self.edge_data[s][t] = None
        
        # Sort edges
        for n in range(self.num_nodes):
            self.edge_data[n] = dict(sorted(self.edge_data[n].items()))
            self.num_edges += len(self.edge_data[n])
    
    # Create the network directory for the dCSR files, and write the YAML files
    def write_yaml(self):
        # Returns a substrate model dictionary
        def substrate_model(model_name, model_type, params=None, states=None, ports=None):
            # Initialize model dictionary
            model_dict = dict()
            model_dict['type'] = self.model_registry[model_type]['graph_type']
            model_dict['modname'] = model_name
            model_dict['modtype'] = self.model_registry[model_type]['model_type']
            
            # Parameters (shared by model instances)
            if params is not None:
                model_dict['param'] = list()
                for k,v in params.items():
                    model_dict['param'].append({'name': k, 'value': v})
            
            # States (mutable or unique per model instance)
            if states is not None:
                model_dict['state'] = list()
                for k,v in states.items():
                    if isinstance(v, dict):
                        model_dict['state'].append({'name': k, **v})
                    else:
                        model_dict['state'].append({'name': k, 'init': 'constant', 'value': v})
            
            # Ports (for external communication)
            if ports is not None:
                model_dict['port'] = list()
                for k,v in ports.items():
                    model_dict['port'].append({'name': k, 'value': v})
            
            # Return the model dictionary
            return model_dict
        
        # Returns a graph model dictionary
        def graph_vertex(model_name, order, shape=None):
            # Initialize model dictionary
            model_dict = dict()
            model_dict['type'] = 'vertex'
            model_dict['modname'] = model_name
            model_dict['order'] = order
        
            # Populations have a shape
            if shape is not None:
                model_dict.update(shape)
            else:
                model_dict.update({'shape': 'point'})
            model_dict['coord'] = [0.0, 0.0, 0.0]
            
            # Return the model dictionary
            return model_dict
        
        def graph_stream(model_name):
            # Initialize model dictionary
            model_dict = dict()
            model_dict['type'] = 'stream'
            model_dict['modname'] = model_name
            model_dict['coord'] = [0.0, 0.0, 0.0]
            
            # Return the model dictionary
            return model_dict
        
        def graph_edge(source, target, model_name, distance=None, connect=None, cutoff=None):
            # Initialize model dictionary
            model_dict = dict()
            model_dict['type'] = 'edge'
            model_dict['source'] = source
            model_dict['target'] = [target]
            model_dict['modname'] = model_name
        
            # Distance computation (default is euclidean)
            if distance is not None:
                for k,v in distance.items():
                    model_dict['dist'] = k
                    model_dict.update(v)
            
            # Connection parameterization
            model_dict['connect'] = list()
            if connect is not None:
                for k,v in connect.items():
                    if isinstance(v, dict):
                        model_dict['connect'].append({'type': k, **v})
            else:
                model_dict['connect'].append({'type': 'uniform', 'prob': 0.0})
            
            # Connection cutoff distances
            if cutoff is not None:
                model_dict['cutoff'] = cutoff
            else:
                model_dict['cutoff'] = 0.0
                
            # Return the model dictionary
            return model_dict
        
        # Reorganize the list of model dictionaries
        def generate_model_yaml(substrate_models):
            # Reorder models in stream, vertex, edge order
            sort_key = {'stream': 0, 'vertex': 1, 'edge': 2}
            substrate_list = sorted(substrate_models,
                                    key = lambda x: sort_key[x['type']])
            
            # Return model dictionary list
            return substrate_list
        
        # Collect the different graph models into a combined dictionary
        def generate_graph_yaml(graph_models):
            graph_dict = dict()
            graph_dict['stream'] = list()
            graph_dict['vertex'] = list()
            graph_dict['edge'] = list()
        
            # Add to collection and remove 'type'
            for model in graph_models:
                if model['type'] == 'stream':
                    graph_dict['stream'].append(model)
                if model['type'] == 'vertex':
                    graph_dict['vertex'].append(model)
                if model['type'] == 'edge':
                    graph_dict['edge'].append(model)
                    
            # Return model dictionary
            return graph_dict
        
        # Returns a simulation configuration
        def generate_conf_yaml(sim_conf, part_conf, time_conf):
            conf_dict = dict()
        
            # General simulation information
            conf_dict.update(sim_conf)
            conf_dict.setdefault('runmode', 'simulate')
            conf_dict.setdefault('plastic', True)
            conf_dict.setdefault('episodic', False)
            conf_dict.setdefault('loadbal', False)
            conf_dict.setdefault('selfconn', True)
        
            # Network partitions
            conf_dict.update(part_conf)
            conf_dict.setdefault('recordir', 'record')
            conf_dict.setdefault('groupdir', 'group')
            conf_dict.setdefault('fileload', '')
            conf_dict.setdefault('filesave', '.out')
        
            # Simulation timing
            conf_dict.update(time_conf)
            conf_dict.setdefault('tstep', 1.0)
            conf_dict.setdefault('teventq', 20.0)
            conf_dict.setdefault('tdisplay', 1000.0)
            conf_dict.setdefault('trecord', 10000.0)
            conf_dict.setdefault('tsave', 1000000.0)
            conf_dict.setdefault('tbalance', 1000000.0)
        
            # Return model dictionary
            return conf_dict
    
        # Create directory for simulation
        os.makedirs(self.netwkdir, exist_ok=True)
        os.makedirs(f"{self.netwkdir}/files", exist_ok=True)
        os.makedirs(f"{self.netwkdir}/input", exist_ok=True)
        os.makedirs(f"{self.netwkdir}/record", exist_ok=True)
        
        # Neuron and synapse dynamics are written in C++
        # but are parameterized through a YAML configuration file
        substrate_models = []
        
        # Stream models
        if self.has_input:
            for name, value in self.input_model.items():
                stream_param = {'n': self.group_count[value['target']]}
                stream_port = {'input': self.input_model[name]['port']}
                substrate_models.append(substrate_model(name, name, params=stream_param, ports=stream_port))
                # Write the input files too
                input_list = []
                for times in self.spike_input.values():
                    input_list.append([float(t) for t in times])
                fname = f"{self.netwkdir}/{self.input_model[name]['port']}"
                with open(fname,"w") as file:
                    yaml.dump({'spike_list': input_list}, file, sort_keys=False)

        # Vertex models
        for name, groups in self.node_set.items():
            model_params = dict()
            model_states = dict()
            for group, g in groups.items(): # currently only supports one group
                if g == 0:
                    group_name = name
                else:
                    group_name = f"{name}_{g}"
                for k, key in enumerate(self.model_registry[name]['param'].keys()):
                    model_params[key] = group[k]
                for key, item in self.model_registry[name]['state'].items():
                    if item['dsl'] is None:
                        model_states[key] = {'init': 'constant', 'value': item['default']}
                    else:
                        model_states[key] = {'init': 'file', 'filetype': 'csv-dense',
                                             'filename': f"files/{name}_{key}.csv"}
                    if 'rep' in item and item['rep'] == 'tick':
                        model_states[key].update({'rep': 'tick'})
                substrate_models.append(substrate_model(group_name, name, params=model_params, states=model_states))

        # Edge models
        for (name, source, target), groups in self.edge_set.items():
            full_name = f"{name}_{source}_{target}"
            model_params = dict()
            model_states = dict()
            for group, g in groups.items(): # currently only supports one group
                if g == 0:
                    group_name = full_name
                else:
                    group_name = f"{full_name}__{g}"
                for k, key in enumerate(self.model_registry[name]['param'].keys()):
                    model_params[key] = group[k]
                for key, item in self.model_registry[name]['state'].items():
                    if item['dsl'] is None:
                        model_states[key] = {'init': 'constant', 'value': item['default']}
                    else:
                        model_states[key] = {'init': 'file', 'filetype': 'csv-sparse',
                                             'filename': f"files/{group_name}_{key}.csv"}
                    if 'rep' in item and item['rep'] == 'tick':
                        model_states[key].update({'rep': 'tick'})
                substrate_models.append(substrate_model(group_name, name, params=model_params, states=model_states))

        # Network structure is parallelized through Charm++
        # and is also parameterized through a YAML configuration file
        graph_models = []
        
        # Create the different network populations
        if self.has_input:
            for name, value in self.input_model.items():
                graph_models.append(graph_stream(name))
        
        # Populations are using the same name as the model name
        for name, count in self.group_count.items():
            if name in self.input_model:
                continue
            graph_models.append(graph_vertex(name, count))
        
        # Create the different network connections
        for (edge_name, source_name, target_name), groups in self.edge_set.items():
            full_name = f"{edge_name}_{source_name}_{target_name}"
            for group, g in groups.items():
                if g == 0:
                    group_name = full_name
                else:
                    group_name = f"{full_name}__{g}"
                model_conn = {'file': {'filetype': 'csv-sparse', 'filename': f"files/{group_name}_delay.csv"}}
                graph_models.append(graph_edge(source_name, target_name, group_name, connect=model_conn))
            
        # Generate the model file
        fname = f"{self.netwkdir}/{self.filebase}.model"
        with open(fname,"w") as file:
            # Each of these are their own YAML file
            substrate_yaml = generate_model_yaml(substrate_models)
            yaml.dump_all(substrate_yaml, file, sort_keys=False)

        # Generate the graph file
        fname = f"{self.netwkdir}/{self.filebase}.graph"
        with open(fname,"w") as file:
            # Graph information is one file
            graph_yaml = generate_graph_yaml(graph_models)
            yaml.dump(graph_yaml, file, sort_keys=False)

        # Generate main configuration
        sim_conf = {'runmode': 'simulate',
                    'randseed': 1421}
        
        # Partitions
        part_conf = {'netwkdir': self.netwkdir,
                     'recordir': self.recordir,
                     'filebase': self.filebase,
                     'netfiles': self.netfiles,
                     'netparts': self.netparts}
        
        # Timing
        time_conf = {'tmax': self.timesteps} # ms
        
        # Generate the simulation configuration file
        fname = f"{self.netwkdir}/{self.filebase}.yml"
        with open(fname,"w") as file:
            conf_yaml = generate_conf_yaml(sim_conf, part_conf, time_conf)
            yaml.dump(conf_yaml, file, sort_keys=False)

    # Write the topology out to dCSR
    def write_dcsr(self):
        # Parts to files bookkeeping
        part_div = self.netparts // self.netfiles
        part_rem = self.netparts % self.netfiles
        num_part = [0 for _ in range(self.netfiles)]
        part_prefix = [0 for _ in range(self.netfiles + 1)]
        for fileidx in range(self.netfiles):
            if fileidx < part_rem:
                num_part[fileidx] = part_div + 1
                part_prefix[fileidx] = fileidx * part_div + fileidx
            else:
                num_part[fileidx] = part_div
                part_prefix[fileidx] = fileidx * part_div + part_rem
        part_prefix[self.netfiles] = self.netparts
        
        # Nodes to parts bookkeeping
        node_div = self.num_nodes // self.netparts
        node_rem = self.num_nodes % self.netparts
        node_part = [0 for _ in range(self.netparts)]
        node_prefix = [0 for _ in range(self.netparts + 1)]
        for partidx in range(self.netparts):
            if partidx < node_rem:
                node_part[partidx] = node_div + 1
                node_prefix[partidx] = partidx * node_div + partidx
            else:
                node_part[partidx] = node_div
                node_prefix[partidx] = partidx * node_div + node_rem
        node_prefix[self.netparts] = self.num_nodes
        
        # Adjcy
        for fileidx in range(self.netfiles):
            fname = f"{self.netwkdir}/{self.filebase}.adjcy.{fileidx}"
            with open(fname,"w") as file:
                for n in range(node_prefix[part_prefix[fileidx]], node_prefix[part_prefix[fileidx+1]]):
                    #print(' ' + ' '.join(str(key) for key in self.edge_data[n]))
                    file.write(' ' + ' '.join(str(key) for key in self.edge_data[n]) + '\n')
    
        # State
        edge_prefix = [0 for _ in range(self.netparts + 1)]
        state_prefix = [0 for _ in range(self.netparts + 1)]
        stick_prefix = [0 for _ in range(self.netparts + 1)]
        for fileidx in range(self.netfiles):
            fname = f"{self.netwkdir}/{self.filebase}.state.{fileidx}"
            with open(fname,"w") as file:
                for partidx in range(part_prefix[fileidx], part_prefix[fileidx+1]):
                    edge_prefix[partidx+1] = edge_prefix[partidx]
                    state_prefix[partidx+1] = state_prefix[partidx]
                    stick_prefix[partidx+1] = stick_prefix[partidx]
                    for target in range(node_prefix[partidx], node_prefix[partidx+1]):
                        info = []
                        # nodes
                        #info.append(self.node_data[target]['model'])
                        info.append(self.node_data[target]['group_name'])
                        state_info = []
                        stick_info = []
                        for key, value in self.model_registry[self.node_data[target]['model']]['state'].items():
                            if 'rep' in value and value['rep'] == 'tick':
                                stick_info.append(f'{int(self.node_data[target][key])*self.ticks_per_ms:x}')
                                stick_prefix[partidx+1] += 1
                            else:
                                state_info.append(str(self.node_data[target][key]))
                                state_prefix[partidx+1] += 1
                        info.extend(state_info + stick_info)
                        # edges
                        for source, value in self.edge_data[target].items():
                            edge_prefix[partidx+1] += 1
                            if value is None:
                                info.append('none')
                            else:
                                #info.append(f"{value['model']}_{self.node_data[source]['model']}_{self.node_data[target]['model']}")
                                info.append(f"{value['group_name']}")
                                # states then sticks
                                state_info = []
                                stick_info = []
                                for key, item in self.model_registry[value['model']]['state'].items():
                                    if 'rep' in item and item['rep'] == 'tick':
                                        stick_info.append(f'{int(value[key])*self.ticks_per_ms:x}')
                                        stick_prefix[partidx+1] += 1
                                    else:
                                        state_info.append(str(value[key]))
                                        state_prefix[partidx+1] += 1
                                info.extend(state_info + stick_info)
                        #print(' ' + ' '.join(info))
                        file.write(' ' + ' '.join(info) + '\n')
    
        # Index
        for fileidx in range(self.netfiles):
            fname = f"{self.netwkdir}/{self.filebase}.index.{fileidx}"
            with open(fname,"w") as file:
                for n in range(node_prefix[part_prefix[fileidx]], node_prefix[part_prefix[fileidx+1]]):
                    # print(' ' + ' '.join(str(index) for index in [n, self.group_index[self.node_data[n]['model']],
                    #                                               self.local_index[n]]))
                    file.write(' ' + ' '.join(str(index) for index in [n, self.group_index[self.node_data[n]['group_name']],
                                                                       self.local_index[n]]) + '\n')
            
        # Coord
        for fileidx in range(self.netfiles):
            fname = f"{self.netwkdir}/{self.filebase}.coord.{fileidx}"
            with open(fname,"w") as file:
                for n in range(node_prefix[part_prefix[fileidx]], node_prefix[part_prefix[fileidx+1]]):
                    # print(' ' + ' '.join(str(0.0) for _ in range(3)))
                    file.write(' ' + ' '.join(str(0.0) for _ in range(3)) + '\n')
            
        # Event
        for fileidx in range(self.netfiles):
            fname = f"{self.netwkdir}/{self.filebase}.event.{fileidx}"
            with open(fname,"w") as file:
                for n in range(node_prefix[part_prefix[fileidx]], node_prefix[part_prefix[fileidx+1]]):
                    # print(' 0')
                    file.write(' 0')

        # Dist
        fname = f"{self.netwkdir}/{self.filebase}.dist"
        with open(fname,"w") as file:
            for partidx in range(self.netparts + 1):
                # print(' '.join(str(num) for num in [node_prefix[partidx], edge_prefix[partidx],
                #                                     state_prefix[partidx], stick_prefix[partidx], 0]))
                file.write(' '.join(str(num) for num in [node_prefix[partidx], edge_prefix[partidx],
                                                         state_prefix[partidx], stick_prefix[partidx], 0]) + '\n')

        # Metis
        fname = f"{self.netwkdir}/{self.filebase}.metis"
        with open(fname,"w") as file:
            for fileidx in range(self.netfiles + 1):
                # print(' '.join(str(num) for num in [node_prefix[part_prefix[fileidx]],
                #                                     edge_prefix[part_prefix[fileidx]]))
                file.write(' '.join(str(num) for num in [node_prefix[part_prefix[fileidx]],
                                                         edge_prefix[part_prefix[fileidx]]]) + '\n')
        
    # Write the topology out to input files (for building)
    def write_file(self):
        # Vertex models
        for name, groups in self.node_set.items():
            filename = dict()
            file = dict()
            for group, g in groups.items():
                if g == 0:
                    group_name = name
                else:
                    group_name = f"{name}_{g}"
                # Filenames for each state
                for key, item in self.model_registry[name]['state'].items():
                    if item['dsl'] is not None:
                        filename[key] = f"{self.netwkdir}/files/{group_name}_{key}.csv"
                if not filename:
                    continue
                # Open files per state
                with ExitStack() as stack:
                    for key, fname in filename.items():
                        file[key] = stack.enter_context(open(fname, 'w'))
                    # Loop over node data (somewhat inefficient)
                    for data in self.node_data:
                        if data['group_name'] == group_name:
                            for key in file.keys():
                                file[key].write(f"{data[key]}\n")

        # Edge models

        for (name, source, target), groups in self.edge_set.items():
            full_name = f"{name}_{source}_{target}"
            model_params = dict()
            model_states = dict()
            for group, g in groups.items(): # currently only supports one group
                if g == 0:
                    group_name = full_name
                else:
                    group_name = f"{full_name}__{g}"

        
        for (edge_name, source_name, target_name), groups in self.edge_set.items():
            full_name = f"{edge_name}_{source_name}_{target_name}"
            filename = dict()
            file = dict()
            for group, g in groups.items():
                if g == 0:
                    group_name = full_name
                else:
                    group_name = f"{full_name}__{g}"
                # Filenames for each state
                for key, item in self.model_registry[edge_name]['state'].items():
                    if item['dsl'] is not None:
                        filename[key] = f"{self.netwkdir}/files/{group_name}_{key}.csv"
                if not filename:
                    continue
                # Open files per state
                with ExitStack() as stack:
                    for key, fname in filename.items():
                        file[key] = stack.enter_context(open(fname, 'w'))
                    # Loop over edge data (even more inefficient)
                    for target, value in enumerate(self.edge_data):
                        if self.node_data[target]['group_name'] == target_name:
                            for source, data in value.items():
                                if (data is not None and
                                    data['group_name'] == group_name and
                                    self.node_data[source]['group_name'] == source_name):
                                    for key in file.keys():
                                        file[key].write(f"{self.local_index[source]}:{data[key]},")
                            for key in file.keys():
                                file[key].write('\n')

    # Collect any output from the simulation
    def read_spikes(self):
        # Load main configuration file
        fname = f"{self.netwkdir}/{self.filebase}.yml"
        with open(fname,"r") as file:
            conf_yaml = yaml.safe_load(file)

        # Calculate some information from the graph file
        fname = f"{self.netwkdir}/{self.filebase}.graph"
        with open(fname,"r") as file:
            graph_yaml = yaml.safe_load(file)
            # Vertex population sizes
            self.vertex_modname = []
            self.vertex_order = []
            # Section for streams
            if 'stream' in graph_yaml:
                for vertex in graph_yaml['stream']:
                    self.vertex_modname.append(vertex['modname'])
                    self.vertex_order.append(1)
            # Section for vertices
            for vertex in graph_yaml['vertex']:
                self.vertex_modname.append(vertex['modname'])
                self.vertex_order.append(vertex['order'])
        self.vertex_prefix = [0] + [sum(self.vertex_order[:i+1]) for i in range(len(self.vertex_order))]

        # Because the neuron models are potentially distributed over multiple
        # partitions, we need to find the reindexing mapping for cleaner plotting
        self.vertex_remap = np.zeros(self.vertex_prefix[-1]).astype(int)
        for fileidx in range(conf_yaml['netfiles']):
            fname = f"{self.netwkdir}/{self.filebase}.index.{fileidx}"
            with open(fname, 'r') as findex:
                for line in findex:
                    global_index, group_index, local_index = line.split()
                    self.vertex_remap[int(global_index)] = self.vertex_prefix[int(group_index)] + int(local_index)
        
        # Reading in data from event logs, which are stored 
        # by recording interval in simulation iterations
        record_interval = int(conf_yaml['trecord']/conf_yaml['tstep'])
        record_max = int(conf_yaml['tmax']/conf_yaml['tstep'])
        self.record_points = list(range(record_interval, record_max, record_interval))+[record_max]

        # Read the record files and collect spikes into an event list
        self.spike_list = [[] for _ in range(self.vertex_prefix[-1])]
        for record in self.record_points:
            for fileidx in range(conf_yaml['netfiles']):
                fname = f"{self.netwkdir}/{self.recordir}/{self.filebase}.evtlog.{record}.{fileidx}"
                with open(fname, 'r') as file:
                    for line in file:
                        # the event format is [event type, timestamp, vertex index, optional payload]
                        event = line.split()
                        event_type = int (event[0])
                        # spikes are event type "0"
                        if event_type == 0:
                            timestamp = float(int(event[1], 16)) / self.ticks_per_ms
                            # reindex the events and add to list
                            global_index = int(event[2])
                            index = int(self.vertex_remap[global_index])
                            self.spike_list[index].append(timestamp)

        return self.spike_list

    # Return spikes as event list
    def get_spikes(self):
        if self.spike_list is None:
            return self.read_spikes()
        else:
            return self.spike_list

    # Plotting as event plot
    def plot_spikes(self, figsize=(8,6), linelengths=0.8, linewidths=1.0,
                    color_dict={'LIF': 'C0', 'IN': 'C1'}):
        if self.spike_list is None:
            self.read_spikes()
            
        # Plot the event list information
        plt.figure(figsize=figsize)

        # We can also color the rows according to population
        if color_dict is None:
            color_dict = {key: f"C{i%10}" for i, key in enumerate(self.vertex_modname)}
        event_color = []
        for index, modname in enumerate(self.vertex_modname):
            if modname not in color_dict:
                color_dict[modname] = f"C{len(color_dict)%10}"
            event_color.extend([color_dict[modname]] * self.vertex_order[index])
        # colored lines (for legend)
        for key in color_dict.keys():
            plt.plot(0,0,'-',color=color_dict[key],linewidth=2.0)
        
        # The spike raster is plotted using eventplot
        plt.eventplot(self.spike_list, colors=event_color, lineoffsets=1,
                      linelengths=linelengths, linewidths=linewidths)
        
        plt.title('Spike Raster')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron (index)')
        plt.tight_layout()
        plt.legend(color_dict.keys())