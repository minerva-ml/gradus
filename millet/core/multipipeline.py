from functools import reduce
import networkx as nx
from networkx.algorithms.dag import ancestors, lexicographical_topological_sort
from .base import BaseStep, SupervTransformer, UnsupervTransformer, DataLoader
from .utils import MilletException, MilletTypeException, MilletNameException


class MultiPipeline(object):
    def __init__(self):
        super().__init__()
        self._graph = nx.DiGraph()

    def _check_add_step(self, step: BaseStep, node_name: str):
        if self._graph.has_node(node_name):
            raise MilletNameException(f'Step with node_name {node_name} '
                                      'already exists in this multipipeline')
        self._graph.add_node(node_name, step=step, output=None)

    def add_dataloader(self, dataloader: DataLoader, node_name: str):
        if not isinstance(dataloader, DataLoader):
            raise MilletTypeException('dataloader must be a DataLoader'
                                      ' instance')
        self._check_add_step(dataloader, node_name)

    def get_step(self, node_name: str):
        return self._graph.nodes[node_name]['step']

    def add_superv(self, transformer: SupervTransformer, node_name: str,
                   input_mapping: dict, superv_mapping: dict):
        """

        :param transformer:
        :param node_name:
        :param input_mapping: {<new data_dict key>: (<node_name>, <src data_dict key>)}
        :param superv_mapping: {<new data_dict key>: (<node_name>, <src data_dict key>)}
        :param outputs:
        :return:
        """
        if not isinstance(transformer, SupervTransformer):
            raise MilletTypeException('transformer must be a SupervTransformer'
                                      ' instance')
        self._check_add_step(transformer, node_name)
        self._connect_input_mapping(node_name, input_mapping)
        self._connect_superv_mapping(node_name, superv_mapping)

    def add_unsuperv(self, transformer: UnsupervTransformer, node_name: str,
                     input_mapping: dict):
        """

        :param transformer:
        :param node_name:
        :param input_mapping: {<new data_dict key>: (<node_name>, <src data_dict key>)}
        :param outputs:
        :return:
        """
        if not isinstance(transformer, UnsupervTransformer):
            raise MilletTypeException('transformer must be an UnsupervTransformer'
                                      ' instance')
        self._check_add_step(transformer, node_name)
        self._connect_input_mapping(node_name, input_mapping)

    def _connect_input_mapping(self, node_name, input_mapping):

        # TODO: very similar code repeated: refactor?
        for new_data_key, src_name_and_data_key in input_mapping.items():
            src_name, src_key = src_name_and_data_key

            if not self._graph.has_edge(src_name, node_name):
                self._graph.add_edge(src_name, node_name)

            if not ('source_data_keys' in self._graph.edges[src_name, node_name]):
                self._graph.edges[src_name, node_name]['source_data_keys'] = {}

            self._graph.edges[src_name, node_name]['source_data_keys'][new_data_key] = src_key

    def _connect_superv_mapping(self, node_name, superv_mapping):

        for new_data_key, src_name_and_data_key in superv_mapping.items():
            src_name, src_key = src_name_and_data_key

            if not self._graph.has_edge(src_name, node_name):
                self._graph.add_edge(src_name, node_name)

            if not ('source_superv_keys' in self._graph.edges[src_name, node_name]):
                self._graph.edges[src_name, node_name]['source_superv_keys'] = {}

            self._graph.edges[src_name, node_name]['source_superv_keys'][new_data_key] = src_key

    def get_step_output(self, node_name: str):
        return self._graph.nodes[node_name]['output']

    def clear_step_output(self, node_name: str):
        self._graph.nodes[node_name]['output'] = None

    def clear_all_outputs(self):
        for node_name in self._graph.nodes:
            print(f'[MultiPipeline] Clearing output of node_name: {node_name}')
            self.clear_step_output(node_name)

    def run(self,
            input_info: dict,
            output_node_names=(),
            fit_node_names=(),
            ):

        # TODO: Ensure that the requested steps are indeed in the graph

        self.clear_all_outputs()
        up_to_steps_and_fit_steps = set(output_node_names).union(fit_node_names)
        ancestors_to_run = reduce(set.union,
                                  [ancestors(self._graph, node_name)
                                   for node_name in up_to_steps_and_fit_steps])
        _names_steps_to_run = up_to_steps_and_fit_steps.union(ancestors_to_run)

        subgraph_to_run = self._graph.subgraph(_names_steps_to_run)

        # Iterate over steps in the multipipeline
        for node_name in lexicographical_topological_sort(subgraph_to_run):
            print(f'[MultiPipeline] Running node {node_name}')
            node = self._graph.nodes[node_name]
            step = node['step']

            # Only BaseDataLoader descendants should be able to load data
            if isinstance(step, DataLoader):
                print(f'[MultiPipeline]   Calling load_data')
                node['output'] = step.load_data(input_info)

            # For BaseTransformer descendants feed them their inputs and cache
            # outputs
            elif isinstance(step, SupervTransformer):
                print('[MultiPipeline]   Translating inputs')
                input_dict = self._translate_inputs(node_name)
                if node_name in fit_node_names:
                    superv_dict = self._translate_superv(node_name)
                    print(f'[MultiPipeline]   Calling fit_transform with input_dict {input_dict.keys()} and superv_dict {superv_dict.keys()}')
                    node['output'] = step.fit_transform(input_dict, superv_dict)
                else:
                    print(f'[MultiPipeline]   Calling transform with input_dict {input_dict.keys()}')
                    node['output'] = step.transform(input_dict)

            elif isinstance(step, UnsupervTransformer):
                print('[MultiPipeline]   Translating inputs')
                input_dict = self._translate_inputs(node_name)
                if node_name in fit_node_names:
                    print(f'[MultiPipeline]   Calling fit_transform with input_dict {input_dict.keys()}')
                    node['output'] = step.fit_transform(input_dict)
                else:
                    print(f'[MultiPipeline]   Calling transform with input_dict {input_dict.keys()}')
                    node['output'] = step.transform(input_dict)

            else:
                raise MilletTypeException(f'[MultiPipeline] Not an instance of a recognized class: {step}')

        return {node_name: self.get_step_output(node_name)
                for node_name in output_node_names}

    def _translate_inputs(self, node_name):
        new_dict = {}
        for src_node_name, _, edge_data in self._graph.in_edges(node_name, data=True):
            # print(f'[DEBUG] Edge data: {edge_data}')
            if 'source_data_keys' in edge_data:
                for new_data_key, src_data_key in edge_data['source_data_keys'].items():
                    src_output = self.get_step_output(src_node_name)
                    new_dict[new_data_key] = src_output[src_data_key]
                    print(f'[MultiPipeline]     Forwarding data {src_node_name}:{src_data_key} to data {node_name}:{new_data_key}')
        return new_dict

    def _translate_superv(self, node_name):
        new_dict = {}
        for src_node_name, _, edge_data in self._graph.in_edges(node_name, data=True):
            if 'source_superv_keys' in edge_data:
                for new_data_key, src_data_key in edge_data['source_superv_keys'].items():
                    src_output = self.get_step_output(src_node_name)
                    new_dict[new_data_key] = src_output[src_data_key]
                    print(f'[MultiPipeline]     Forwarding data {src_node_name}:{src_data_key} to supervision data {node_name}:{new_data_key}')
        return new_dict

    def print(self):
        print(*nx.generate_multiline_adjlist(self._graph), sep='\n')

    def draw(self):
        pos = nx.circular_layout(self._graph)
        nx.draw_networkx_nodes(self._graph, pos, node_color='g', alpha=0.3, node_size=350)
        nx.draw_networkx_labels(self._graph, pos)
        nx.draw_networkx_edges(self._graph, pos, arrows=True)
