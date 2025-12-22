import networkx as nx
import uuid
import copy
from collections import OrderedDict
import pickle


def default_propagate_func(parent_node, child_node):
    return parent_node["payload"]


class FitDAG(nx.DiGraph):  # or FitForest?
    """
    A directed acyclic graph (DAG) representing fitting instructions.

    Each node defines an action executed under the context of the input from
    root node and the payload from the parent node. Each action updates
    the payload in the current node.

    # FIXME: This is the current policy. We could allow more flexible policies
    # e.g. combine the payloads from multiple parent nodes or root nodes.
    # modify get_input_source and get_payload_source methods accordingly.

    If multiple parent nodes exist, use the one has the nearest distance to
    the root node.

    If multiple root nodes exist, use the one has the nearest distance to the
    current node.

    payload is any data structure that is updated along the DAG.


    Node Attributes
    ----------------
    description: str
        string description of the current node
    inputs: dict
    action: list
        list of operations being executed in the current node

    Edge Attributes
    ---------------
    description: str
        string description of the current edge
    patch: callable
        function to modify its sucessors' context
    """

    def __init__(self):
        super().__init__()
        self.all_names = []
        self.default_node = {
            "description": "",
            "inputs": {},
            "payload": {},
            "level": 100,
            "action": [],
            "name": "",
            "id": None,
        }

        # FIXME: Edges defines the relations between nodes, e.g. how the child
        # node's context is modified by the parent node's outcome.
        # For now, we only define a placeholder function.
        self.default_edge = {
            "description": "",
            "u": None,
            "v": None,
            "propagate_func": default_propagate_func,
        }

    def furnish_node_dict(self, node_dict):
        template_node = copy.deepcopy(self.default_node)
        node_dict = {
            key: (
                node_dict[key]
                if key in node_dict.keys()
                else template_node[key]
            )
            for key in template_node.keys()
        }
        # FIXME: refine the error later
        assert "action" in node_dict
        name = (
            node_dict["name"]
            if node_dict["name"] != ""
            else ", ".join(node_dict["action"])
        )
        if name in self.all_names:
            count = 1
            new_name = f"{name}_{count}"
            while new_name in self.all_names:
                count += 1
                new_name = f"{name}_{count}"
            name = new_name
        else:
            name = name
            self.all_names.append(name)
        node_dict["name"] = name
        node_dict["id"] = node_dict.get("id", str(uuid.uuid4()))
        return node_dict

    def furnish_edge_dict(self, edge_dict):
        template_edge = copy.deepcopy(self.default_edge)
        edge_dict = {
            key: (
                edge_dict[key]
                if key in edge_dict.keys()
                else template_edge[key]
            )
            for key in template_edge.keys()
        }
        return edge_dict

    def from_dict(self, data):
        """
        Initialize the fitting diagram from a dictionary representation.

        Parameters
        ----------
        data : dict
            A dictionary representation of the fitting diagram.
            e.g.:
            {
                "nodes": [
                    {"id": "1", "name": "node1", "action": "scale"},
                    {"id": "2", "name": "node2", "action": "alpha"},
                    {"id": "3", "name": "node3", "action": "all"},
                ],
                "edges": [
                {"u": "1", "v": "2"},
                {"u": "2", "v": "3"}
                ],
            }
        """
        all_node_values = []
        for node in data["nodes"]:
            node["action"] = [v.strip() for v in node["action"].split(",")]
            node = self.furnish_node_dict(node)
            self.add_node(node["id"], **node)
            all_node_values.extend(node["action"])
        for edge in data["edges"]:
            edge = self.furnish_edge_dict(edge)
            self.add_edge(edge["u"], edge["v"], **edge)
        self._prepare_nodes_structure()

    def from_str(self, dag_str):
        """
        Parse a linear DAG from a string representation.

        Parameters
        ----------
        diag_str : str
            String representation of the fitting diagram.
            E.g.: "scale->alpha->a->qdamp->all"
        """
        actions = [
            [v.strip() for v in value.split(",")]
            for value in dag_str.split("->")
        ]
        node = {"id": str(uuid.uuid4()), "action": actions[0]}
        node = self.furnish_node_dict(node)
        self.add_node(node["id"], **node)
        parent_node_id = node["id"]
        for i in range(1, len(actions)):
            node = {"id": str(uuid.uuid4()), "action": actions[i]}
            node = self.furnish_node_dict(node)
            self.add_node(node["id"], **node)
            child_node_id = node["id"]
            edge = self.furnish_edge_dict({})
            self.add_edge(parent_node_id, child_node_id, **edge)
            parent_node_id = child_node_id
        self._prepare_nodes_structure()

    def from_workflow(self, filename):
        """
        Load the fitting diagram from a serialized DAG file.

        Parameters
        ----------
        filename : str
            Path to the serialized DAG file.
        """
        with open(filename, "rb") as f:
            graph = pickle.load(f)
        for node_id, node_content in graph.nodes(data=True):
            self.add_node(node_id, **node_content)
        for u, v, edge_content in graph.edges(data=True):
            self.add_edge(u, v, **edge_content)
        self._prepare_nodes_structure()

    def get_node_by_name(self, name):
        nodes = OrderedDict()
        for node_id, node_content in self.nodes(data=True):
            if node_content["name"].split("_")[0] == name.split("_")[0]:
                nodes[node_id] = node_content
        return nodes

    def get_edge_by_name(self, u_name, v_name):
        u_node_id = list(self.get_node_by_name(u_name).keys())[0]
        v_node_id = list(self.get_node_by_name(v_name).keys())[0]
        return self[u_node_id][v_node_id]

    @property
    def root_nodes(self):
        return [
            node_id for node_id in self.nodes() if self.in_degree(node_id) == 0
        ]

    @property
    def leaf_nodes(self):
        return [
            node_id
            for node_id in self.nodes()
            if self.out_degree(node_id) == 0
        ]

    def load_inputs(self, inputs, names=None):
        root_names = [
            self.nodes[node_id]["name"] for node_id in self.root_nodes
        ]
        if names:
            for name in names:
                # FIXME: refine the error later
                assert name in root_names
                index = root_names.index(name)
                self.nodes[self.root_nodes[index]]["inputs"] = inputs[index]
        else:
            for i, root_node_id in enumerate(self.root_nodes):
                self.nodes[root_node_id]["inputs"] = inputs[i]

    def _prepare_nodes_structure(self):
        self.map_to_root_node = {}
        self.node_levels = {}
        for root_node_id in self.root_nodes:
            levels = nx.single_source_shortest_path_length(self, root_node_id)
            for node_id, level in levels.items():
                if self.map_to_root_node.get(node_id, None) is not None:
                    # FIXME: refine the error later
                    assert self.map_to_root_node[node_id] != level
                    if self.map_to_root_node[node_id] > level:
                        self.map_to_root_node[node_id] = root_node_id
                        self.node_levels[node_id] = level
                    else:
                        continue
                else:
                    self.map_to_root_node[node_id] = root_node_id
                    self.node_levels[node_id] = level

    def get_input_source(self, node_id):
        return self.map_to_root_node.get(node_id, None)

    def get_payload_source(self, node_id):
        parent_ids = list(self.predecessors(node_id))
        levels = [
            self.node_levels.get(parent_id, float("inf"))
            for parent_id in parent_ids
        ]
        return parent_ids[levels.index(min(levels))]

    def clean_copy(self, with_payload=False, with_besides_str=False):
        graph = (
            nx.DiGraph()
        )  # use nx.DiGraph so copies can be serialized with minimal compatible issues
        if with_payload and with_besides_str:
            filter_func = lambda k, v: True
        elif (not with_payload) and with_besides_str:
            filter_func = lambda k, v: k != "payload"
        elif with_payload and (not with_besides_str):
            filter_func = lambda k, v: isinstance(v, str)
        else:
            filter_func = lambda k, v: (k != "payload") and isinstance(v, str)
        for node_id, node_content in self.nodes(data=True):
            node_content = {
                k: v for k, v in node_content.items() if filter_func(k, v)
            }
            graph.add_node(node_id, **node_content)

        for u, v, edge_content in self.edges(data=True):
            edge_content = {
                k: v for k, v in edge_content.items() if filter_func(k, v)
            }
            graph.add_edge(u, v, **edge_content)
        return graph

    def render(self, filename="graph.html"):
        from pyvis.network import Network

        # Create PyVis network
        net = Network(
            directed=True,
            notebook=False,  # IMPORTANT
            cdn_resources="remote",  # IMPORTANT
        )

        graph = self.clean_copy(with_payload=False, with_besides_str=False)
        node_ids, node_lables = [], []
        for node_id, node_content in graph.nodes(data=True):
            node_ids.append(node_id)
            node_lables.append(node_content["name"])
        net.add_nodes(node_ids, label=node_lables)
        net.add_edges(graph.edges())
        # Save and render
        net.write_html(filename)
