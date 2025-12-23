import networkx as nx
import uuid
import copy
from collections import OrderedDict
import pickle


def default_propagate_func(parent_node, child_node):
    child_node["payload"] = copy.deepcopy(parent_node["payload"])


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

    If multiple root nodes exist, use the one has the furthest distance to the
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
            "type": "normal",  # could be "start", "normal", "end"
            "action": [],
            "name": "",
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
        template_node = self.default_node
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
            self.all_names.append(name)
        else:
            name = name
            self.all_names.append(name)
        node_dict["name"] = name
        return node_dict

    def furnish_edge_dict(self, edge_dict):
        template_edge = self.default_edge
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
        self.clear()
        all_node_values = []
        for node_content in data["nodes"]:
            node_content["action"] = [
                v.strip() for v in node_content["action"].split(",")
            ]
            node_content = self.furnish_node_dict(node_content)
            node_id = node_content.get("id", str(uuid.uuid4()))
            self.add_node(node_id, **node_content)
            all_node_values.extend(node_content["action"])
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
        self.clear()
        actions = [
            [v.strip() for v in value.split(",")]
            for value in dag_str.split("->")
        ]
        node_content = {"action": actions[0]}
        node_content = self.furnish_node_dict(node_content)
        parent_node_id = str(uuid.uuid4())
        self.add_node(parent_node_id, **node_content)
        for i in range(1, len(actions)):
            node_content = {"action": actions[i]}
            node_content = self.furnish_node_dict(node_content)
            child_node_id = str(uuid.uuid4())
            self.add_node(child_node_id, **node_content)
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
        self.clear()
        with open(filename, "rb") as f:
            graph = pickle.load(f)
        for node_id, node_content in graph.nodes(data=True):
            self.all_names.append(node_content["name"])
            self.add_node(node_id, **node_content)
        for u, v, edge_content in graph.edges(data=True):
            self.add_edge(u, v, **edge_content)
        self._prepare_nodes_structure()

    def get_node_by_name(self, name):
        nodes = []
        for node_id, node_content in self.nodes(data=True):
            if node_content["name"] == name:
                nodes.append((node_id, node_content))
        return nodes

    def get_edge_by_name(self, u_name, v_name):
        u_node_id, _ = self.get_node_by_name(u_name)[0]
        v_node_id, _ = self.get_node_by_name(v_name)[0]
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
        # Add start nodes if not exist
        for root_node_id in self.root_nodes:
            if self.nodes[root_node_id]["type"] != "start":
                start_node = {
                    "action": [],
                    "type": "start",
                    "name": "start",
                    "level": 0,
                }
                start_node_id = str(uuid.uuid4())
                start_node = self.furnish_node_dict(start_node)
                self.add_node(start_node_id, **start_node)
                edge = self.furnish_edge_dict({})
                self.add_edge(start_node_id, root_node_id, **edge)

        for root_node_id in self.root_nodes:
            levels = nx.single_source_shortest_path_length(self, root_node_id)
            for node_id, level in levels.items():
                if self.map_to_root_node.get(node_id, None) is not None:
                    # FIXME: refine the error later
                    assert self.node_levels[node_id] != level
                    if self.node_levels[node_id] > level:
                        self.map_to_root_node[node_id] = root_node_id
                        self.node_levels[node_id] = level
                    else:
                        continue
                else:
                    self.map_to_root_node[node_id] = root_node_id
                    self.node_levels[node_id] = level

        for node_id, level in self.node_levels.items():
            self.nodes[node_id]["level"] = self.node_levels[node_id]

    def get_input_source(self, node_id):
        return self.map_to_root_node.get(node_id, None)

    def get_payload_source(self, node_id):
        if node_id in self.root_nodes:
            return node_id
        parent_ids = list(self.predecessors(node_id))
        levels = [
            self.node_levels.get(parent_id, float("inf"))
            for parent_id in parent_ids
        ]
        return parent_ids[levels.index(max(levels))]

    def clean_copy(
        self,
        with_payload=False,
        with_besides_str=False,
        with_same_id=True,
        instance_type="networkx",
    ):
        if instance_type == "networkx":
            graph = (
                nx.DiGraph()
            )  # use nx.DiGraph so copies can be serialized with minimal compatible issues
        elif instance_type == "FitDAG":
            graph = FitDAG()
        else:
            raise ValueError(
                f"Unsupported instance_type: {instance_type}. "
                "Supported types are 'networkx' and 'FitDAG'."
            )
        id_maps = {}
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
            if with_same_id:
                id_maps[node_id] = node_id
            else:
                new_node_id = str(uuid.uuid4())
                id_maps[node_id] = new_node_id
                node_id = new_node_id
            graph.add_node(node_id, **node_content)

        for u, v, edge_content in self.edges(data=True):
            edge_content = {
                k: v for k, v in edge_content.items() if filter_func(k, v)
            }
            graph.add_edge(id_maps[u], id_maps[v], **edge_content)
        return graph

    def to_json(self):
        def clean_dict(d):
            out_dict = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    out_dict[key] = clean_dict(value)
                elif (
                    isinstance(value, str)
                    or isinstance(value, int)
                    or isinstance(value, float)
                ):
                    out_dict[key] = value
                else:
                    "Unserializable value found, skip it."
            return out_dict

        out_dict = {}
        for node_id, node_content in self.nodes(data=True):
            out_dict["id"] = node_id
            out_dict.update(clean_dict(node_content))
        return clean_dict(out_dict)

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

    def clear(self):
        super().clear()
        self.all_names = []

    def merge_dag(
        self,
        child_dag,
        parent_node_id,
        child_node_id,
    ):
        for node_id, node_content in child_dag.nodes(data=True):
            if node_id not in self.nodes():
                node_content = self.furnish_node_dict(node_content)
                self.add_node(node_id, **node_content)
        for u, v, edge_content in child_dag.edges(data=True):
            if not self.has_edge(u, v):
                edge_content = self.furnish_edge_dict(edge_content)
                self.add_edge(u, v, **edge_content)
        edge = self.furnish_edge_dict({})
        self.add_edge(parent_node_id, child_node_id, **edge)
        self._prepare_nodes_structure()
