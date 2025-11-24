import networkx as nx
import uuid


class FitDiag:
    "A directed acyclic graph (DAG) representing fitting instructions."

    def __init__(self, runner):
        self.init_from_runner(runner)

    def init_from_runner(self, runner):
        """
        Initialize the fitting diagram from a runner object.

        Parameters
        ----------
        runner : object
            An object that contains fitting instructions.
            It should have an attribute 'allowed_actions' which is a dict
            mapping action names to the corresponding methods
        """
        self.allowed_actions = runner.allowed_actions

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
                    {"id": "1", "value": "scale"},
                    {"id": "2", "value": "alpha"},
                    {"id": "3", "value": "all"},
                ],
                "edges": [("1", "2"), ("2", "3")],
            }
        """
        self.graph = nx.DiGraph()
        nodes_values = []
        for node in data["nodes"]:
            self.graph.add_node(node["id"], value=node["value"])
            nodes_values.append(node["value"])
        for edge in data["edges"]:
            self.graph.add_edge(edge[0], edge[1])
        self.__check_node_values(nodes_values)

    def from_str(self, diag_str):
        """
        Parse a linear DIAG from a string representation.

        Parameters
        ----------
        diag_str : str
            String representation of the fitting diagram.
            E.g.: "scale->alpha->a->qdamp->all"
        """
        self.graph = nx.DiGraph()
        nodes_values = diag_str.split(",")
        nodes_values = [node.strip() for node in nodes_values]
        parent_node_id = str(uuid.uuid4())
        self.graph.add_node(parent_node_id, value=nodes_values[0])
        for i in range(1, len(nodes_values)):
            child_node_id = str(uuid.uuid4())
            self.graph.add_node(child_node_id, value=nodes_values[i])
            self.graph.add_edge(parent_node_id, child_node_id)
            parent_node_id = child_node_id
        self.__check_node_values(nodes_values)

    def to_json(self):
        """
        Serialize the fitting diagram to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary representation of the fitting diagram.
        """
        data = {
            "nodes": [
                {"id": node, "value": self.graph.nodes[node]["value"]}
                for node in self.graph.nodes
            ],
            "edges": [[u, v] for u, v in self.graph.edges],
        }
        return data

    def __check_node_values(self, nodes_values):
        nodes_values_set = set(nodes_values)
        if set(self.allowed_actions.keys()) != nodes_values_set:
            missing = set(self.allowed_actions.keys()) - nodes_values_set
            extra = nodes_values_set - set(self.allowed_actions.keys())
            error_msg = "Fitting diagram does not match allowed actions."
            if missing:
                error_msg += f" Missing actions: {missing}."
            if extra:
                error_msg += f" Extra actions: {extra}."
            raise ValueError(error_msg)
