from BaseAdapter import BaseAdapter
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from scipy.optimize import least_squares
from collections import OrderedDict
import uuid
import copy
from typing import Literal
from networkx import DiGraph
from FitDAG import FitDAG
import numpy
import warnings

# copy.deepcopy(adapter)
# ERROR   RuntimeError: Pickling of
# "diffpy.srreal.parallel.ParallelPairQuantity" instances is not enabled
# (http://www.boost.org/libs/python/doc/v2/pickle.html)


class FitRunner:
    def __init__(self):
        pass

    def run_workflow(
        self,
        dag: FitDAG,
        Adapter: type,
        inputs: list,
        payloads: list,
        names=None,
    ):
        adapters_dict = {}
        dag.load_inputs(inputs, names)
        for i, node_id in enumerate(dag.root_nodes):
            adapter = Adapter()
            adapter.load_inputs(dag.nodes[node_id]["inputs"])
            # FIXME: name could be duplicated as it was not chekced!
            # fix it latter
            if names:
                index = names.index(dag.nodes[node_id]["name"])
                adapter.apply_payload(payloads[index])
            else:
                adapter.apply_payload(payloads[i])
            adapters_dict[node_id] = adapter

        frontier_node_ids = dag.root_nodes

        while frontier_node_ids:
            upcoming_frontier_node_ids = {}
            for node_id in frontier_node_ids:
                node = dag.nodes[node_id]
                adapter = adapters_dict[node_id]
                adapter.action_func_factory(node["action"])()
                node["payload"] = adapter.get_payload()
                succ_ids = list(dag.successors(node_id))
                upcoming_frontier_node_ids.update(
                    {succ_id: node_id for succ_id in succ_ids}
                )
            for succ_id, parent_id in upcoming_frontier_node_ids.items():
                succ_node = dag.nodes[succ_id]
                input_source_node_id = dag.get_input_source(succ_id)
                if (
                    input_source_node_id == dag.get_input_source(parent_id)
                    and parent_id in adapters_dict
                ):
                    adapters_dict[succ_id] = adapters_dict.pop(parent_id)
                else:
                    adapter = Adapter()
                    adapter.load_inputs(
                        dag.nodes[input_source_node_id]["inputs"]
                    )
                    adapters_dict[succ_id] = adapter
                payload_source_node_id = dag.get_payload_source(succ_id)
                # The current policy use the direct parent node at the lowest
                # level as the payload source, so it must have been processed
                # before.
                # The assertion is a sanity check.
                assert dag.nodes[payload_source_node_id]["payload"] is not None
                payload = dag[payload_source_node_id][succ_id][
                    "propagate_func"
                ](
                    dag.nodes[payload_source_node_id],
                    succ_node,
                )
                adapters_dict[succ_id].apply_payload(payload)
            frontier_node_ids = list(set(upcoming_frontier_node_ids.keys()))

        return dag, adapters_dict
