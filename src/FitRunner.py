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
        stop_before: str = None,
    ):
        dag.load_inputs(inputs, names)
        adapters_dict = {}
        frontier_node_ids = []
        wait_list = []
        for i, root_node_id in enumerate(dag.root_nodes):
            if names:
                index = names.index(dag.nodes[root_node_id]["name"])
                payload = payloads[index]
            else:
                payload = payloads[i]
            adapter = Adapter()
            adapter.load_inputs(dag.nodes[root_node_id]["inputs"])
            adapter.apply_payload(payload)
            root_node = dag.nodes[root_node_id]
            root_node["payload"] = adapter.get_payload()
            adapters_dict[root_node_id] = adapter
            succ_ids = list(dag.successors(root_node_id))
            # Root node should have only one child node
            assert len(succ_ids) == 1
            succ_id = succ_ids[0]
            payload_source_node = dag.nodes[dag.get_payload_source(succ_id)]
            if payload_source_node["payload"] != {}:
                dag[root_node_id][succ_id]["propagate_func"](
                    dag.nodes[root_node_id],
                    dag.nodes[succ_id],
                )
                frontier_node_ids.append(succ_id)
            else:
                wait_list.append(succ_id)
            adapters_dict[succ_id] = adapters_dict.pop(root_node_id)
        frontier_node_ids = list(set(frontier_node_ids))

        while frontier_node_ids:
            if stop_before:
                names = [
                    dag.nodes[node_id]["name"] for node_id in frontier_node_ids
                ]
                if stop_before in names:
                    return dag, adapters_dict
            upcoming_runnable_frontier_node_ids = []
            for node_id in frontier_node_ids:
                node = dag.nodes[node_id]
                if node["payload"] == {}:
                    if node_id in wait_list:
                        continue
                    else:
                        wait_list.append(node_id)
                        continue
                else:
                    if node_id in wait_list:
                        wait_list.remove(node_id)
                    adapter = adapters_dict[node_id]
                    adapter.apply_payload(node["payload"])
                    adapter.action_func_factory(node["action"])()
                    node["payload"] = adapter.get_payload()
                    succ_ids = list(dag.successors(node_id))
                    if len(succ_ids) != 0 and all(
                        [
                            dag.get_input_source(succ_id)
                            != dag.get_input_source(node_id)
                            for succ_id in succ_ids
                        ]
                    ):
                        adapters_dict.pop(node_id)
                    upcoming_runnable_frontier_node_ids.extend(
                        [(node_id, succ_id) for succ_id in succ_ids]
                    )
            if upcoming_runnable_frontier_node_ids == []:
                if wait_list:
                    warnings.warn(
                        "Deadlock detected in FitDAG execution. "
                        "Some nodes are waiting for payloads that will never arrive. "
                        "Check the DAG structure and payload propagation functions."
                    )
                return dag, adapters_dict, wait_list
            for parent_id, succ_id in upcoming_runnable_frontier_node_ids:
                succ_node = dag.nodes[succ_id]
                input_source_node_id = dag.get_input_source(succ_id)
                # pass the adapter
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
                # pass the payload
                assert dag.nodes[payload_source_node_id]["payload"] is not None
                dag[payload_source_node_id][succ_id]["propagate_func"](
                    dag.nodes[payload_source_node_id],
                    succ_node,
                )
            frontier_node_ids = [
                succ_id for _, succ_id in upcoming_runnable_frontier_node_ids
            ]
            frontier_node_ids.extend(wait_list)
            frontier_node_ids = list(set(frontier_node_ids))
