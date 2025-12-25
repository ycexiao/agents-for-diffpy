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

    def run_node(
        self,
        dag,
        node_id: str,
    ):
        # FIXME: implement the error later
        assert dag.hasInitialized(node_id)
        node = dag.nodes[node_id]
        adapter = node["buffer"]["adapter"]
        adapter.apply_payload(node["buffer"]["payload"])
        node["payload"] = adapter.get_payload()

    def update_successors(self, dag, node_id, Adapter):
        succ_ids = list(dag.successors(node_id))
        this_node_input_source_id = dag.get_input_source_node_id(node_id)
        for succ_id in succ_ids:
            payload_source_node_id = dag.get_payload_source_node_id(succ_id)
            if payload_source_node_id == node_id:
                dag[node_id][succ_id]["pass_payload_func"](
                    dag.nodes[node_id],
                    dag.nodes[succ_id],
                )
            input_source_node_id = dag.get_input_source_node_id(succ_id)
            count = 0
            if input_source_node_id == this_node_input_source_id:
                count += 1
                if count == 0:
                    dag.nodes[succ_id]["buffer"]["adapter"] = dag.nodes[
                        node_id
                    ]["buffer"].pop("adapter")
                else:
                    adapter = Adapter()
                    adapter.load_inputs(
                        dag.nodes[input_source_node_id]["inputs"]
                    )
                    dag.nodes[succ_id]["buffer"]["adapter"] = adapter
        return succ_ids

    def run_workflow(
        self,
        dag: FitDAG,
        Adapter: type,
        inputs: list,
        payloads: list,
        names=None,
        stop_before: str = None,
    ):
        # load inputs to root nodes
        dag.load_inputs(inputs, names)
        # load payload and adapter to root nodes
        for i, root_node_id in enumerate(dag.root_nodes):
            if names:
                index = names.index(dag.nodes[root_node_id]["name"])
                payload = payloads[index]
            else:
                payload = payloads[i]
            # init root nodes
            adapter = Adapter()
            adapter.load_inputs(dag.nodes[root_node_id]["inputs"])
            adapter.apply_payload(payload)
            root_node = dag.nodes[root_node_id]
            # run the root node
            root_node["buffer"]["adapter"] = adapter
            root_node["buffer"]["payload"] = adapter.get_payload()
        # init running variables
        total_nodes_number = len(dag.nodes)
        finished_nodes_number = 0
        max_iter = total_nodes_number + 1
        iter_count = 0
        ready_node_ids = list(dag.root_nodes)
        while ready_node_ids:
            iter_count += 1
            if iter_count > max_iter:
                warnings.warn(
                    "Maximum iterations reached in FitDAG execution. "
                    "Possible cyclic dependency or deadlock."
                )
                return dag
            if stop_before:
                names = [
                    dag.nodes[node_id]["name"] for node_id in ready_node_ids
                ]
                if stop_before in names:
                    warnings.warn(
                        f"Stopping before node {stop_before} as requested."
                    )
                    return dag
            all_succ_ids = []
            for node_id in ready_node_ids:
                # FIXME: implement the error later
                assert dag.hasInitialized(node_id)
                self.run_node(
                    dag,
                    node_id,
                )
                succ_ids = self.update_successors(dag, node_id, Adapter)
                all_succ_ids.extend(succ_ids)
                finished_nodes_number += 1
            succ_ids = list(set(all_succ_ids))
            ready_node_ids = [id for id in succ_ids if dag.hasInitialized(id)]
            print(
                f"In {iter_count} iterations:\n"  # noqa: E501
                f"\tFinished {finished_nodes_number-len(dag.root_nodes)} / {total_nodes_number-len(dag.root_nodes)} nodes."  # noqa: E501
                f"\nNext nodes: {[dag.nodes[id]['name'] for id in ready_node_ids]}"  # noqa: E501
            )
        if finished_nodes_number < total_nodes_number:
            warnings.warn(
                "Not all nodes were finished in FitDAG execution. "
                "Possible cyclic dependency or deadlock."
            )
        return dag
