from FitDAG import FitDAG
from FitDAG import FitDAG
import warnings
import threading
import PDFAdapter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numpy._typing import _128Bit
import psutil, os
import re
import networkx as nx
import copy
from collections import defaultdict

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
        # A fail-safe check
        assert dag.is_marked(node_id, "initialized")
        node = dag.nodes[node_id]
        adapter = node["buffer"]["adapter"]
        adapter.apply_payload(node["buffer"]["payload"])
        adapter.action_func_factory(node["action"])()
        node["payload"] = adapter.get_payload()

    def update_successors(self, dag, node_id, Adapter):
        succ_ids = list(dag.successors(node_id))
        this_node = dag.nodes[node_id]
        adapter = this_node["buffer"].pop("adapter")
        count = 0
        for succ_id in succ_ids:
            succ_node = dag.nodes[succ_id]
            # pass the payload
            succ_node["buffer"] = {"payload": this_node["payload"]}
            dag.mark(succ_id, "hasPayload")
            if count == 0:
                succ_node["buffer"]["adapter"] = adapter
            else:
                succ_node["buffer"]["adapter"] = adapter.clone()
            count += 1
            dag.mark(succ_id, "hasAdapter")
        dag.mark(node_id, "completed")
        return succ_ids

    def run_workflow(
        self,
        dag: FitDAG,
        Adapter: type,
        inputs: dict,
        payload: dict,
    ):
        "Run workflow containing only one set of inputs."
        start_time = time.time()
        # init the iteration parameters
        assert len(dag.root_nodes) == 1
        root_node_id = dag.root_nodes[0]
        root_node = dag.nodes[root_node_id]
        adapter = Adapter()
        adapter.load_inputs(inputs)
        root_node["buffer"]["adapter"] = adapter
        root_node["buffer"]["payload"] = payload
        dag.mark(root_node_id, "hasPayload")
        dag.mark(root_node_id, "hasAdapter")
        total_nodes_number = len(dag.nodes)
        finished_nodes_number = 0
        max_iter = len(dag.nodes) + 1
        iter_count = 0
        ready_node_ids = [root_node_id]
        while ready_node_ids:
            if iter_count > max_iter:
                warnings.warn(
                    "Maximum iterations reached in FitDAG execution. "
                    "Possible cyclic dependency or deadlock."
                )
                return dag
            iter_count += 1
            all_succ_ids = []
            for node_id in ready_node_ids:
                self.run_node(
                    dag,
                    node_id,
                )
                succ_ids = self.update_successors(dag, node_id, Adapter)
                all_succ_ids.extend(succ_ids)
                finished_nodes_number += 1
            succ_ids = list(set(all_succ_ids))
            ready_node_ids = [
                id for id in succ_ids if dag.is_marked(id, "initialized")
            ]
            print(
                f"In {iter_count} iterations:\n"  # noqa: E501
                f"\tFinished {finished_nodes_number-len(dag.root_nodes)} / {total_nodes_number-len(dag.root_nodes)} nodes."  # noqa: E501
                f"\nNext nodes: {[dag.nodes[id]['name'] for id in ready_node_ids]}"  # noqa: E501
            )
        end_time = time.time()
        print(f"Workflow finished. Totally caused {end_time-start_time}s")
        return dag
