from FitDAG import FitDAG
from FitDAG import FitDAG
import warnings
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
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
    def __init__(self, lock):
        self.thread = None
        self.lock = lock
        self.ready = threading.Event()

    def _run_node(
        self,
        dag,
        node_id: str,
    ):
        # FIXME: implement the error later
        assert dag.is_marked_as(node_id, "initialized")
        node = dag.nodes[node_id]
        adapter = node["buffer"]["adapter"]
        adapter.apply_payload(node["buffer"]["payload"])
        adapter.action_func_factory(node["action"])()
        with self.lock:
            # Only place lock in write operations since the analyzer thread
            # only have read operations.
            node["payload"] = adapter.get_payload()

    def _update_successors(self, dag, node_id, Adapter):
        succ_ids = list(dag.successors(node_id))
        this_node_input_source_id = dag.get_input_source_node_id(node_id)
        count = 0
        for succ_id in succ_ids:
            payload_source_node_id = dag.get_payload_source_node_id(succ_id)
            if payload_source_node_id == node_id:
                # pass the payload
                with self.lock:
                    dag[node_id][succ_id]["pass_payload_func"](
                        dag.nodes[node_id],
                        dag.nodes[succ_id],
                    )
                dag.mark(succ_id, "hasPayload")
            input_source_node_id = dag.get_input_source_node_id(succ_id)
            if input_source_node_id == this_node_input_source_id:
                if count == 0:
                    with self.lock:
                        dag.nodes[succ_id]["buffer"]["adapter"] = dag.nodes[
                            node_id
                        ]["buffer"].pop("adapter")
                else:
                    adapter = Adapter()
                    adapter.load_inputs(
                        dag.nodes[input_source_node_id]["inputs"]
                    )
                    with self.lock:
                        dag.nodes[succ_id]["buffer"]["adapter"] = adapter
                count += 1
                dag.mark(succ_id, "hasAdapter")
        with self.lock:
            dag.nodes[node_id].pop("buffer")
        dag.mark(node_id, "completed")
        return succ_ids

    def run_workflow(
        self,
        dag: FitDAG,
        Adapter: type,
        stop_before: str = None,
    ):
        self.thread = threading.Thread(
            target=self._run_workflow,
            daemon=True,
            args=(
                dag,
                Adapter,
                stop_before,
            ),
        )
        self.thread.start()

    def load_workflow(
        self, dag, Adapter, inputs, payloads, input_nodes_names=None
    ):
        # load inputs to root nodes
        with self.lock:
            dag.load_inputs(inputs, input_nodes_names)
        # determine the order of payloads
        if input_nodes_names is not None:
            root_node_names = [
                dag.nodes[node_id]["name"] for node_id in dag.root_nodes
            ]
            assert set(input_nodes_names) == set(root_node_names)
            index = [
                root_node_names.index(input_name[i])
                for input_name in input_nodes_names
            ]
        else:
            index = list(range(len(dag.root_nodes)))
        # load all payloads to root nodes
        for i, node_id in enumerate(dag.root_nodes):
            payload = payloads[index[i]]
            adapter = Adapter()
            adapter.load_inputs(dag.nodes[node_id]["inputs"])
            adapter.apply_payload(payload)
            with self.lock:
                root_node = dag.nodes[node_id]
                # init root nodes
                root_node["buffer"]["adapter"] = adapter
                root_node["buffer"]["payload"] = adapter.get_payload()
            dag.mark(node_id, "hasPayload")
            dag.mark(node_id, "hasAdapter")
        self.ready.set()
        return dag

    def _run_workflow(
        self,
        dag: FitDAG,
        Adapter: type,
        stop_before: str = None,
    ):
        self.ready.wait()
        start_time = time.time()
        # init the iteration parameters
        total_nodes_number = len(dag.nodes)
        finished_nodes_number = 0
        max_iter = len(dag.nodes) + 1
        iter_count = 0
        ready_node_ids = copy.deepcopy(dag.root_nodes)
        while ready_node_ids:
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
            iter_count += 1
            all_succ_ids = []
            for node_id in ready_node_ids:
                self._run_node(
                    dag,
                    node_id,
                )
                succ_ids = self._update_successors(dag, node_id, Adapter)
                all_succ_ids.extend(succ_ids)
                finished_nodes_number += 1
            succ_ids = list(set(all_succ_ids))
            ready_node_ids = [
                id for id in succ_ids if dag.is_marked_as(id, "initialized")
            ]
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
        else:
            end_time = time.time()
            print(f"Workflow finished. Totally caused {end_time-start_time}s")
        return dag
