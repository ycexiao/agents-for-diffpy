import uuid
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
from collections import defaultdict, OrderedDict
import queue


class FitRunner:
    def __init__(self):
        # window_id: {x_data:[1,2,3], y_data:[1,2,3], update_mode: 'append'|'replace',**kwargs}
        self.data_for_plot = OrderedDict({})
        self.collect_data_realtime_event = {}

    def watch(
        self,
        dag,
        start,
        end,
        pname,
        update_mode,
        source="paylaod",
        title=None,
        style="sparse",
    ):
        if not title:
            title = pname
        window_id = str(uuid.uuid4())
        self.data_for_plot[window_id] = {
            "ydata": queue.Queue(),
            "title": title,
            "update_mode": update_mode,
            "style": style,
        }
        try:
            uuid.UUID(start)
        except ValueError:
            start = dag.name_to_id[start]
        try:
            uuid.UUID(end)
        except ValueError:
            end = dag.name_to_id[end]
        node_ids = nx.shortest_path(dag, start, end)
        for node_id in node_ids:
            this_event = {
                "window_id": window_id,
                "pname": pname,
                "source": source,
            }
            if node_id not in self.collect_data_realtime_event:
                self.collect_data_realtime_event[node_id] = [this_event]
            else:
                self.collect_data_realtime_event[node_id].append(this_event)

    def _collect_data_realtime(self, dag, node_id):
        assert dag.is_marked(node_id, "completed")  # sanity check
        events = self.collect_data_realtime_event.get(node_id, None)
        if not events:
            return
        for this_event in events:
            # Get ydata
            pname = this_event["pname"]
            window_id = this_event["window_id"]
            if this_event["source"] == "payload":
                ydata = dag.nodes[node_id]["payload"].get(pname, None)
            elif this_event["source"] == "adapter":
                ydata = dag.nodes[node_id]["buffer"]["adapter"].snapshots.get(
                    pname, None
                )
            else:
                raise TypeError(
                    f"Not supported data source {this_event['source']}"
                )
            if ydata is None:
                raise KeyError(f"{pname} not found in {this_event['source']}")
            # Store ydata
            self.data_for_plot[window_id]["ydata"].put(ydata)
        time.sleep(0.05)  # 20 Hz

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
        dag.mark(node_id, "completed")
        self._collect_data_realtime(dag, node_id)

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
        return succ_ids

    def run_workflow(
        self,
        dag: FitDAG,
        Adapter: type,
        inputs: dict,
        payload: dict,
    ):
        "Run workflow containing only one set of inputs."
        assert len(dag.root_nodes) == 1
        root_node_id = dag.root_nodes[0]
        root_node = dag.nodes[root_node_id]
        adapter = Adapter()
        adapter.load_inputs(inputs)
        root_node["buffer"]["adapter"] = adapter
        root_node["buffer"]["payload"] = payload
        dag.mark(root_node_id, "hasPayload")
        dag.mark(root_node_id, "hasAdapter")
        start_time = time.time()
        total_nodes_number = len(dag.nodes)
        finished_nodes_number = 0
        max_iter = len(dag.nodes) + 1
        iter_count = 0
        ready_node_ids = dag.root_nodes
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
                f"\tFinished {finished_nodes_number-len(dag.root_nodes)+1} / {total_nodes_number-len(dag.root_nodes)} nodes."  # noqa: E501
                f"\nNext nodes: {[dag.nodes[id]['name'] for id in ready_node_ids]}"  # noqa: E501
            )
        end_time = time.time()
        print(f"Workflow finished. Totally caused {end_time-start_time}s")
        return dag
