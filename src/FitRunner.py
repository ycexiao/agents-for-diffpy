import uuid
from FitDAG import FitDAG
from FitDAG import FitDAG
import warnings
import threading
import matplotlib.pyplot as plt
import time
import networkx as nx
from collections import OrderedDict
import queue


class FitRunner:
    def __init__(self):
        # Interact with FitPlotter
        # {window_id: {"ydata": queue.Queue(), "title": str, "update_mode": str, "style": str}}
        self.data_for_plot = OrderedDict({})
        # Capture and store data in self.data_for_plot
        # {window_id: {"trigger_func": lambda dag, node_id: bool, "pname": str, "source": str}}
        self.collect_data_event = OrderedDict({})

    def watch(
        self,
        trigger_func,
        pname,
        update_mode,
        source="payload",
        title=None,
        style="sparse",
        window_id=None,
    ):
        """Set the ploting variables. Can be called multiple times."""
        if not title:
            title = pname
        if not window_id:
            window_id = str(uuid.uuid4())
        if window_id not in self.data_for_plot:
            self.data_for_plot[window_id] = {
                "ydata": queue.Queue(),
                "title": title,
                "update_mode": update_mode,
                "style": style,
            }
        if not self.collect_data_event:
            self.collect_data_event = {}
        this_event = {
            "trigger_func": trigger_func,
            "pname": pname,
            "source": source,
        }
        self.collect_data_event[window_id] = this_event
        return window_id

    def _collect_data_realtime(self, dag, node_id):
        assert dag.is_marked(node_id, "completed")  # sanity check
        if not self.collect_data_event:
            return
        for window_id, this_event in self.collect_data_event.items():
            if not this_event["trigger_func"](dag, node_id):
                continue
            pname = this_event["pname"]
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

    def _run_node(
        self,
        dag,
        node_id: str,
    ):
        """Run a single node in the DAG."""
        # A fail-safe check
        assert dag.is_marked(node_id, "initialized")
        node = dag.nodes[node_id]
        adapter = node["buffer"]["adapter"]
        adapter.apply_payload(node["buffer"]["payload"])
        adapter.action_func_factory(node["action"])()
        node["payload"] = adapter.get_payload()
        dag.mark(node_id, "completed")
        self._collect_data_realtime(dag, node_id)

    def _update_successors(self, dag, node_id, Adapter):
        """Update the sucessors for the current node"""
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

    def _run_dag(
        self,
        dag: FitDAG,
        Adapter: type,
        inputs: dict,
        payload: dict,
    ):
        assert len(dag.root_nodes) == 1
        root_node_id = dag.root_nodes[0]
        root_node = dag.nodes[root_node_id]
        adapter = Adapter()
        adapter.load_inputs(inputs)
        root_node["buffer"] = {"adapter": adapter, "payload": payload}
        dag.mark(root_node_id, "hasPayload")
        dag.mark(root_node_id, "hasAdapter")
        start_time = time.time()
        finished_nodes_number = 0
        total_nodes_number = len(dag.nodes)
        max_iter = len(dag.nodes) + 1
        iter_count = 0
        ready_node_ids = dag.root_nodes
        while ready_node_ids:
            print(
                f"\tFinished nodes {finished_nodes_number} / {total_nodes_number}. This iteration: "  # noqa: E501
                f"{[dag.nodes[id]['name'] for id in ready_node_ids]}"  # noqa: E501
            )
            if iter_count > max_iter:
                warnings.warn(
                    "Maximum iterations reached in FitDAG execution. "
                    "Possible cyclic dependency or deadlock."
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
                id for id in succ_ids if dag.is_marked(id, "initialized")
            ]
        end_time = time.time()
        print(f"\tThis dag is finished. Caused {end_time-start_time}s")
        return dag

    def _run_sequential_dags(
        self, dags: list, Adapter: type, inputs: dict, payload: dict
    ):
        for i in range(len(dags)):
            print(f"Processing {i+1}/{len(dags)} dags")  # noqa: E501
            self._run_dag(
                dags[i], Adapter=Adapter, inputs=inputs[i], payload=payload
            )
            last_node_id = list(nx.topological_sort(dags[i]))[-1]
            payload = dags[i].nodes[last_node_id]["payload"]
        return dags

    def get_run_dag_thread(
        self,
        dag: FitDAG,
        Adapter: type,
        inputs: dict,
        payload: dict,
    ):
        kwargs = {
            "dag": dag,
            "Adapter": Adapter,
            "inputs": inputs,
            "payload": payload,
        }
        t = threading.Thread(target=self._run_dag, kwargs=kwargs)
        return t

    def get_run_sequential_thread(
        self, dags: list, Adapter: type, inputs: dict, payload: dict
    ):

        kwargs = {
            "dags": dags,
            "Adapter": Adapter,
            "inputs": inputs,
            "payload": payload,
        }
        t = threading.Thread(target=self._run_sequential_dags, kwargs=kwargs)
        return t
