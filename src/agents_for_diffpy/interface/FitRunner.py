import uuid
import warnings
import threading
import time
from collections import OrderedDict, defaultdict
import queue
from agents_for_diffpy.interface import FitDAG


class FitRunner:
    def __init__(self):
        # Interact with FitPlotter
        # {window_id: {
        #   "ydata": queue.Queue(),
        #   "title": str,
        #   "update_mode": str,
        #    "style": str}}
        self.data_for_plot = OrderedDict({})
        # Capture and store data in self.data_for_plot
        # {window_id: {
        #   "trigger_func": lambda dag, node_id: bool,
        #   "pname": str,
        #   "source": str}}
        self.collect_data_event = OrderedDict({})
        # Temporary storage for running information
        self.running_info = {}

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
        """Set the ploting variables.

        FitRunner decomposes "How to moniter the results" into three parts:
        (1) when to collect the data
        (2) what data to collect
        (3) where to store the collected data

        (1) the `trigger_func` will be executed at the end of
        each node. It takes (dag, node_id) as input and returns a boolean
        indicating whether the data should be collected.
        (2) the `pname` and `source` specify what data to collect.
        `source` could be either "payload" or "adapter".
        If "payload", the data will be collected from the node's payload.
        If "adapter", the data will be collected from the adapter's snapshots.
        `pname` is the key of the payload/snapshots from which to collect
        the data.
        (3) FitRunner keeps an internal dictionary `data_for_plot` to store
        the collected data. The key for the dictionary is the `window_id`.
        If not provided, a new UUID will be generated and used as the window
        ID. This way, the production and consumption of the data can be
        decoupled.

        Parameters
        ----------
        trigger_func : callable
            The function that takes (dag, node_id) and returns True
            if the data should be collected.
        pname : str
            The name of the parameter to collect.
        update_mode : str
            The mode of updating the plot, e.g., "append" or "replace".
            If it is "append", the new data will be appended to the existing
            data in the plot.
            If it is "replace", the new data will replace the existing data
            in the plot.
        source : str, optional
            The source of the data, either "payload" or "adapter".
            Default is "payload".
        title : str, optional
            The title of the plot. Default is the parameter name.
        style : str, optional
            The style of the plot. Default is "sparse".
        window_id : str, optional
            The ID of the plot window. If not provided,
            a new UUID will be generated.

        Returns
        -------
        str
            The window ID for the plot.
        """
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
        assert self.is_marked(node_id, "completed")  # sanity check
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
        assert self.is_marked(node_id, "initialized")
        node = dag.nodes[node_id]
        adapter = node["buffer"]["adapter"]
        adapter.apply_payload(node["buffer"]["payload"])
        adapter.action_func_factory(node["action"])()
        node["payload"] = adapter.get_payload()
        self.mark(node_id, "completed")
        self._collect_data_realtime(dag, node_id)

    def _update_successors(self, dag, node_id, Adapter):
        """Update the sucessors for the current node."""
        succ_ids = list(dag.successors(node_id))
        this_node = dag.nodes[node_id]
        adapter = this_node["buffer"].pop("adapter")
        count = 0
        for succ_id in succ_ids:
            succ_node = dag.nodes[succ_id]
            # pass the payload
            succ_node["buffer"] = {"payload": this_node["payload"]}
            self.mark(succ_id, "hasPayload")
            if count == 0:
                succ_node["buffer"]["adapter"] = adapter
            else:
                succ_node["buffer"]["adapter"] = adapter.clone()
            count += 1
            self.mark(succ_id, "hasAdapter")
        return succ_ids

    def _run_dag(
        self,
        dag: FitDAG,
        Adapter: type,
        inputs: dict,
        payload: dict,
    ):
        self.running_info = {}
        assert len(dag.root_nodes) == 1
        root_node_id = dag.root_nodes[0]
        root_node = dag.nodes[root_node_id]
        adapter = Adapter()
        adapter.load_inputs(inputs)
        root_node["buffer"] = {"adapter": adapter, "payload": payload}
        self.mark(root_node_id, "hasPayload")
        self.mark(root_node_id, "hasAdapter")
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
                id for id in succ_ids if self.is_marked(id, "initialized")
            ]
        end_time = time.time()
        print(f"\tThis dag is finished. Caused {end_time-start_time}s")
        return dag

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

    def mark(self, node_id, tag):
        """Mark the running status of a node.

        Used by FitRunner.
        """
        if "node_status" not in self.running_info:
            self.running_info["node_status"] = defaultdict(list)
        allowed_tags = ["hasPayload", "hasAdapter", "completed"]
        # FIXME: implement the error later
        assert tag in allowed_tags
        self.running_info["node_status"][node_id].append(tag)

    def is_marked(self, node_id, status):
        """Check the running status of a node.

        Used by FitRunner.
        """
        if "node_status" not in self.running_info:
            return False
        if status == "initialized":
            return set(["hasPayload", "hasAdapter"]) == set(
                self.running_info["node_status"][node_id]
            )
        elif status == "completed":
            return (
                len(self.running_info["node_status"][node_id]) == 3
                and "completed"
                == self.running_info["node_status"][node_id][-1]
            )
