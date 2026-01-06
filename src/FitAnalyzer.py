import matplotlib.pyplot as plt
import threading
import re
import networkx as nx
import warnings
from matplotlib.animation import FuncAnimation
from queue import Empty


class FitAnalyzer:
    def __init__(self, lock):
        self.lock = lock
        self.dag_plot_data = []
        self.dag_plot_artists = []
        self.adapter_plot_data = []
        self.adapter_plot_artists = []

    def watch_dag(self, dag, step_name, pname, mode="append"):
        """Watch 'pname' parameter of all nodes with 'step_name' as prefix."""
        step_names = []
        node_ids = []
        if step_name is not None:
            for node_id, node_content in dag.nodes(data=True):
                assert step_name in dag.unique_names
                if node_content["name"].startswith(step_name):
                    step_names.append(node_content["name"])
                    node_ids.append(node_id)
                    step_names[step_names.index(step_name)] = step_name + "_0"
                    order = [
                        int(re.findall(r"_(\d+)$", name)[0])
                        for name in step_names
                    ]
                    sorted_node_ids = [
                        id for _, id in sorted(zip(order, node_ids))
                    ]
        else:
            for node_id, node_content in dag.nodes(data=True):
                node_ids.append(node_id)
                order = [dag.nodes[id]["level"] for id in node_ids]
                sorted_node_ids = [
                    id for _, id in sorted(zip(order, node_ids))
                ]

        self.dag_plot_data.append(
            {
                "ids": sorted_node_ids,
                "pname": pname,
                "mode": mode,
            }
        )

    def track_dag(
        self,
        dag,
        start_id,
        end_id,
        pname,
        mode="append",
    ):
        """Track 'pname' parameter from 'start_id' to 'end_id'."""
        node_ids = nx.shortest_path(dag, source=start_id, target=end_id)
        self.dag_plot_data.append(
            {
                "ids": node_ids,
                "pname": pname,
                "mode": mode,
            }
        )

    def watch_adapter(
        self,
        adapter,
        pname,
    ):
        """Watch 'pname' parameter of the adapter snapshots."""
        self.adapter_plot_data.append(
            {
                "adapter": adapter,
                "pname": pname,
            }
        )

    def _prepare_plot(self):
        for i in range(len(self.dag_plot_data)):
            fig, ax = plt.subplots()
            if self.dag_plot_data[i]["mode"] == "append":
                (line,) = ax.plot([], [], marker="o")
            elif self.dag_plot_data[i]["mode"] == "replace":
                (line,) = ax.plot([], [])
            self.dag_plot_artists.append((fig, line))

        for i in range(len(self.adapter_plot_data)):
            fig, ax = plt.subplots()
            (line,) = ax.plot([], [])
            self.adapter_plot_artists.append((fig, line))

    def dag_plot_func_factory(self, line, plot_data, dag):
        def update(frame, line=line, plot_data=plot_data, dag=dag):
            node_ids = plot_data["ids"]
            pname = plot_data["pname"]
            with self.lock:
                if plot_data["mode"] == "append":
                    completed_node_ids = [
                        id
                        for id in node_ids
                        if dag.is_marked_as(id, "completed")
                    ]
                    xtick_label = [
                        dag.nodes[id]["name"] for id in completed_node_ids
                    ]
                    y = [
                        dag.nodes[id]["payload"][pname]
                        for id in completed_node_ids
                    ]
                    ax = line.axes
                    ax.set_xticks(range(len(xtick_label)))
                    ax.set_xticklabels(xtick_label, rotation=45, ha="right")
                elif plot_data["mode"] == "replace":
                    completed_node_ids = [
                        id
                        for id in node_ids
                        if dag.is_marked_as(id, "completed")
                    ]
                    if len(completed_node_ids) == 0:
                        y = []
                    else:
                        y = dag.nodes[completed_node_ids[-1]]["payload"][pname]
                    ax = line.axes
            line.set_data(range(len(y)), y)
            line.set_data(range(len(y)), y)
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"Parameter: {pname}")
            return (line,)

        return update

    def adapter_plot_func_factory(self, line, plot_data):
        def update(frame, line=line, plot_data=plot_data):
            pname = plot_data["pname"]
            # if plot_data["adapter"].snapshots[pname].empty():
            #     return (line,)
            y = plot_data["adapter"].snapshots[pname].get()
            if y is None:
                plot_data["adapter"].snapshots[pname].task_done()
            ax = line.axes
            line.set_data(range(len(y)), y)
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"Adapter Parameter: {pname}")
            return (line,)

        return update

    def on(self, dag, life_time=120):
        self._prepare_plot()
        for i, plot_data in enumerate(self.dag_plot_data):
            update_func = self.dag_plot_func_factory(
                self.dag_plot_artists[i][1], plot_data, dag
            )
            ani = FuncAnimation(
                self.dag_plot_artists[i][0],
                update_func,
                interval=100,
                blit=False,
                cache_frame_data=False,
            )

        for i, plot_data in enumerate(self.adapter_plot_data):
            update_func = self.adapter_plot_func_factory(
                self.adapter_plot_artists[i][1], plot_data
            )
            ani = FuncAnimation(
                self.adapter_plot_artists[i][0],
                update_func,
                interval=100,
                blit=False,
                cache_frame_data=False,
            )
        plt.show()
