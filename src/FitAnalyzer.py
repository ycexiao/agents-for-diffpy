import matplotlib.pyplot as plt
import threading
import re
import networkx as nx
import warnings
from matplotlib.animation import FuncAnimation


class FitAnalyzer:
    def __init__(self, lock):
        self.lock = lock
        self.plot_data = []
        self.plot_artists = []

    def watch(self, dag, step_name, pname, mode="append"):
        assert step_name in dag.unique_names
        step_names = []
        node_ids = []
        for node_id, node_content in dag.nodes(data=True):
            if node_content["name"].startswith(step_name):
                step_names.append(node_content["name"])
                node_ids.append(node_id)
        step_names[step_names.index(step_name)] = step_name + "_0"
        order = [int(re.findall(r"_(\d+)$", name)[0]) for name in step_names]
        sorted_node_ids = [id for _, id in sorted(zip(order, node_ids))]
        self.plot_data.append(
            {
                "ids": sorted_node_ids,
                "pname": pname,
                "mode": mode,
            }
        )

    def track(self, dag, start_id, end_id, pname, mode="append"):
        node_ids = nx.shortest_path(dag, source=start_id, target=end_id)
        self.plot_data.append(
            {
                "ids": node_ids,
                "pname": pname,
                "mode": mode,
            }
        )

    def _prepare_plot(self):
        for i in range(len(self.plot_data)):
            fig, ax = plt.subplots()
            if self.plot_data[i]["mode"] == "append":
                (line,) = ax.plot([], [], marker="o")
            elif self.plot_data[i]["mode"] == "replace":
                (line,) = ax.plot([], [])
            self.plot_artists.append((fig, line))

    def update_func_factory(self, line, plot_data, dag):
        def update(frame, line=line, plot_data=plot_data, dag=dag):
            node_ids = plot_data["ids"]
            pname = plot_data["pname"]
            with self.lock:
                if plot_data["mode"] == "append":
                    completed_node_ids = [
                        id for id in node_ids if dag.hasCompleted(id)
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
                        id for id in node_ids if dag.hasCompleted(id)
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

    def on(self, dag, life_time=120):
        self._prepare_plot()
        for i, plot_data in enumerate(self.plot_data):
            update_func = self.update_func_factory(
                self.plot_artists[i][1], plot_data, dag
            )
            ani = FuncAnimation(
                self.plot_artists[i][0],
                update_func,
                interval=100,
                blit=False,
                cache_frame_data=False,
            )
        plt.show()
