from FitDAG import FitDAG
from FitRunner import FitRunner
from FitPlotter import FitPlotter
from PDFAdapter import PDFAdapter
from pathlib import Path
import re
import time
import threading
import networkx as nx
import uuid


class PDFFitLauncher:
    def __init__(self):
        self.profile_folder = None
        self.structure_file = None
        self.initial_payload = None
        self.template_dag = None
        self.last_payload = None
        self.collect_data_event = {}
        self.profiles_known = []
        self.profiles_finished = []
        self.profiles_running = []
        self.runner = FitRunner()
        self.plotter = FitPlotter()

    def _launch(self):
        while True:
            time.sleep(0.05)
            if not self.profile_folder:
                raise ValueError("Profile folder is not set.")
            if not self.structure_file:
                raise ValueError("Structure file is not set.")
            files = [file for file in self.profile_folder.glob("*.gr")]
            order = [
                int(re.findall(r"(\d+)K\.gr", file.name)[-1]) for file in files
            ]
            files = [file for _, file in sorted(zip(order, files))]
            if self.profiles_known != files[: len(self.profiles_known)]:
                raise ValueError(
                    "Profiles order has changed. "
                    "Please ensure newer profiles are strictly have higher indices."
                )
            self.profiles_known = files
            self.profiles_running = [
                file
                for file in self.profiles_known
                if file not in self.profiles_finished
            ]
            if not self.profiles_running:
                continue
            for i, profile in enumerate(self.profiles_running):
                if self.last_payload is not None:
                    payload = self.last_payload
                else:
                    payload = self.initial_payload
                inputs = {
                    "profile_string": profile.read_text(),
                    "structure_string": self.structure_file.read_text(),
                    "xmin": 1.5,
                    "xmax": 50,
                    "dx": 0.01,
                    "qmax": 25.0,
                    "qmin": 0.1,
                }
                dag = self.template_dag.clean_copy(
                    with_payload=False,
                    with_same_id=False,
                    return_type="FitDAG",
                )
                self.runner._run_dag(
                    dag, Adapter=PDFAdapter, inputs=inputs, payload=payload
                )
                last_node_id = list(nx.topological_sort(dag))[-1]
                self.last_payload = dag.nodes[last_node_id]["payload"]
                dag.to_json(
                    self.dump_folder
                    / f"{self.dump_filename}_{profile.stem}.json"
                )
                self.profiles_finished.append(profile)
                print(f"Finsihed {len(self.profiles_finished)+1} fit tasks.")
            self.profiles_running = []

    def set_meta_inputs(
        self,
        profile_folder: Path,
        structure_file: Path,
        initial_payload: dict,
        dump_folder: Path,
        dump_filename: str,
    ):
        self.profile_folder = profile_folder
        self.structure_file = structure_file
        self.initial_payload = initial_payload
        self.dump_folder = dump_folder
        self.dump_filename = dump_filename

    def set_template_dag(self, template_dag: FitDAG):
        self.template_dag = template_dag

    def watch_stable(self, pname, node_name="all", **kwargs):
        """View the partial results realtime"""
        window_id = str(uuid.uuid4())
        default_options = {
            "update_mode": "append",
            "source": "payload",
            "title": pname,
            "style": "sparse",
            "window_id": window_id,
        }
        default_options.update(kwargs)
        self.runner.watch(
            trigger_func=lambda dag, node_id: dag.nodes[node_id]["name"]
            == node_name,
            pname=pname,
            **default_options,
        )

    def watch_intermediate(self, pname, **kwargs):
        """View the intermediate results realtime.
        The intermediate results are usually only for the diagnostic purposes
        of the fitting process.
        """
        window_id = str(uuid.uuid4())
        default_options = {
            "update_mode": "append",
            "source": "adapter",
            "title": pname,
            "style": "dense",
            "window_id": window_id,
        }
        default_options.update(kwargs)
        self.runner.watch(
            trigger_func=lambda dag, node_id: True,
            pname=pname,
            **default_options,
        )

    def launch(self):
        if self.runner.data_for_plot:
            self.plotter.connect_to_runner(self.runner)
        t = threading.Thread(target=self._launch)
        t.start()
        self.plotter.on()


if __name__ == "__main__":
    launcher = PDFFitLauncher()
    launcher.set_meta_inputs(
        profile_folder=Path("example/data/sequential_fit"),
        structure_file=Path("data/Ni.cif"),
        initial_payload={
            "scale": 0.4,
            "a": 3.52,
            "Uiso_0": 0.005,
            "delta2": 2.0,
            "qdamp": 0.04,
            "qbroad": 0.02,
        },
        dump_folder=Path("example/data/results"),
        dump_filename="fit_results",
    )
    template_dag = FitDAG()
    template_dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
    launcher.set_template_dag(template_dag)
    launcher.watch_stable("a", node_name="all")
    launcher.watch_intermediate("ycalc_0", update_mode="replace")
    launcher.launch()
