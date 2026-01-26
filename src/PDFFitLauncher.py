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
        self.filename_pattern = None
        self.profiles_known = []
        self.profiles_finished = []
        self.profiles_running = []
        self.runner = FitRunner()
        self.plotter = FitPlotter()

    def _check_for_new_profiles(self):
        if not self.profile_folder:
            raise ValueError("Profile folder is not set.")
        if not self.structure_file:
            raise ValueError("Structure file is not set.")
        files = [file for file in self.profile_folder.glob("*")]
        order = [
            int(re.findall(self.filename_pattern, file.name)[0])
            for file in files
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

    def _launch(self):
        while True:
            time.sleep(0.05)
            self._check_for_new_profiles()
            if not self.profiles_running:
                continue
            for profile in self.profiles_running:
                if self.last_payload is not None:
                    payload = self.last_payload
                else:
                    payload = self.initial_payload
                inputs = {
                    "profile_string": profile.read_text(),
                    "structure_string": self.structure_file.read_text(),
                    **self.inputs_kwargs,
                }
                dag = self.template_dag.copy(
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
        template_dag: FitDAG,
        xmin,
        xmax,
        dx,
        qmin,
        qmax,
        remove_vars,
        filename_pattern: str = "(\d+)K\.gr",
    ):
        self.profile_folder = profile_folder
        self.structure_file = structure_file
        self.initial_payload = initial_payload
        self.dump_folder = dump_folder
        self.dump_filename = dump_filename
        self.template_dag = template_dag
        self.filename_pattern = filename_pattern
        self.inputs_kwargs = {
            "xmin": xmin,
            "xmax": xmax,
            "dx": dx,
            "qmin": qmin,
            "qmax": qmax,
            "remove_vars": remove_vars,
        }

    def watch(self, pname, when, **kwargs):
        """
        Moniter the results generated during the fit process.

        Parameters
        ----------
        pname : str
            The name of the plot to monitor.
        when : {"node end", "dag end", "all"}
            Specifies when to update the plot.
            "node end" updates the plot at the end of each node,
            "dag end" updates the plot at the end of each DAG, and
            "all" updates the plot during every iteration in the least-squares.

            By default, when in "node end" and "dag end" mode,
            `source` is set to "payload", `update_mode` is set to "append", and
            when in "all" mode, `source` is set to "adapter", `update_mode` is
            set to "replace". Please see FitRunner.watch for more details.

        **kwargs : dict
            Additional keyword arguments for the plot.
        """
        window_id = str(uuid.uuid4())
        if when == "node end":
            trigger_func = lambda dag, node_id: True
            source = "payload"
            update_mode = "append"
        elif when == "dag end":
            trigger_func = (
                lambda dag, node_id: node_id
                == list(nx.topological_sort(dag))[-1]
            )
            source = "payload"
            update_mode = "append"
        elif when == "all":
            trigger_func = lambda dag, node_id: True
            source = "adapter"
            update_mode = "replace"
        else:
            raise ValueError(
                f"Invalid value for 'when': {when} "
                "Please choose one of {'node end', 'dag end', 'all'}."
            )
        if "source" not in kwargs:
            kwargs["source"] = source
        if "update_mode" not in kwargs:
            kwargs["update_mode"] = update_mode
        self.runner.watch(
            trigger_func=trigger_func,
            pname=pname,
            **kwargs,
        )

    def launch(self):
        if self.runner.data_for_plot:
            self.plotter.connect_to_runner(self.runner)
        t = threading.Thread(target=self._launch)
        t.start()
        self.plotter.on()


if __name__ == "__main__":
    launcher = PDFFitLauncher()
    template_dag = FitDAG()
    template_dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
    launcher.set_meta_inputs(
        profile_folder=Path("example/data/sequential_fit"),
        structure_file=Path("example/data/Ni.cif"),
        initial_payload={
            "scale": 0.4,
            "a": 3.52,
            "Uiso_0": 0.005,
            "delta2": 2.0,
            "qdamp": 0.04,
            "qbroad": 0.02,
        },
        dump_folder=Path("example/results"),
        dump_filename="fit_results",
        template_dag=template_dag,
        filename_pattern="(\d+)K\.gr",
        xmin=1.5,
        xmax=50,
        dx=0.01,
        qmax=25.0,
        qmin=0.1,
        remove_vars=["delta1"],
    )
    launcher.watch("a", when="dag end")
    launcher.watch("ycalc_0", when="all", update_mode="replace")
    launcher.launch()
