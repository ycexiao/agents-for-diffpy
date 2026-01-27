from agents_for_diffpy.interface import (
    FitDAG,
    FitRunner,
    FitPlotter,
    PDFAdapter,
)
from pathlib import Path
import re
import time
import threading
import networkx as nx


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
                "Please ensure newer profiles are strictly have higher "
                "indices."
            )
        self.profiles_known = files
        self.profiles_running = [
            file
            for file in self.profiles_known
            if file not in self.profiles_finished
        ]

    def set_start_profile(self, start_from):
        if start_from is None:
            return
        else:
            self._check_for_new_profiles()
            ind = self.profiles_known.index(
                Path(self.profile_folder / start_from)
            )
            self.profiles_finished = self.profiles_known[:ind]
            result_files = [file for file in self.dump_folder.glob("*.json")]
            result_to_profile_stem = [
                re.findall(f"{self.dump_filename}_(.*).json", file.name)[0]
                for file in result_files
            ]
            if Path(start_from).stem not in result_to_profile_stem:
                raise ValueError(
                    f"No result file found for start profile {start_from}"
                )
            result_file = result_files[
                result_to_profile_stem.index(Path(start_from).stem)
            ]
            dag = FitDAG()
            dag.from_json(result_file)
            self.last_payload = dag.nodes[list(nx.topological_sort(dag))[-1]][
                "payload"
            ]

    def _launch(self):
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
                self.dump_folder / f"{self.dump_filename}_{profile.stem}.json"
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
        filename_pattern: str = r"(\d+)K\.gr",
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
        """Moniter the results generated during the fit process.

        Parameters
        ----------
        pname : str
            The name of the plot to monitor.
        when : {"node end", "dag end", "all"}
            The string used to specify when to update the plot.
            "node end" updates the plot at the end of each node,
            "dag end" updates the plot at the end of each DAG, and
            "all" updates the plot during every iteration in the least-squares.

            This variable specifies several default behaviors for the plot,
            including the trigger function, the source of the data,
            the update mode, and the plot style. Please see FitRunner.watch
            for more details.

        **kwargs : dict
            Additional keyword arguments for the plot.
        """
        if when == "node end":

            def trigger_func(dag, node_id):
                return True

            source = "payload"
            update_mode = "append"
            style = "sparse"
        elif when == "dag end":
            trigger_func = (
                lambda dag, node_id: node_id
                == list(nx.topological_sort(dag))[-1]
            )
            source = "payload"
            update_mode = "append"
            style = "sparse"
        elif when == "all":

            def trigger_func(dag, node_id):
                return True

            source = "adapter"
            update_mode = "replace"
            style = "dense"
        else:
            raise ValueError(
                f"Invalid value for 'when': {when} "
                "Please choose one of {'node end', 'dag end', 'all'}."
            )
        if "source" not in kwargs:
            kwargs["source"] = source
        if "update_mode" not in kwargs:
            kwargs["update_mode"] = update_mode
        if "style" not in kwargs:
            kwargs["style"] = style
        if "trigger_func" not in kwargs:
            kwargs["trigger_func"] = trigger_func
        self.runner.watch(
            pname=pname,
            **kwargs,
        )

    def launch(self, mode="stream"):
        """Launch the fitting process.

        Parameters
        ----------
        mode : {"stream", "batch"}, optional
            The mode in which to launch the fitting process.
            Default is 'stream'.
            When in "stream" mode, the fitting tasks are created whenever
            new data is available.
            When in "batch" mode, the fitting tasks are  created sequentially
            for data available when the fitting process is launched.
        start_from : str
            The profile name from which the fitting process should start.
            Default is None, which means the fitting process will start from
            the profile with minimum order number (e.g. "1K.gr")
            When start_from is specified, the preceding profiles should have
            been executed and their results should be available. If not,
            RuntimeError will be raised.
        """
        if mode == "stream":

            def _launch_stream():
                while True:
                    time.sleep(0.05)  # 20 Hz
                    self._check_for_new_profiles()
                    self._launch()

            t = threading.Thread(target=_launch_stream)

        elif mode == "batch":

            def _launch_batch():
                self._check_for_new_profiles()
                if self.profiles_running is not None:
                    self._launch()

            t = threading.Thread(target=_launch_batch)
        t.start()
        plotter = FitPlotter()
        plotter.connect_to_runner(self.runner)
        plotter.on()


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
        filename_pattern=r"(\d+)K\.gr",
        xmin=1.5,
        xmax=50,
        dx=0.01,
        qmax=25.0,
        qmin=0.1,
        remove_vars=["delta1"],
    )
    launcher.watch("a", when="dag end")
    launcher.watch("ycalc_0", when="all", update_mode="replace")
    launcher.set_start_profile("Ni_PDF_20250922-234708_04f3a7_60K.gr")
    launcher.launch(mode="batch")
