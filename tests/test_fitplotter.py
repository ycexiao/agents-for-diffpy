import queue

from matplotlib.pyplot import style
from FitRunner import FitRunner
from FitPlotter import FitPlotter
from FitDAG import FitDAG
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
import sys
from types import SimpleNamespace
from queue import Queue
import threading
import uuid
import numpy as np
import time
from PDFAdapter import PDFAdapter
import re

## test_plotter_without_fitrunner
# queues = [Queue() for _ in range(5)]

# def produce():
#     while True:
#         t = time.time()
#         for j, queue in enumerate(queues):
#             values = [np.sin(t + i * 0.1 + j) for i in range(10)]
#             queue.put(values)
#         time.sleep(0.5)

# producer = threading.Thread(target=produce, daemon=True)
# producer.start()

# data_for_plot = {}
# for i in range(5):
#     data_for_plot[str(uuid.uuid4())] = {
#         "ydata": queues[i],
#         "title": f"test_{i}",
#     }
# runner = SimpleNamespace(data_for_plot=data_for_plot)
# plotter = FitPlotter()
# plotter.connect_to_runner(runner)
# plotter.on()


## plot single dag
# profile_path = Path("data/Ni.gr")
# structure_path = Path("data/Ni.cif")
# inputs = {
#     "profile_string": profile_path.read_text(),
#     "structure_string": structure_path.read_text(),
#     "xmin": 1.5,
#     "xmax": 50,
#     "dx": 0.01,
#     "qmax": 25.0,
#     "qmin": 0.1,
# }
# payload = {
#     "scale": 0.4,
#     "a": 3.52,
#     "Uiso_0": 0.005,
#     "delta2": 2.0,
#     "qdamp": 0.04,
#     "qbroad": 0.02,
# }
# dag = FitDAG()
# dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
# runner = FitRunner()
# window_id = str(uuid.uuid4())
# runner.watch(
#     trigger_func=lambda dag, node_id: True,
#     pname="a",
#     update_mode="append",
#     source="payload",
#     title="lattice parameter",
#     style="sparse",
#     window_id=window_id,
# )
# plotter = FitPlotter()
# plotter.connect_to_runner(runner)
# kwargs = {
#     "dag": dag,
#     "Adapter": PDFAdapter,
#     "inputs": inputs,
#     "payload": payload,
# }
# t = runner.get_run_dag_func(**kwargs)
# t.start()
# plotter.on()
# t.join()


## plot sequential dags
# data_dir_path = Path().cwd() / "data" / "sequential_fit"
# files = [file for file in data_dir_path.glob("*.gr")]
# order = [int(re.search(r"(\d+)K\.gr", file.name).group(1)) for file in files]
# files = [file for _, file in sorted(zip(order, files))]
# structure_path = Path().cwd() / "data" / "Ni.cif"
# inputs = [
#     {
#         "profile_string": files[i].read_text(),
#         "structure_string": structure_path.read_text(),
#         "xmin": 1.5,
#         "xmax": 50,
#         "dx": 0.01,
#         "qmax": 25.0,
#         "qmin": 0.1,
#     }
#     for i in range(3)
# ]
# payload = {
#     "scale": 0.4,
#     "a": 3.52,
#     "Uiso_0": 0.005,
#     "delta2": 2.0,
#     "qdamp": 0.04,
#     "qbroad": 0.02,
# }
# dag = FitDAG()
# dag.from_str("a->scale->qdamp->Uiso_0->delta2->all")
# dags = [dag]
# for _ in range(2):
#     new_dag = dag.clean_copy(
#         with_payload=False, with_same_id=False, return_type="FitDAG"
#     )
#     dags.append(new_dag)

# runner = FitRunner()
# runner.watch(
#     trigger_func=lambda dag, node_id: True,
#     pname="a",
#     update_mode="append",
#     source="payload",
#     title="a",
#     style="sparse",
# )
# kwargs = {
#     "dags": dags,
#     "Adapter": PDFAdapter,
#     "inputs": inputs,
#     "payload": payload,
# }
# plotter = FitPlotter()
# plotter.connect_to_runner(runner)
# t = runner.get_run_sequential_func(**kwargs)
# t.start()
# plotter.on()
# t.join()
