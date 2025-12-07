from BaseAdapter import BaseAdapter
from PDFAdapter import PDFAdapter
from FitDiag import FitDiag
from scipy.optimize import least_squares
from collections import OrderedDict
import uuid
import copy
from typing import Literal
from networkx import DiGraph
from FitDiag import FitDiag
import numpy


class FitRunner:
    def __init__(self, parameter_names=[]):
        self.refine_operation = {}
        self.workers = []

    def load_workflow(
        self,
        workflow: DiGraph,
        adapter: PDFAdapter,
    ):
        frontiers = {
            str(uuid.uuid4()): n
            for n in workflow.nodes
            if workflow.in_degree(n) == 0
        }
        workers = {
            key: FitWorker(adapter, depends_on=None)
            for key in frontiers.keys()
        }
        jobs_to_workers_id = {}
        while frontiers:
            next_frontiers = {}
            for key in frontiers:
                # If no successor, ignore it
                if workflow.out_degree(frontiers[key]) == 0:
                    continue
                # If one successor, move forward
                elif workflow.out_degree(frontiers[key]) == 1:
                    succ = list(workflow.successors(frontiers[key]))[0]
                    next_frontiers[key] = succ
                    job_id = str(uuid.uuid4())
                    workers[key].assign_job(
                        [workflow.nodes[frontiers[key]]["value"]],
                        job_id=job_id,
                    )
                    jobs_to_workers_id[job_id] = key
                # If multiple successors, branch out
                elif workflow.out_degree(frontiers[key]) > 1:
                    succs = list(workflow.successors(frontiers[key]))
                    next_frontiers[key] = succs[0]
                    main_job_id = str(uuid.uuid4())
                    workers[key].assign_job(
                        [workflow.nodes[frontiers[key]]["value"]],
                        job_id=main_job_id,
                    )
                    jobs_to_workers_id[main_job_id] = key
                    for succ in succs[1:]:
                        new_key = str(uuid.uuid4())
                        next_frontiers[new_key] = succ
                        job_id = str(uuid.uuid4())
                        workers[new_key] = copy.deepcopy(workers[key])
                        workers[new_key].depends_on = main_job_id
                        workers[new_key].assign_job(
                            [workflow.nodes[frontiers[key]]["value"]],
                            job_id=job_id,
                        )
                        jobs_to_workers_id[job_id] = new_key
            frontiers = next_frontiers
        self.workers = workers
        self.jobs_to_workers_id = jobs_to_workers_id

    def run(self):
        active_workers = [
            worker
            for key, worker in self.workers.items()
            if worker.depends_on is None
        ]
        branch_job_to_worker = {
            worker.depends_on: worker
            for _, worker in self.workers.items()
            if worker.depends_on is not None
        }
        max_depth = 50
        for _ in range(max_depth):
            if not active_workers:
                break
            next_active_workers = []
            for worker in active_workers:
                job_id, on_params, array_summary, terminated = (
                    worker.next_job()
                )
                if not terminated:
                    next_active_workers.append(worker)
                if job_id in branch_job_to_worker:
                    next_worker = branch_job_to_worker[job_id]
                    next_worker.adapter.apply_parameters_values(
                        worker.adapter.get_parameters_values_dict()
                    )
                    next_active_workers.append(next_worker)


class FitWorker:
    def __init__(self, adapter: PDFAdapter, depends_on=None):
        self.adapter = adapter
        self.parameter_names = adapter.parameter_names
        self.job_id_queue = []
        self.refine_params_queue = []
        self.terminated = False
        self.depends_on = depends_on
        self.summary = []

    def assign_job(self, refine_param_names, job_id):
        if not set(refine_param_names).issubset(set(self.parameter_names)):
            raise ValueError("Refine worker get unknown parameter names.")
        self.refine_params_queue.append(refine_param_names)
        self.job_id_queue.append(job_id)

    def next_job(self):
        job_id = self.job_id_queue.pop(0)
        refine_param_names = self.refine_params_queue.pop(0)
        self.adapter.free_parameters(refine_param_names)
        param_value_before_fit = [
            self.adapter.get_parameters_values_dict()[key]
            for key in refine_param_names
        ]
        out = least_squares(
            self.adapter.residual, self.adapter.initial_values, x_scale="jac"
        )
        param_value_after_fit = [
            self.adapter.get_parameters_values_dict()[key]
            for key in refine_param_names
        ]
        param_variation = [
            (param_value_after_fit[i] - param_value_before_fit[i])
            / param_value_before_fit[i]
            for i in range(len(refine_param_names))
        ]
        formatted_param_names = []
        formatted_param_variations = []
        for i, pname in enumerate(refine_param_names):
            if "Uiso" in pname:
                suffix = pname.split("Uiso")[-1]
                formatted_param_names.append(f"U11{suffix}")
                formatted_param_names.append(f"U22{suffix}")
                formatted_param_names.append(f"U33{suffix}")
                formatted_param_variations.append(param_variation[i])
                formatted_param_variations.append(param_variation[i])
                formatted_param_variations.append(param_variation[i])
            else:
                formatted_param_names.append(pname)
                formatted_param_variations.append(param_variation[i])
        params_variations = numpy.zeros(len(self.adapter.parameters_slots))
        for i, pname in enumerate(formatted_param_names):
            slot_index = self.adapter.parameters_slots.index(pname)
            params_variations[slot_index] = formatted_param_variations[i]
        on_params = numpy.where(params_variations != 0)[0]
        array_summary = (
            *params_variations,
            out.cost,
            out.fun,
            out.optimality,
            out.jac,
        )
        terminated = len(self.job_id_queue) == 0
        return job_id, on_params, array_summary, terminated
