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

    def proceed_from_graph(
        self,
        workflow: DiGraph,
        adapter: PDFAdapter,
    ):
        frontier_nodes = {
            str(uuid.uuid4()): n
            for n in workflow.nodes
            if workflow.in_degree(n) == 0
        }
        workers = {key: FitWorker(adapter) for key in frontier_nodes.keys()}
        while frontier_nodes:
            upcoming_frontier_nodes = {}
            for key in frontier_nodes:
                workers[key].refine(
                    [workflow.nodes[frontier_nodes[key]]["value"]],
                )
                # If no successor, ignore it
                if workflow.out_degree(frontier_nodes[key]) == 0:
                    continue
                # If one successor, move forward
                elif workflow.out_degree(frontier_nodes[key]) == 1:
                    succ = list(workflow.successors(frontier_nodes[key]))[0]
                    upcoming_frontier_nodes[key] = succ
                # If multiple successors, branch out
                elif workflow.out_degree(frontier_nodes[key]) > 1:
                    succs = list(workflow.successors(frontier_nodes[key]))
                    upcoming_frontier_nodes[key] = succs[0]
                    for succ in succs[1:]:
                        new_key = str(uuid.uuid4())
                        upcoming_frontier_nodes[new_key] = succ
                        workers[new_key] = FitWorker(copy.deepcopy(adapter))
                        workers[new_key].adapter.apply_parameters_values(
                            workers[key].adapter.get_parameters_values_dict()
                        )
            frontier_nodes = upcoming_frontier_nodes
        self.workers = workers


class FitWorker:
    def __init__(self, adapter: PDFAdapter):
        self.adapter = adapter
        self.terminated = False
        self.allowed_actions = list(adapter.parameters.keys())
        self.refine_steps = []

    def refine(self, refine_param_names):
        if not set(refine_param_names).issubset(set(self.allowed_actions)):
            raise ValueError("Refine worker get unknown parameter names.")
        self.adapter.free_parameters(refine_param_names)
        param_value_before_fit = self.adapter.parameter_values_in_slots
        out = least_squares(
            self.adapter.residual, self.adapter.initial_values, x_scale="jac"
        )
        param_value_after_fit = self.adapter.parameter_values_in_slots
        param_variation = [
            (
                (param_value_after_fit[i] - param_value_before_fit[i])
                / param_value_before_fit[i]
                if param_value_before_fit[i] != 0
                else 0
            )
            for i in range(len(refine_param_names))
        ]
        array_summary = (
            *param_variation,
            out.success,
            out.cost,
            out.fun,
            out.optimality,
            out.jac,
        )
        self.refine_steps.append((refine_param_names, array_summary))
