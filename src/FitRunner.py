from BaseAdapter import BaseAdapter
from PDFAdapter import PDFAdapter
from FitDAG import FitDAG
from scipy.optimize import least_squares
from collections import OrderedDict
import uuid
import copy
from typing import Literal
from networkx import DiGraph
from FitDAG import FitDAG
import numpy
import warnings

# copy.deepcopy(adapter)
# ERROR   RuntimeError: Pickling of
# "diffpy.srreal.parallel.ParallelPairQuantity" instances is not enabled
# (http://www.boost.org/libs/python/doc/v2/pickle.html)


class FitRunner:
    def __init__(self):
        self.workers = {}

    def run_workflow(
        self,
        dag: FitDAG,
        adapter: PDFAdapter,
        start_nodes_ids=None,
        strict=False,
    ):
        if start_nodes_ids is None:
            frontier_node_ids = []
            for node_id in dag.graph.nodes:
                if dag.graph.in_degree(node_id) == 0:
                    frontier_node_ids.append(node_id)
                    dag.graph.nodes[node_id]["parameter_values"] = (
                        copy.deepcopy(adapter.parameter_values_in_slots)
                    )
                    dag.tag_node(
                        node_id,
                        "initialized",
                    )
        else:
            frontier_node_ids = start_nodes_ids
            for node_id in frontier_node_ids:
                paraent_node_id = list(dag.graph.predecessors(node_id))[0]
                dag.graph.nodes[node_id]["parameter_values"] = copy.deepcopy(
                    dag.graph.nodes[paraent_node_id]["parameter_values"]
                )
                dag.tag_node(
                    node_id,
                    "initialized",
                )
        while frontier_node_ids:
            upcoming_frontier_node_ids = []
            for node_id in frontier_node_ids:
                if dag.tag_is(node_id, "initialized"):
                    (
                        action,
                        status,
                        reward,
                        observation,
                        parameter_values,
                    ) = self.run_node(
                        adapter,
                        dag.graph.nodes[node_id]["parameter_values"],
                        dag.graph.nodes[node_id]["value"],
                        strict=strict,
                    )
                    dag.graph.nodes[node_id].update(
                        {
                            "action": action,
                            "status": status,
                            "reward": reward,
                            "observation": observation,
                            "parameter_values": parameter_values,
                        }
                    )
                    dag.tag_node(node_id, "completed")
                else:
                    # FIXME
                    pass
                # If no successor, ignore it
                if dag.graph.out_degree(node_id) == 0:
                    continue
                # If one successor, move forward
                elif dag.graph.out_degree(node_id) == 1:
                    succ_id = list(dag.graph.successors(node_id))[0]
                    dag.graph.nodes[succ_id]["parameter_values"] = (
                        copy.deepcopy(
                            dag.graph.nodes[node_id]["parameter_values"]
                        )
                    )
                    dag.tag_node(succ_id, "initialized")
                    upcoming_frontier_node_ids.append(succ_id)
                # If multiple successors, branch out
                elif dag.graph.out_degree(frontier_node_ids[node_id]) > 1:
                    succ_ids = list(
                        dag.graph.successors(frontier_node_ids[node_id])
                    )
                    upcoming_frontier_node_ids.append(succ_ids[0])
                    for succ_id in succ_ids[1:]:
                        dag.graph.nodes[succ_id]["parameter_values"] = (
                            copy.deepcopy(
                                dag.graph.nodes[frontier_node_ids[node_id]][
                                    "parameter_values"
                                ]
                            )
                        )
                        dag.tag_node(succ_id, "initialized")
                        upcoming_frontier_node_ids.append(succ_id)
            frontier_node_ids = upcoming_frontier_node_ids
        return dag

    def run_node(
        self, adapter, parameter_values, refine_param_names, strict=False
    ):
        if "all" in refine_param_names:
            refine_param_names = ["all"]
        else:
            if not set(refine_param_names).issubset(
                set(adapter.parameters.keys())
            ):
                if strict:
                    raise ValueError(
                        "Refine worker get unknown parameter names."
                    )
                else:
                    invalid_operations = list(
                        set(refine_param_names)
                        - set(adapter.parameters.keys())
                    )
                    warnings.warn(
                        "Refine worker get Invalide parameter names. "
                        f"Ignoring {invalid_operations}."
                    )
                    refine_param_names = list(
                        set(refine_param_names).intersection(
                            set(adapter.parameters.keys())
                        )
                    )
        adapter.free_parameters(refine_param_names)
        res = least_squares(
            adapter.residual, adapter.initial_values, x_scale="jac"
        )
        refine_param_indexes = [
            1 if name in refine_param_names else 0
            for name in adapter.parameter_names_in_slots
        ]
        action = refine_param_indexes
        status = [
            adapter.parameter_slots_mask,
            adapter.parameter_values_in_slots,
        ]
        reward = adapter.residual_scalar
        observation = adapter.observation
        parameter_values = adapter.parameter_values_in_slots
        adapter.fix_parameters(refine_param_names)
        return action, status, reward, observation, parameter_values
