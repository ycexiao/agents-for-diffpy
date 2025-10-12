"""
Helper funcitons to handle the interface compatibility
"""

from collections import OrderedDict


def recipe_parameters_to_refinement_variales(parameters: OrderedDict):
    out = {}
    for par in parameters:
        out[par.name] = par.value
    return out
