# -*- coding: utf-8 -*-
"""Test utils module"""
from smefit import utils


def test_set_paths():

    root_path = __file__
    result_id = "fit_test"
    for pto in ["LO", "NLO"]:
        path_dict = utils.set_paths(root_path, pto, result_id)

        assert "table_path" in path_dict
        assert "corrections_path" in path_dict
        assert "theory_path" in path_dict
        assert "results_path" in path_dict
        assert pto in path_dict["corrections_path"]
        assert path_dict["root_path"] == __file__
