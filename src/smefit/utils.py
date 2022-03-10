# -*- coding: utf-8 -*-

# TODO: remove this?
def set_paths(root_path, pto, resultID):
    """
    Set up configurations paths given the dictionary

    Parameters
    ----------
        root_path: str
            root path where tables and results are located
        pto: str
            pto for the EFT corrections LO, NLO
        resultID: str
            result ID name

    Returns
    -------
        path_dict: dict
            dictionary with path configuarions
    """
    # Root path
    path_dict = {"root_path": f"{root_path}"}

    # Set paths to theory tables
    path_dict.update(
        {
            "table_path": f"{path_dict['root_path']}/commondata/",
            "corrections_path": f"{path_dict['root_path']}/tables/operator_res/{pto}/",
            "theory_path": f"{path_dict['root_path']}/theory/",
            "results_path": f"{path_dict['root_path']}/results/{resultID}",
        }
    )
    return path_dict
