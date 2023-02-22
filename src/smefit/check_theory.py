# -*- coding: utf-8 -*-
import pathlib
import pandas as pd
import numpy as np
from typing import Optional
import pandas as pd
from rich.table import Table
from rich.console import Console

from .loader import Loader

console = Console()


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    obs_name: str,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    rich_table = Table(title=obs_name)
    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [f"{x:.6f}" for x in value_list]
        rich_table.add_row(*row)

    return rich_table


def check_stadard_model(data_path, theory_path, threshold):
    """Check that |SM| provide in various tables is consistent.

    Parameters
    ----------
    data_path : pathlib.Path
        path to commondata folder, commondata excluded
    theory_path : pathlib.Path
        path to theory folder, theory excluded
    threshold : float
        threshold limit for NLO / SM best
    """

    Loader.commondata_path = pathlib.Path(data_path.absolute())
    for file in theory_path:
        log_table = {}
        Loader.theory_path = pathlib.Path(file.parent)
        data_name = file.stem
        for order in ["LO", "NLO"]:
            loader = Loader(
                data_name,
                operators_to_keep=[],
                order=order,
                use_quad=False,
                use_theory_covmat=True,
                use_multiplicative_prescription=False,
                rot_to_fit_basis=None,
            )
            raw_theory = loader.load_raw_theory()
            log_table["Exp"] = loader.central_values
            log_table["SM best"] = loader.sm_prediction
            log_table[order] = raw_theory[order]["SM"]

        log_table = pd.DataFrame(log_table)
        log_table["Ratio NLO/best"] = log_table.NLO / log_table["SM best"]
        log_table["Ratio LO/best"] = log_table.LO / log_table["SM best"]

        if np.any(np.abs(log_table["Ratio NLO/best"] - 1.0) < threshold):
            console.log(df_to_table(log_table, data_name), style="green")
        else:
            console.log(df_to_table(log_table, data_name), style="red")
