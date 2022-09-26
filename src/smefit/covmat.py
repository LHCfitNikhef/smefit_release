import numpy as np
import pandas as pd
import scipy.linalg as la

from .covmat_utils import construct_covmat

INTRA_DATASET_SYS_NAME = ("UNCORR", "CORR", "THEORYUNCORR", "THEORYCORR")


def dataset_inputs_covmat_from_systematics(
    dataset_inputs_loaded_cd_with_cuts,
    _list_of_central_values=None
):
    """Given a list containing :py:class:`validphys.coredata.CommonData` s,
    construct the full covariance matrix.

    This is similar to :py:meth:`covmat_from_systematics`
    except that special corr systematics are concatenated across all datasets
    before being multiplied by their transpose to give off block-diagonal
    contributions. The other systematics contribute to the block diagonal in the
    same way as :py:meth:`covmat_from_systematics`.

    Parameters
    ----------
    dataset_inputs_loaded_cd_with_cuts : list[(stat_errors: np.array, sys_errors: pd.DataFrame)]
        list of stat_errors and sys_errors for each dataset.

    _list_of_central_values: None, list[np.array]
        list of 1-D arrays which contain alternative central values which are
        combined with the multiplicative errors to calculate their absolute
        contribution. By default this is None and the experimental central
        values are used.

    Returns
    -------
    cov_mat : np.array
        Numpy array which is N_dat x N_dat (where N_dat is the number of data points after cuts)
        containing uncertainty and correlation information.

    Example
    -------
    This function can be called directly from the API:

    >>> dsinps = [
    ...     {'dataset': 'NMC'},
    ...     {'dataset': 'ATLASTTBARTOT', 'cfac':['QCD']},
    ...     {'dataset': 'CMSZDIFF12', 'cfac':('QCD', 'NRM'), 'sys':10}
    ... ]
    >>> inp = dict(dataset_inputs=dsinps, theoryid=162, use_cuts="internal")
    >>> cov = API.dataset_inputs_covmat_from_systematics(**inp)
    >>> cov.shape
    (235, 235)

    Which properly accounts for all dataset settings and cuts.

    """
    special_corrs = []
    block_diags = []

    if _list_of_central_values is None:
        # want to just pass None to systematic_errors method
        _list_of_central_values = [None] * len(dataset_inputs_loaded_cd_with_cuts)

    for cd, central_values in zip(
        dataset_inputs_loaded_cd_with_cuts,
        _list_of_central_values
    ):
        
        sys_errors = cd[0]
        stat_errors = cd[1]
        # separate out the special uncertainties which can be correlated across
        # datasets
        is_intra_dataset_error = sys_errors.columns.isin(INTRA_DATASET_SYS_NAME)
        block_diags.append(construct_covmat(
            stat_errors, sys_errors.loc[:, is_intra_dataset_error]))
        special_corrs.append(sys_errors.loc[:, ~is_intra_dataset_error])

    import pdb; pdb.set_trace()
    # concat systematics across datasets
    special_sys = pd.concat(special_corrs, axis=0, sort=False)
    # non-overlapping systematics are set to NaN by concat, fill with 0 instead.
    special_sys.fillna(0, inplace=True)

    diag = la.block_diag(*block_diags)
    covmat = diag + special_sys.to_numpy() @ special_sys.to_numpy().T
    return covmat