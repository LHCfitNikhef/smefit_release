import pathlib

import tests.test_loader as ld
import numpy as np
import smefit.optimize as opt

commondata_path = pathlib.Path(__file__).parents[0] / "fake_data"


# fix the wilson coefficients to 1 and compute the corresponding expected value of the chi2

coeffs_dict = {
    "Op1": {  # fixed to 1
        "constrain": True,
        "value": 1,
        "min": -2,
        "max": 1,
    },
    "Op2": {  # fixed to 1
        "constrain": True,
        "value": 1,
        "min": -2,
        "max": 1,
    },
    "Op3": {  # fixed to 1
        "constrain": True,
        "value": 1,
        "min": -2,
        "max": 1,
    },
    "Op4": {  # fixed to 1
        "constrain": True,
        "value": 1,
        "min": -2,
        "max": 1,
    },
}

# theory predictions for dataset1, assuming all the wilson coefficients are 1
th_pred_1 = (
    np.array(ld.theory_test_1["best_sm"])
    + np.array(ld.theory_test_1["LO"]["Op1"])
    + np.array(ld.theory_test_1["LO"]["Op2"])
    + np.array(ld.theory_test_1["LO"]["Op3"])
    + np.array(ld.theory_test_1["LO"]["Op1*Op1"])
    + np.array(ld.theory_test_1["LO"]["Op2*Op2"])
    + np.array(ld.theory_test_1["LO"]["Op2*Op1"])
    + np.array(ld.theory_test_1["LO"]["Op2*Op3"])
)

# compute the chi2 for dataset1
exp_data_1 = np.array(ld.exp_test_1["data"])
exp_stat_1 = np.array([0.2, 0.3])
exp_sys_1 = np.array([[0.01, 0.01], [0.02, 0.02]])
exp_cov_1 = np.diag(exp_stat_1**2) + exp_sys_1 @ exp_sys_1.T
tot_cov_1 = exp_cov_1 + np.array(ld.theory_test_1["theory_cov"])

chi2_1 = (
    (exp_data_1 - th_pred_1) @ np.linalg.inv(tot_cov_1) @ (exp_data_1 - th_pred_1).T
)

# theory predictions for dataset2, assuming all the wilson coefficients are 1
th_pred_2 = (
    np.array(ld.theory_test_2["best_sm"])
    + np.array(ld.theory_test_2["LO"]["Op1"])
    + np.array(ld.theory_test_2["LO"]["Op2"])
    + np.array(ld.theory_test_2["LO"]["Op4"])
    + np.array(ld.theory_test_2["LO"]["Op1*Op1"])
    + np.array(ld.theory_test_2["LO"]["Op2*Op2"])
    + np.array(ld.theory_test_2["LO"]["Op2*Op1"])
    + np.array(ld.theory_test_2["LO"]["Op2*Op4"])
)

exp_data_2 = np.array(ld.exp_test_2["data"])
exp_stat_2 = np.array([0.2, 0.3, 0.2, 0.3])
exp_sys_2 = np.array([[0.03, 0.03], [0.04, 0.04], [0.05, 0.05], [0.06, 0.06]])
exp_cov2 = np.diag(exp_stat_2**2) + exp_sys_2 @ exp_sys_2.T
tot_cov_2 = exp_cov2 + np.array(ld.theory_test_2["theory_cov"])

# compute the chi2 for dataset2
chi2_2 = (
    (exp_data_2 - th_pred_2) @ np.linalg.inv(tot_cov_2) @ (exp_data_2 - th_pred_2).T
)

# total expected chi2
chi2_tot = chi2_1 + chi2_2

datasets = ["data_test1", "data_test2"]
config = {}
config["data_path"] = commondata_path
config["coefficients"] = coeffs_dict
config["result_path"] = commondata_path
config["result_ID"] = "test"
config["datasets"] = datasets
config["order"] = "LO"
config["use_quad"] = True
config["use_theory_covmat"] = True
config["theory_path"] = commondata_path
config["rot_to_fit_basis"] = None


class TestOptimize:

    test_opt = opt.ns.NSOptimizer.from_dict(config)

    def test_init(self):
        assert self.test_opt.results_path == commondata_path / "test"
        np.testing.assert_equal(self.test_opt.loaded_datasets.ExpNames, datasets)
        np.testing.assert_equal(
            self.test_opt.coefficients.op_name, ["Op1", "Op2", "Op3", "Op4"]
        )

    def test_chi2_func(self):
        np.testing.assert_allclose(self.test_opt.chi2_func(), chi2_tot, rtol=1e-10)
