import pathlib

import tests.test_loader as ld
import numpy as np
import scipy.linalg as la
import smefit.optimize as opt

commondata_path = pathlib.Path(__file__).parents[0] / "fake_data"


# fix the wilson coefficients to some random value and compute the corresponding expected chi2

coeffs_dict = {
    "Op1": {
        "min": -2,
        "max": 1,
    },
    "Op2": {
        "min": -2,
        "max": 1,
    },
    "Op3": {
        "min": -2,
        "max": 1,
    },
    "Op4": {
        "constrain": True,
        "value": 0.0,
        "min": -2,
        "max": 1,
    },
}

random_point = np.random.rand(3)
prior = random_point * (
    np.array([1.0, 1.0, 1.0]) - np.array([-2.0, -2.0, -2.0])
) + np.array([-2.0, -2.0, -2.0])

# generate random values for the wilson coefficients
wilson_coeff = np.random.rand(3)

### first consider the case of dataset 1 and dataset 2, which do not have common systemtics

# Theory predictions for dataset1
th_pred_1 = (
    np.array(ld.theory_test_1["best_sm"])
    + np.array(ld.theory_test_1["LO"]["Op1"]) * wilson_coeff[0]
    + np.array(ld.theory_test_1["LO"]["Op2"]) * wilson_coeff[1]
    + np.array(ld.theory_test_1["LO"]["Op3"]) * wilson_coeff[2]
    + np.array(ld.theory_test_1["LO"]["Op1*Op1"]) * wilson_coeff[0] ** 2
    + np.array(ld.theory_test_1["LO"]["Op2*Op2"]) * wilson_coeff[1] ** 2
    + np.array(ld.theory_test_1["LO"]["Op2*Op1"]) * wilson_coeff[1] * wilson_coeff[0]
    + np.array(ld.theory_test_1["LO"]["Op2*Op3"]) * wilson_coeff[1] * wilson_coeff[2]
)

# chi2 for dataset1
exp_data_1 = np.array(ld.exp_test_1["data"])
exp_stat_1 = np.array([0.2, 0.3])
exp_sys_1 = np.array([[0.01, 0.01], [0.02, 0.02]])
exp_cov_1 = np.diag(exp_stat_1**2) + exp_sys_1 @ exp_sys_1.T
tot_cov_1 = exp_cov_1 + np.array(ld.theory_test_1["theory_cov"])

chi2_1 = (
    (exp_data_1 - th_pred_1) @ np.linalg.inv(tot_cov_1) @ (exp_data_1 - th_pred_1).T
)

# theory predictions for dataset2
th_pred_2 = (
    np.array(ld.theory_test_2["best_sm"])
    + np.array(ld.theory_test_2["LO"]["Op1"]) * wilson_coeff[0]
    + np.array(ld.theory_test_2["LO"]["Op2"]) * wilson_coeff[1]
    + np.array(ld.theory_test_2["LO"]["Op1*Op1"]) * wilson_coeff[0] ** 2
    + np.array(ld.theory_test_2["LO"]["Op2*Op2"]) * wilson_coeff[1] ** 2
    + np.array(ld.theory_test_2["LO"]["Op2*Op1"]) * wilson_coeff[1] * wilson_coeff[0]
)

exp_data_2 = np.array(ld.exp_test_2["data"])
exp_stat_2 = np.array([0.2, 0.3, 0.2, 0.3])
exp_sys_2 = np.array([[0.03, 0.03], [0.04, 0.04], [0.05, 0.05], [0.06, 0.06]])
exp_cov2 = np.diag(exp_stat_2**2) + exp_sys_2 @ exp_sys_2.T
tot_cov_2 = exp_cov2 + np.array(ld.theory_test_2["theory_cov"])

# chi2 for dataset2
chi2_2 = (
    (exp_data_2 - th_pred_2) @ np.linalg.inv(tot_cov_2) @ (exp_data_2 - th_pred_2).T
)

# total expected chi2. Since there are no correlations between the two datasets I can sum the
# two independent contributions
chi2_tot = chi2_1 + chi2_2

datasets_no_corr = ["data_test1", "data_test2"]
config_no_corr = {}
config_no_corr["data_path"] = commondata_path
config_no_corr["coefficients"] = coeffs_dict
config_no_corr["result_path"] = commondata_path
config_no_corr["result_ID"] = "test"
config_no_corr["datasets"] = datasets_no_corr
config_no_corr["order"] = "LO"
config_no_corr["use_quad"] = True
config_no_corr["use_theory_covmat"] = True
config_no_corr["theory_path"] = commondata_path
config_no_corr["rot_to_fit_basis"] = None


### now consider the case of dataset 3 and dataset 4 having a common systematic named SPECIAL
# The theory predictions for the dataset 3 are the same as for dataset 1

th_pred_3 = th_pred_1

# while dataset 4 we have

theory_test_4 = {
    "best_sm": [1.1, 2.3, 3.0],
    "theory_cov": [
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3],
    ],
    "LO": {
        "Op1": [1.1, 1.2, 1.3],
        "Op4": [2.5, 1.6, 1.7],
        "Op2": [1.3, 4.4, 4.5],
        "Op1*Op1": [1.7, 1.8, 1.9],
        "Op2*Op2": [1.9, 1.11, 2.1],
        "Op2*Op1": [3.12, 2.13, 3.1],
        "Op2*Op4": [9.14, 5.15, 5.16],
    },
    "NLO": {
        "Op1": [2.1, 2.2, 2.3],
        "Op4": [3.5, 3.6, 2.7],
        "Op2": [4.3, 4.4, 5.5],
        "Op1*Op1": [2.7, 1.8, 2.9],
        "Op2*Op2": [9.9, 5.11, 6.16],
        "Op2*Op1": [1.12, 2.13, 6.18],
        "Op2*Op4": [7.14, 4.15, 4.17],
    },
}

th_pred_4 = (
    np.array(theory_test_4["best_sm"])
    + np.array(theory_test_4["LO"]["Op1"]) * wilson_coeff[0]
    + np.array(theory_test_4["LO"]["Op2"]) * wilson_coeff[1]
    + np.array(theory_test_4["LO"]["Op1*Op1"]) * wilson_coeff[0] ** 2
    + np.array(theory_test_4["LO"]["Op2*Op2"]) * wilson_coeff[1] ** 2
    + np.array(theory_test_4["LO"]["Op2*Op1"]) * wilson_coeff[1] * wilson_coeff[0]
)


# This time the total covmat is not block diagonal, and it is given by

tot_cov_corr = [
    [1.53, 0.53, 0.02, 0.02, 0.02],
    [0.53, 1.53, 0.02, 0.02, 0.02],
    [0.02, 0.02, 1.38, 0.34, 0.34],
    [0.02, 0.02, 0.34, 1.38, 0.34],
    [0.02, 0.02, 0.34, 0.34, 1.38],
]


# The experimental central values are

exp_data_3 = np.array([1.0, 2.0])
exp_data_4 = np.array([1.0, 2.0, 3.0])


diff = np.concatenate([[exp_data_3 - th_pred_3], [exp_data_4 - th_pred_4]], axis=1)
chi2_corr = diff @ np.linalg.inv(tot_cov_corr) @ diff.T

datasets_corr = ["data_test3", "data_test4"]
config_corr = {}
config_corr["data_path"] = commondata_path
config_corr["coefficients"] = coeffs_dict
config_corr["result_path"] = commondata_path
config_corr["result_ID"] = "test"
config_corr["datasets"] = datasets_corr
config_corr["order"] = "LO"
config_corr["use_quad"] = True
config_corr["use_theory_covmat"] = True
config_corr["theory_path"] = commondata_path
config_corr["rot_to_fit_basis"] = None


class TestOptimize:

    test_opt = opt.ns.NSOptimizer.from_dict(config_no_corr)
    test_opt_corr = opt.ns.NSOptimizer.from_dict(config_corr)

    def test_init(self):
        assert self.test_opt.results_path == commondata_path / "test"
        np.testing.assert_equal(
            self.test_opt.loaded_datasets.ExpNames, datasets_no_corr
        )
        np.testing.assert_equal(
            self.test_opt.coefficients.name, ["Op1", "Op2", "Op3", "Op4"]
        )

    def test_free_parameters(self):
        np.testing.assert_equal(
            list(self.test_opt.free_parameters.index), ["Op1", "Op2", "Op3"]
        )

    def test_chi2_func_ns(self):
        # set free parameters to random values generated above
        params = wilson_coeff
        np.testing.assert_allclose(
            self.test_opt.chi2_func_ns(params), chi2_tot, rtol=1e-10
        )
        np.testing.assert_allclose(
            self.test_opt_corr.chi2_func_ns(params), chi2_corr, rtol=1e-10
        )

    def test_gaussian_loglikelihood(self):
        params = wilson_coeff
        np.testing.assert_allclose(
            self.test_opt.gaussian_loglikelihood(params), -0.5 * chi2_tot, rtol=1e-10
        )
        np.testing.assert_allclose(
            self.test_opt_corr.gaussian_loglikelihood(params),
            -0.5 * chi2_corr,
            rtol=1e-10,
        )

    def test_flat_prior(self):
        np.testing.assert_equal(self.test_opt.flat_prior(random_point), prior)
