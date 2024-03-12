import subprocess
import smefit

projection_runcard = '/Users/jaco/Documents/smefit_release/runcards/projections/closure_baseline.yaml'
fit_runcard = '/Users/jaco/Documents/smefit_release/runcards/A_smefit_baseline_glob_NLO_NHO.yaml'

n_exp = 1
for i in range(n_exp):
    # run projection module
    subprocess.run(['smefit', 'PROJ', '--closure', projection_runcard])

    # run analytic module without samples
    subprocess.run(['smefit', 'A', fit_runcard])
