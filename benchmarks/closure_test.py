import subprocess

smefit = "/project/theorie/jthoeve/miniconda3/envs/smefit/bin/smefit"
projection_runcard = '/data/theorie/jthoeve/smefit_release/benchmarks/closure_hl_only.yaml'
fit_runcard = '/data/theorie/jthoeve/smefit_release/benchmarks/A_smefit_hllhc_only_glob_NLO_NHO.yaml'

n_exp = 5000
for i in range(n_exp):
    # run projection module
    subprocess.run([smefit, 'PROJ', '--closure', projection_runcard])

    # run analytic module without samples
    subprocess.run([smefit, 'A', fit_runcard])
