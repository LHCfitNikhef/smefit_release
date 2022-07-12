# -*- coding: utf-8 -*-
import pathlib

import packutil as pack
from setuptools import find_packages, setup

# write version on the fly - inspired by numpy
MAJOR = 0
MINOR = 1
MICRO = 0

repo_path = pathlib.Path(__file__).absolute().parent

# TODO: remove this setup.py into pyproject .toml

# TODO: is it necessary to deploy on pypi?


def setup_package():
    # write version
    pack.versions.write_version_py(
        MAJOR,
        MINOR,
        MICRO,
        pack.versions.is_released(repo_path),
        filename="src/smefit/version.py",
    )
    # paste Readme
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    # do it
    setup(
        name="smefit",
        version=pack.versions.mkversion(MAJOR, MINOR, MICRO),
        description="Standard Model Effective Field Theory Fitter",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Tommaso Giani, Jaco Ter Hoeve, Giacomo Magni",
        author_email="tgiani@nikhef.nl",
        # url="https://github.com/LHCfitNikhef/SMEFT",
        package_dir={"": "src/"},
        packages=find_packages("src/"),
        # package_data={"smefit": ["tables/*.yaml"]},
        zip_safe=False,
        classifiers=[
            "Operating System :: Unix",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Physics",
        ],
        install_requires=[
            "rich",
            "matplotlib",
            "pyyaml",
            "numpy",
            "pandas",
            "mpi4py",
            "pymultinest",
            "PyPDF2",
            "scipy",
        ],
        python_requires=">=3.7",
    )


if __name__ == "__main__":
    setup_package()
