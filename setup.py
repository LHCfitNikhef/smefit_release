# -*- coding: utf-8 -*-
import pathlib
from setuptools import setup, find_packages

import packutil as pack

# write version on the fly - inspired by numpy
MAJOR = 0
MINOR = 1
MICRO = 0

repo_path = pathlib.Path(__file__).absolute().parent


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
    with open("README.md", "r") as fh:
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
        #url="https://github.com/LHCfitNikhef/SMEFT",
        package_dir={"": "src/"},
        packages=find_packages("src/"),
        #package_data={"smefit": ["tables/*.yaml"]},
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
        ],
        python_requires=">=3.7",
    )


if __name__ == "__main__":
    setup_package()