#!/usr/bin/env python

# Original from
#  https://raw.githubusercontent.com/ionelmc/python-nameless/purepython/setup.py

# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    # id
    name="irides",
    url="https://gitlab.com/irides-dev/irides",
    # key meta
    version="1.0.1",
    license="Apache License 2.0",
    description="""Irides provides references implementations for select linear time-invariant filters.""",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    # authorship and related
    author="Jay Damask",
    author_email="jaydamask@buell-lane-press.co",
    # project meta
    project_urls={
        "Documentation": "https://gitlab.com/irides-dev/irides/",
        "Changelog": "https://gitlab.com/irides-dev/irides/",
        "Issue Tracker": "https://gitlab.com/irides-dev/irides/issues",
    },
    keywords=["multivariate statistics"],
    classifiers=[
        # ref to https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    # packaging details
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    # requirements and dependencies
    python_requires=">=3.7",
    install_requires=["numpy>=1.23"],
    extras_require={},
)
