# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Build Cython fast-path extensions for MoE-Sched.

Usage:
    python setup_cython.py build_ext --inplace

This compiles the .pyx files in moe_sched/runtime/_fast/ into C extensions
that are automatically used by the compiler when FAST_PATH_AVAILABLE is True.
"""

from setuptools import setup
from Cython.Build import cythonize

extensions = cythonize(
    [
        "moe_sched/runtime/_fast/_cache.pyx",
        "moe_sched/runtime/_fast/_scheduler.pyx",
        "moe_sched/runtime/_fast/_hooks.pyx",
    ],
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "language_level": "3",
    },
)

setup(
    name="moe_sched_fast",
    ext_modules=extensions,
    packages=[],
)
