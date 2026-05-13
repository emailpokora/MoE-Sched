# Copyright (c) 2026 Jesse Pokora — MIT License (see LICENSE)
"""Build configuration for Cython extensions (Phase 2).

Usage:
    python setup.py build_ext --inplace

This compiles the Cython fast-path modules in moe_sched/runtime/_fast/
into native extension modules.

Status: STUB — activate after Phase 2 .pyx files are implemented.
"""

from setuptools import setup, find_packages

from Cython.Build import cythonize

ext_modules = cythonize([
    "moe_sched/runtime/_fast/_cache.pyx",
], compiler_directives={
    "boundscheck": False,
    "wraparound": False,
    "language_level": "3",
})

setup(
    name="moe-sched",
    version="1.0.0",
    author="Jesse Pokora",
    description="A domain-specific language for Mixture-of-Experts scheduling policies",
    license="MIT",
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.10",
    install_requires=[
        "lark>=1.1,<2",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pyyaml"],
        "gpu": [
            "torch>=2.0",
            "transformers>=5.0.0",
            "accelerate",
            "requests>=2.33.0",
            "urllib3>=2.6.3",
            "filelock>=3.20.3",
        ],
        "cython": ["cython>=3.0"],
        "eval": ["matplotlib>=3.7", "pyyaml", "pandas", "pillow>=12.2.0"],
    },
)
