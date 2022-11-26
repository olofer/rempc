from setuptools import Extension, setup
import numpy as np

setup(
    name = 'rempc-package',
    version = '0.0',
    description = 'Python interface to multi-stage optimization solver',
    ext_modules = [
        Extension(
            name = 'rempc',  # can have . in it
            sources = ['rempc-basic.c'], 
            include_dirs = [np.get_include()],
            define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            extra_compile_args = [],
            extra_link_args = [],
        ),
    ]
)
