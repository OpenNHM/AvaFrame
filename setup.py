from setuptools import setup, Extension
import numpy

from Cython.Build import cythonize

# Define extension modules conditionally
# ext_modules = []

# List all potential Cython file locations
cython_files = [
    'avaframe/com1DFA/DFAfunctionsCython.pyx',
    'avaframe/com1DFA/DFAToolsCython.pyx',
    'avaframe/com1DFA/damCom1DFA.pyx',
]

# Define macros to ensure NumPy 1.x API compatibility
# This allows Cython 3.x to generate code that works with numpy < 2.0
define_macros = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    # Target NumPy 1.x C API even when building with Cython 3.x
    ("NPY_TARGET_VERSION", "NPY_1_22_API_VERSION"),
]

extensions = [
    Extension(
        name=file.replace('/', '.').replace('.pyx', ''),
        sources=[file],
        include_dirs=[numpy.get_include()],
        define_macros=define_macros,  # Add the macros here
    )
    for file in cython_files
]

ext_modules = cythonize(
    extensions,
    compiler_directives={
        'language_level': '3',
        'linetrace': True,
        # Ensure Cython generates code compatible with older NumPy
        'binding': True,
    }
)

setup_options = {"build_ext": {"inplace": True}}

setup(
    options=setup_options,
    ext_modules=ext_modules,
    # install_requires=[
    #     'numpy',
    #     'scipy',
    #     'cython',
    #     'matplotlib',
    #     'pandas'
    # ],
    # python_requires='>=3.8',
)
