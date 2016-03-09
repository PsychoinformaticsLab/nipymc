import sys
from setuptools import setup

__version__ = '0.0.1'

if len(set(('test', 'easy_install')).intersection(sys.argv)) > 0:
    import setuptools

tests_require = ['dill', 'pygraphviz']

extra_setuptools_args = {}
if 'setuptools' in sys.modules:
    tests_require.append('nose')
    extra_setuptools_args = dict(
        test_suite='nose.collector',
        extras_require=dict(
            test='nose>=0.10.1')
    )

setup(
    name="pymcwrap",
    version=__version__,
    description="...",
    author='Tal Yarkoni',
    author_email='tyarkoni@gmail.com',
    packages=["pymcwrap"],
    install_requires=['six'],
    tests_require=tests_require,
    license='MIT',
    **extra_setuptools_args
)
