
import setuptools


setuptools.setup(
    name='Vatic',
    version='0.4.0-a1',
    description='lightweight PJM power grid interface for Egret + Pyomo',

    author='Michal Radoslaw Grzadkowski',
    author_email='mg2332@princeton.edu',

    packages=setuptools.find_packages(exclude=["vatic.tests"]),
    entry_points={
        'console_scripts': [
            'vatic-det=vatic.command_line:run_deterministic',
            ],
        },

    python_requires='>=3.8,<3.11',
    install_requires=[
        'numpy>1.21', 'pandas', 'scipy', 'dill', 'matplotlib',
        'pyomo>=6', 'gurobipy',
        'gridx-egret @ git+https://github.com/shrivats-pu/Egret.git'
        ],
    )
