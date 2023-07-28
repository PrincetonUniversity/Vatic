
import setuptools

setuptools.setup(
    name='Vatic',
    version='0.5.2-rc1',
    description='lightweight PJM power grid simulation engine and interface',

    author='Michal Radoslaw Grzadkowski, Alice Fang',
    author_email='mg2332@princeton.edu, jf3375@princeton.edu',

    packages=setuptools.find_packages(exclude=["vatic.tests"]),
    entry_points={
        'console_scripts': [
            'vatic-det=vatic.command_line:run_deterministic',
            ],
        },

    python_requires='>=3.8,<3.12',
    install_requires=['numpy>1.21', 'pandas<2', 'scipy', 'dill', 'matplotlib',
                      'pyomo', 'gurobipy', 'ordered-set',
                      'gridx-egret @ git+https://github.com/shrivats-pu/Egret.git'],
    )
