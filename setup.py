import setuptools

setuptools.setup(
    name='Vatic',
    version='0.1.0-a4',
    description='Pyomo + Egret',
    author='Michal Radoslaw Grzadkowski',
    author_email='mg2332@princeton.edu',
    python_requires='==3.8.*',

    packages=setuptools.find_packages(exclude=["vatic.tests"]),

    entry_points = {
        'console_scripts': [
            'vatic-det=vatic.command_line:run_deterministic',
            ],
        },

    install_requires = [
        'numpy<1.21',
        'gridx-prescient @ git+https://github.com/shrivats-pu/Prescient.git@vatic'
        ],
    )
