import setuptools

setuptools.setup(
    name='Vatic',
    version='0.1-a0',
    description='Pyomo + Egret',
    author='Michal Radoslaw Grzadkowski',
    author_email='mg2332@princeton.edu',

    entry_points = {
        'console_scripts': [
            'vatic-det=vatic.command_line:run_deterministic',
            ],
        },
    )
