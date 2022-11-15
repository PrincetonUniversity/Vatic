# Vatic #

Vatic is a Python package for running simulations of a power grid using the
[PJM](https://www.e-education.psu.edu/ebf483/node/814) framework consisting of alternating day-ahead unit commitment
(UC) and real-time economic dispatch (ED) steps. Vatic was originally designed as a lightweight adaptation of
[Prescient](https://github.com/grid-parity-exchange/Prescient); it likewise applies mixed-integer linear programming
optimization as implemented in [Pyomo](http://www.pyomo.org/) to power grid formulations created using
[Egret](https://github.com/grid-parity-exchange/Egret).


## Installing Vatic ##

After making sure you have Python 3.8 installed, clone the repository using one of the following from command line:

```git clone https://github.com/PrincetonUniversity/Vatic.git```

```git clone git@github.com:PrincetonUniversity/Vatic.git```

Then, from inside the cloned directory, install Vatic:
```
cd Vatic
pip install .
```

The Vatic installation includes the [May 2021 version of the Texas-7k grid dataset](
https://electricgrids.engr.tamu.edu/electric-grid-test-cases/datasets-for-arpa-e-perform-program/)
produced by Adam Birchfield et al. at Texas A&M, as well as the [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) grid.


## Running Vatic ##

Installing Vatic adds the command `vatic-det` to your command line namespace. The simplest way to invoke this command
is:

```vatic-det $input_grid $start_date $num_days ```

`input_grid` can be either of the currently supported grids: "RTS-GMLC" or "Texas-7k".

`start_date` is the first day that will be simulated by Vatic, given in YYYY-MM-DD format. For RTS-GMLC, only dates in
2020 are supported, while the Texas grids only support dates in 2018.

`num_days` is the number of days to simulate from the starting date and must therefore be given as a positive integer.

`vatic-det` also supports the following optional arguments further controlling its behaviour:

 - `--out-dir (-o)` Where the output will be stored; if not specified it will be saved to the location `vatic-det`
                    was invoked from.

 - `--solver` The solver to use for RUC and SCED instances, such as `cbc` or `gurobi`. The default is
              [cbc](https://github.com/coin-or/Cbc); note that you may have to install your preferred solver separately.

 - `--lmps` If this flag is given, Vatic will calculate bus-specific locational marginal prices at each real-time SCED.
            Note that this tends to increase SCED runtime by roughly 25%.

 - `--threads (-t)` The number of compute cores to be used for parallelization. If you are running `vatic-det` on a
                    remote compute cluster, do not use more cores than what has been allocated to you for a particular
                    job. The default value is 1, which will not parallelize any computation. Must be a non-negative
                    integer; a value of 0 will instruct the solvers to use all possible nodes, which is not recommended
                    when running on remote clusters.

 - `--sced-horizon` How far ahead in hours each security-constrained economic dispatch instance will look ahead.
                    Must be a positive integer; the default value is 4.

 - `--ruc-horizon` How many hours each reliability unit commitment will consider in its optimization. Must be a positive
                   integer; the default value is 48 to avoid horizon effects when planning towards the end of the
                   current day.

 - `--ruc-mipgap (-g)` The relative optimality gap used by each reliability unit commitment instance to decide when to
                       terminate. Expressed as a ratio of the difference between the lower and upper objective bound and
                       the incumbent objective value. The default value is 0.01.

 - `--reserve-factor (-r)` How much headroom or spare capacity must the system plan for at each operating time step
                           given as a proportion of the total load demand at a time step; the default value is 0.05.

 - `--csv` Save output to a collection of .csv files instead of a serialized Python pickle.

 - `--init-ruc-file` If this file exists, it will be treated as a saved reliability unit commitment from a previous
                     iteration of Vatic that used the same grid and starting date. If it doesn't exist, Vatic will save
                     the RUC from this run to the file for future use. The cached RUC file takes the form of a `.p`
                     pickled Python object that is in the ballpark of `600K` and `30M` in size for the RTS-GMLC and the
                     Texas grids respectively.

 - `--solver-args` A list of arguments to modify the behaviour of the solver used for RUCs and SCEDs. These should be
                   given in the format ```--solver-args arg1=x arg2=y arg3=z ...```. For example, if you are using
                   Gurobi, you might specify ```--solver-args Cuts=1 Presolve=1 Heuristics=0.03```.

 - `--light-output (-l)` If this flag is used, Vatic will avoid saving data for individual transmission line and load
                         demand bus behaviour at each time step. This is useful when running Vatic as part of
                         large-scale experiments where storing this level of detail would result in a cumbersome amount
                         of saved data.
