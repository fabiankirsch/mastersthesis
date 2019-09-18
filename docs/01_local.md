


### Locally {#sec:local_setup}
In this section the steps to reproduce the entire master thesis are explained. The intructions are written for Linux and OSX machines using the default terminal commands. Although the same packages might be available for Windows the presented commands will need to be adapted.


#### Base setup

It is helpful to use virtual environments to keep the packages of a machine learning project separate from the operating system. The conda package manager offers virtual environments as well as python package distributions for machine learning and will be used here. Alternatively the pip package manager and python's virtual environments can be used.

Get and install miniconda (Python 3) for your system^[https://conda.io/en/latest/miniconda.html]. Once miniconda is installed open a terminal and install the jupyter package with: `conda install jupyter`. This will install the jupyter package in the base environment. This is needed so we can later register other conda environments as so called ipython kernels, which makes them available to other tools that run independent of the conda environments like the nbconvert command which is used later to run the files.
