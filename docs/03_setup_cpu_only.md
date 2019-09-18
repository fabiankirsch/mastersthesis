
#### Setup environment for executing code with CPU only


This requires no further steps, but to set up a conda environment with the correct packages.

  ```
  conda activate base
  conda create -n kirsch anaconda tensorflow keras
  ipython kernel install --user --name=kirsch
  conda activate kirsch
  ```
