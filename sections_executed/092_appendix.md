
## B - reproducing the thesis


### Google Colaboratory - no local setup required (recommended) {#sec:colab}
Google provides online resources for interactively running python code in jupyter notebooks. They also provide GPU processing units, which allow for high processing speeds. A demo of Google Colaboratory can be found here^[https://colab.research.google.com (2019-06-03)]. The following steps describe how to run parts of this thesis on Google Colaboratory without the need for setting up a local machine.

* Sign in to google.com (create a free account if needed)
* Go to drive.google.com -> Click on 'New' -> 'More'
* Check if 'Colaboratory' is in the list, if not click on 'Connect more apps' and add 'Colaboratory'
* Upload the .ipynb file that should be reproduced from the 'accompanying_digital_storage:Appendix A/sections_executed' directory to google drive
* just double-clicking the uploaded ipynb file on google drive should open it in Google Colaboratory
* Go to "Runtime" (top menu bar) -> "Change runtime type" -> Set "Hardware accelerator" to "GPU" and "Save"
* Open the sidebar by clicking on the arrow on the left -> Go to "Files" and "Upload"
* Upload all files from the 'accompanying_digital_storage:Appendix A/code' directory. These are the custom python modules and necessay for execution of the code.
* Click on the first cell and start executing cells one by one by hitting "Shift+Enter".
* Note: some figures might not be presented correctly for two reasons: 1) the figures are not uploaded, 2) the code produces only references to figures, and the figures are only integrated in a later processing step, which is not performed here



### Locally {#sec:local_setup}
In this section the steps to reproduce the entire master thesis are explained. The intructions are written for Linux and OSX machines using the default terminal commands. Although the same packages might be available for Windows the presented commands will need to be adapted.


#### Base setup

It is helpful to use virtual environments to keep the packages of a machine learning project separate from the operating system. The conda package manager offers virtual environments as well as python package distributions for machine learning and will be used here. Alternatively the pip package manager and python's virtual environments can be used.

Get and install miniconda (Python 3) for your system^[https://conda.io/en/latest/miniconda.html]. Once miniconda is installed open a terminal and install the jupyter package with: `conda install jupyter`. This will install the jupyter package in the base environment. This is needed so we can later register other conda environments as so called ipython kernels, which makes them available to other tools that run independent of the conda environments like the nbconvert command which is used later to run the files.

#### Setup environment for executing code with GPU and CPU

This section contains an examplary setup process of GPU drivers, CUDA and cudnn on an ubuntu machine. The process likely looks different for other systems.

System specifications:

* Ubuntu 18.04
* Kernel: 4.15.0
* GPU: (Nvidia) Asus GeForce GTX 970, 4GB
* CPU: AMD FX 3850


##### Installing GPU drivers
The drivers can also be installed during installation of the CUDA toolkit, however this did not work for me as the nouveau drivers could not be deactivate during that process. So I installed the GPU drivers first manually and the toolkit afterwards.
```
  sudo apt install openjdk-8-jdk

  sudo add-apt-repository ppa:graphics-drivers/ppa
  sudo apt update
  sudo apt upgrade

  # remove everything nvidia related
  sudo apt purge nvidia*

  #install new drivers
  ubuntu-drivers devices
  sudo ubuntu-drivers autoinstall

  # instead of the autoinstall one can also do for example
  sudo apt-get install nvidia-390

  sudo reboot
  ```

* check if drivers are properly installed with: `nvidia-smi`
* check that **no** nouveau driver is loaded with (this should not give any output): `lsmod | grep nouveau`

##### Install CUDA toolkit
* check which gcc/g++ version and CUDA version are needed for your kernel: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
* install gcc/g++ if needed, check version with `gcc --version` and `g++ --version`
* follow installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#introduction
* don't install graphic drivers during setup, only toolkit!

    ```
    sudo sh cuda_XXX_linux.run --toolkit
    sudo reboot
    ```

##### Install cudnn
* Follow: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

##### Set up conda environment
* Create, register (needed for later) and activate environement:

    ```
    conda activate base
    conda create -n kirsch_gpu anaconda tensorflow-gpu keras
    ipython kernel install --user --name=kirsch_gpu
    conda activate kirsch_gpu
    ```

* Then check in python if tensorflow recognizes the GPU. The output should not be an empty string, but something with '/device':

    ```python
    import tensorflow as tf
    tf.test.gpu_device_name()
    ```

#### Setup environment for executing code with CPU only


This requires no further steps, but to set up a conda environment with the correct packages.

  ```
  conda activate base
  conda create -n kirsch anaconda tensorflow keras
  ipython kernel install --user --name=kirsch
  conda activate kirsch
  ```

#### Setup environment for automating thesis production

To automate the execution of the machine learning pipeline and creation of a PDF output the following packages need to be installed.

  ```
  # A system wide tex distribution
  sudo apt-get install texlive

  # An environment for pandoc
  conda create -n kirsch_pandoc
  conda activate kirsch_pandoc
  conda install jupyter
  conda install -c conda-forge pandoc=2.7
  conda install -c conda-forge pandoc-crossref
  ```

#### Execution {#sec:execution}

Copy the entire `accompanying_digital_storage:/Appendix A` directory to the machine local drive. Change directory to the `Appendix A` folder. Everything will be executed from here. All following paths are relative to this location. The scripts in the `scripts/` directory are used to wrap some commands for running the code and transforming the output into a beautiful PDF. Each `*.md` file in the `sections_in_progress` directory needs to be executed separately to produce the final PDF. The order of which these files are executed is not important.

  ```
  # Activate the pandoc conda environment first
  conda activate kirsch_pandoc
  ```


##### Conversion from markdown (md) to juypter notebooks (ipynb)

The files in `sections_in_progress` are markdown files with python code snippets.
Lines like `::: {.cell .markdown}` denote what kind of cell the following part in the file will become in the jupyter notebook (markdown or code cell). The conversion is done by the `scripts/convert_md_to_ipynb.py` python script with the bash wrapper script `scripts/b1_convert_md_to_ipynb` on top. This script expects both the filename and the jupyter kernel / conda environment. Run it likes this:

  ```
  ./scripts/b1_convert_md_to_ipynb sections_in_progress/ \
      100_working_002_intro_theory.md \
      kirsch_gpu    
  ```


##### Execution of the ipynb file

Next, the generated ipynb file will run headlessly using the `scripts/b2_exec_ipynb_headless` script. This spins up a notebook server and, executes the notebook and stops the server again using `nbconvert`. The executed notebook file is stored `sections_executed`. This script expects only the name of the file. Run it like this:

  ```
  ./scripts/b2_exec_ipynb_headless \
    sections_in_progress/100_working_002_intro_theory.md
  ```

##### Conversion of executed ipynb back to markdown

The executed ipynb is then converted back to markdown using nbconvert again with some manual fixes of the output. The markdown output file is stored in `sections_executed`. Run it like this:

  ```
  ./scripts/b3_convert_body_iipynb-execd_to_md \
    sections_in_progress/100_working_002_intro_theory.md
  ```

##### Concatenation and formatting
Finally, all the markdown files in `sections_executed` are concatenated and then transformed using `pandoc` and `latex` engines to a properly formatted PDF file. The title, author, table of contents and the bibliography are also added in this step. Run it like this:

  ```
  ./scripts/c0_make_pdf
  ```

##### Wrapper
The scripts b1, b2, b3 and c0 and are wrapped by `scripts/a0_run_all`, which can be run like this:

  ```
  ./scripts/a0_run_all \
    sections_in_progress/100_working_002_intro_theory.md \
    kirsch_cpu
  ```

\cleardoublepage

## C - Running unit tests {#sec:unittests}

To run the unit tests activate the conda environment containing the anaconda distribution (this should be either *kirsch* or *kirsch_gpu* following the instructions in @sec:local_setup) with `conda activate kirsch_gpu` in a terminal. Change directory to *accompanying_digital_storage:/Appendix A/code* and run `pytest` in the terminal. This will automatically execute all functions starting with 'test' in all python files which names start with 'test'.  
