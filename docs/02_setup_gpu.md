
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
