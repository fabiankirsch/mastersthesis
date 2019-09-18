
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
