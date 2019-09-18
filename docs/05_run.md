
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
