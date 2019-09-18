import os
import json
import re
import sys
import copy

def extract_tags_from_line_string_to_array(line):
    tags = line.partition("[")[2].partition("]")[0]
    tags = tags.replace(' ', '')
    tags = tags.split(',')
    tags = [tag[1:-1] for tag in tags]
    return tags

# cell and notebook templates
cell_template = {
  "cell_type": "code",
  "execution_count": 0,
  "metadata": {},
  "outputs": [],
  "source": []
 }

cell_template_md = {
   "cell_type": "markdown",
   "metadata": {
   },
   "source": []
}

nb_skeleton ={
 "cells": [],
  "metadata": {
  "kernelspec": {
   "display_name":"",
   "language": "python",
   "name": ""
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
# set kernel name
nb_skeleton['metadata']['kernelspec']['name'] = sys.argv[2]

# add cells to notebook
pyscript_path = sys.argv[1]
with open (pyscript_path, "r") as myfile:
    python_script=myfile.readlines()

notebook_cells = [cell_template.copy()]
current_cell_type = 'markdown'
no_block_string = True # i.e. """ some text \n """
for line in python_script:
    if re.search('"""', line):
        no_block_string = not no_block_string

    if re.match(":::", line):
        if re.search('\.markdown', line):
            notebook_cells = notebook_cells +  [copy.deepcopy(cell_template_md)]
        if re.search('\.raw', line):
            notebook_cells = notebook_cells +  [copy.deepcopy(cell_template_md)]
        if re.search('\.code', line):
            notebook_cells = notebook_cells +  [copy.deepcopy(cell_template)]
        if re.search(":::$",line):
            # drop this line by passing
            pass
        if re.search("tags.*]", line):
            tags = extract_tags_from_line_string_to_array(line)
            notebook_cells[-1]['metadata']['tags'] = tags

    elif re.search("```",line):
        if (notebook_cells[-1]['cell_type'] == 'code') & (no_block_string):
        # drop the line by not adding to the source of the cell (only for code cells)
            pass
        else:
            notebook_cells[-1]['source'] = notebook_cells[-1]['source'] + [line]

    else:
        # add line to source of cell. Everything in the source will actually be displayed (markdown and raw) or executed (code)
        notebook_cells[-1]['source'] = notebook_cells[-1]['source'] + [line]

nb_skeleton['cells'] = nb_skeleton['cells'] + notebook_cells

# save notebook
filename_nb = sys.argv[3]
with open(filename_nb, 'w') as nb_file:
    nb_file.write(json.dumps(nb_skeleton, indent=2))
print('Saved notebook to ', filename_nb)
