
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
