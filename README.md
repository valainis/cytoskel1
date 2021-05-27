Cytoskel: trajectory inference and analysis
-----------------------------------------

#### Introduction

Cytoskel is a set of algorithms for trajectory inference  and analysis
in high dimensional single cell data. More generally, it can be used
to infer and analyze trajectories in any N by p data set where N is
the number of samples and p is the number of features per sample.


#### Installation

Cytoskel requires a Python 3 installation. To use Cytoskel from R
requires R >= 3.0 (possibly >= 4.0) and the reticulate library.

Cytoskel has been tested with Python 3.6 and 3.7 as found on macOS and
Ubuntu. However, we recommend setting up a Python environment using
the package manager conda and then installing Cytoskel to the conda
environment. If conda has been installed via the Anaconda installer,
it is strongly recommended not to install to the base environment, but
to create a new conda environment. In terminal, in any directory:

> conda  create -name *name_of_env* python=3.7

3.7 is python version , Cytoskel tested  with 3.6, 3.7, 3.8.
Any of those should be ok.

To activate the environment do:

> conda activate *name_of_env*

After cloning Cytoskel or downloading the zip of the Cytoskel 
directory, open a terminal and go to the top Cytoskel directory which 
contains the setup.py file. There do: 

> pip install  .

If there are no errors you can go  to the test folder and run the scripts.
For python script, for example do

> python tgen.py

It just re-does the cytoskel calculation and compares the result
to existing project. It should print out true twice. For the corresponding R script do

> Rscript tgen.R

or run in Rstudio. The R scripts begin with:

```R
library(reticulate)
library(data.table)
csk1 = import("cytoskel1")

It turns out that reticulate finds the python version on the system
which has the first python module to be imported. In the above code
this is the python version which has cytosskel1 module installed.

To deactivate the python enviroment and go back to the old one just
do
> conda deactivate
and activate again if you want to run Cytoskel again

####
Documentation discussing cytoskel algorithms is in directory docs

notebooks contains cytoskel examples in jupyter notebooks.

The test directory contains python and R scripts as examples of
cytoskel runs. The tgen scripts run basic cytoskel. The tlink scripts
do data with time points.




