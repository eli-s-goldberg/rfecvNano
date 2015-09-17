# The rfecvNano Model
The development of advection–dispersion particle transport models (PTM) for transport of nanoparticles in porous media has focused on improving model fit by inclusion of empirical parameters. However, this has done little to disentangle the complex behavior of nanoparticles in porous media and to provide mechanistic insights into nanoparticle transport. The most prominent limitation of current PTMs is that they do not consider the influence of physicochemical conditions of the experiments on the transport of nanomaterials. Here, we overcome this limitation by bypassing traditional advection–dispersion PTMs and relating the physicochemical conditions of the experiments to the experimental outcome using ensemble machine-learning methods. We identify a small set of factors that seem to determine the transport of nanoparticles in column experiments by recursive feature elimination (RFECV) with cross validation.

## How to use the model
I provide an example to follow. It should be rather straightforward. That said, don't forget to match the developmental state (pip freeze > requirements.txt for those of you who want to do this). Also, ignore all of the basic site packages (I am using pycharm as an IDE so there's a bunch of default junk). 

## Useful Functions: helperFunctions.py
This file contains a bunch of useful functions that will enable you to perform the RFECV task with few problems. Other functions specifically included for the tasks of predicting nanomaterial transport (via the examples) are also provided. These functions (e.g., absViscTempWater, debyeLength, pecletNumber, etc), are incorporated into the examples. 

### Class redefinitions
There are several points where the class definitions are modified to suit our needs. Some of these are my own doing, and some are the result of the community. I will try to include references/links to the community-based class modifications, if possible. However, I am human and forget. If you find something, please let me know by email and I would be happy to include a link. 


