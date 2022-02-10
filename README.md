# <div allign='center'>Background estimation through reweighting<div> 

## Where to run?

### On lxplus

Run first the following:

```
source Setup.sh
```

### On ml.cern.ch

Use the following image: gitlab-registry.cern.ch/ai-ml/kubeflow_images/pytorch-notebook-gpu-1.8.1:v0.6.1-30

## Create input H5 files

Use the ```CreateH5files.py``` script to produce H5 files with the information needed for the training.

**How to run?**

Edit the ```datasets``` dictionary if needed before running.

```
python CreateH5files.py
```

### How to check the H5 file(s)?

You can use the ```CHeckH5file.py``` script to make sure a given H5 file make sense (content is right).

You can also use it as a reference on how to open a H5 file and get all the data.


### Make plots from a H5 file

You can use ```MakePlots.py``` to make some plots from a given H5 file (currently only making a single plot comparing HT between events w/ and w/o quark-jets).
