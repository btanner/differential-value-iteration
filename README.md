# differential-value-iteration
Experiments with new and existing average-reward planning algorithms.

# Prerequisites
- Install JAX https://github.com/google/jax#installation

Possibly just:
```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```
# Installation
- Clone the github repository to a local directory.

## Developer
- From the root of the repo, `python setup.py develop --user`

This will install into your user folder.

To check if it is working, you should be able to execute an experiment.

Eg. `python main.py src/differential_value_iteration/experiments/dvi_async_vs_sync.py'

You can uninstall:`python setup.py develop --uninstall`

To run the tests, from the root of the repo (after running setup.py):

`python -m unittest discover src/differential_value_iteration -p '*_test.py'`

## User
- From the root of the repo, `python setup.py install`



