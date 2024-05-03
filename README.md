### Notice
Forked from https://github.com/hichenway/sampling-based-path-planning

I have modified the original code, to make it work with python 3.

It is tested OK on python 3.9.6.

```shell
# Execute these cmds in project root dir after your python env is ready: 

pip install -r requirements.txt

export PYTHONPATH=$(PWD)
python ./RRT_and_Pruning/main.py

```

Below is the original readme:

### sampling-based-path-planning

**RRT and Pruning**

RRT is a path planning algorithm based on random sampling. It can search the whole state space quickly and is widely used to the high-dimensional problems.

Pruning is a simple but efficient algorithm thought.It's intention is to avoid unnecessary search and opearation, or to clip unnecessary parts in result to gain better effect.



**Batch Informed Trees**

Python Implement For Batch-Informed-Trees



Additionally, this code requires the installation of several python libraries, namely:

	- shapely
	- numpy
	- yaml
	- matplotlib
	- descartes