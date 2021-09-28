Companion code to CoRL 2021 paper:  
Vivek Myers, Erdem Bıyık, Nima Anari, Dorsa Sadigh. **"Learning Multimodal Rewards from Rankings"**. *5th Conference on Robot Learning (CoRL)*, London, UK, Nov. 2021.

This code actively learns multimodal reward functions from rankings in various tasks with respect to an information gain acquisition function and compares it to random querying.

The codes for the interface of the user studies are excluded, but the environments can still be simulated with the given trajectory datasets.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [matplotlib](http://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [PyTorch](https://pytorch.org/)
- [SciPy](http://www.scipy.org/)

## Running
You simply run:
```python
	python run.py [task_name]
```
where \[task_name\] is either of the following: lunar, fetch, synthetic.
The output is a PNG file in the main directory that compares the two querying methods.
