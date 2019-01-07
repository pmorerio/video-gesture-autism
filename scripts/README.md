### Running the code
In the root directory you can use the following command to run the main script file:
```
THEANO_FLAGS='floatX=float32,device=gpu0,mode=FAST_RUN,nvcc.fastmath=True' python -m scripts.evaluate_crossval
```

### Visualizations
The file `draw-visualizations.ipynb` is a sample IPython notebook for drawing visualization of attention.
