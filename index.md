## Implementation of the trust-region limited-memory BFGS quasi-Newton optimization in Deep Learning

The example here is using the classification task of MNIST dataset. 

TensorFlow is used to compute the gradients. Numpy and Scipy is used for the matrix computations. 
### Run the Python program

```shell
$ python LBFGS_TR.py -m=10 -minibatch=1000 -num-batch=4

args:
-m=10             # the L-BFGS memory storage
-num-batch=4 # number of overlapped samples --> refer to the paper 
-minibatch=1000   # minibatch size
-use-whole-data # uses whole data to calculate gradients.
```
