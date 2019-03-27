
An early attempt at running anonlink similarity scores on the GPU.

Currently using `cupy`.




## Things to consider

* Before transfer back to host convert to sparse array and apply threshold.
https://docs-cupy.chainer.org/en/stable/reference/sparse.html#conversion-to-from-cupy-ndarrays

* think about sharing a `__device__` function for `popcount`

* need to validate against anonlink/cpu that it is correct
* profile

# Benchmarking

50 M cmp/s/cpucore

