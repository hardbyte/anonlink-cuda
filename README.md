
An early attempt at running anonlink similarity scores on the GPU.

Currently using `cupy`.




## Things to consider

* Before transfer back to host convert to sparse array and apply threshold.
https://docs-cupy.chainer.org/en/stable/reference/sparse.html#conversion-to-from-cupy-ndarrays

* think about sharing a `__device__` function for `popcount`

* need to validate against anonlink/cpu that it is correct
* profile
* sorting the edges by distance on the gpu.

# Benchmarking

Single CPU using [anonlink](https://github.com/n1analytics/anonlink):

50 M cmp/s

Current speed on a GTX 1080

1.3 B cmp/s (without transferring the data back to the host)

120 M cmp/s including data transfer
