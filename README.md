
An early attempt at running anonlink similarity scores on the GPU using `cupy`.


## Running 

To run you will need cuda v8.0 or later. Install deps with:

    python -m pipenv shell

Run the main test with:

    python cudadice.py

### Running with the profiler


`nvprof -f -o pyprof.nvprof python cudadice.py`

Then open pyprof.nvprof in NVIDIA Visual Profiler 



## Things to consider


* need to validate against anonlink/cpu that it is correct

* merge sorting the returned edges on the CPU while the GPU is busy

* streaming


# Benchmarking

Single CPU using [anonlink](https://github.com/n1analytics/anonlink):

50 M cmp/s

Current speed on a GTX 1080:

1.3 B cmp/s including data transfer
