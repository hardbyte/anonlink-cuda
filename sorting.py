import time
import cupy as cp
from filtering import apply_threshold

if __name__ == '__main__':
    a, b = 2**13, 2**10
    # generate our similarity scores as a dense matrix of floats
    data = cp.random.rand(a*b, dtype='float')


    data.get()
    print(f"transfering raw {time.time()-start:.3f}")
    start = time.time()
    s = apply_threshold(data, a, b, threshold=0.99)
    cp.cuda.Stream.null.synchronize()
    print(f"converting to sparse {time.time() - start:.3f}")
    # Transfer sparse matrix back to host
    start = time.time()
    host_result = s.get()