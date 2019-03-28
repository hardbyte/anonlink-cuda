import cupy as cp
from cupyx.scipy.sparse import coo_matrix

def apply_threshold(data, size_a, size_b, threshold=0.75):
    data = data.ravel()

    # 2d indices into our 1d array
    s = coo_matrix((size_a, size_b), dtype=cp.float32)
    col_index = cp.tile(cp.arange(size_b), size_a)
    row_index = cp.arange(size_a).repeat(size_b)

    # apply threshold
    mask = data > threshold
    masked_data = data[mask]
    if len(masked_data) > 0:
        masked_index_a = col_index[mask]
        masked_index_b = row_index[mask]

        # Convert to sparse matrix (on device)
        s = coo_matrix((masked_data, (masked_index_b, masked_index_a)))

    return s


if __name__ == '__main__':
    a, b = 2**13, 2**10
    # generate our similarity scores as a dense matrix of floats
    data = cp.random.rand(a*b, dtype='float')

    import time

    start = time.time()
    data.get()
    print(f"transfering raw {time.time()-start:.3f}")
    start = time.time()
    s = apply_threshold(data, a, b, threshold=0.99)
    cp.cuda.Stream.null.synchronize()
    print(f"converting to sparse {time.time() - start:.3f}")
    # Transfer sparse matrix back to host
    start = time.time()
    host_result = s.get()
    print(f"transfering sparse {time.time() - start:.3f}")

    print(host_result.shape)
    n = 6
    print(data[0:n])
    # Convert to dense matrix for display and comparison purposes
    print(host_result.toarray()[0, 0:n])
