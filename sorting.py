import time
import cupy as cp
from filtering import apply_threshold


def sort_sparse_similarities(s):
    new_order = cp.argsort(s.data)[::-1]

    s.data = s.data[new_order]
    s.row = s.row[new_order]
    s.col = s.col[new_order]


if __name__ == '__main__':
    a, b = 2**13, 2**10
    # generate our similarity scores as a dense matrix of floats
    data = cp.random.rand(a*b, dtype='float')
    #data.get()

    start = time.time()
    s, num_results = apply_threshold(data, a, b, threshold=0.5)
    cp.cuda.Stream.null.synchronize()
    print(f"converting to sparse {time.time() - start:.3f}")

    sort_sparse_similarities(s)

    sparse_result = s.get()

    print(sparse_result.data)
