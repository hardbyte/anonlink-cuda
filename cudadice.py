import humanize
import time
import cupy as cp

from chunking import chunk

encoding_size_in_bits = 1024
bflen = encoding_size_in_bits // 32



dice_kernel = cp.RawKernel(f'''
extern "C" __global__
void dice(float *out, 
          const unsigned *A, 
          const unsigned *B, 
          int asize, 
          int bsize) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > asize || j > bsize || i > j)
        return;

    unsigned a, b;
    unsigned pop_a = 0, pop_b = 0, pop_ab = 0;
    unsigned t;
    for (int k = 0; k < {bflen}; ++k) {{
        a = A[i*{bflen} + k];
        b = B[j*{bflen} + k];
        unsigned a_and_b = a & b;

        pop_a += __popc(a);
        pop_b += __popc(b);
        pop_ab += __popc(a_and_b);
    }}

    out[i * bsize + j] = 2.0*pop_ab/(pop_a + pop_b);
}}
''', 'dice')


def generate_random_encoding(number_of_encodings=2**20, n_bits=1024):
    maxval = 2 ** 32 - 1
    assert n_bits % 32 == 0, "nbits must be divisible by 32"
    size = number_of_encodings * bflen
    return cp.random.randint(0, maxval, size=size, dtype='uint32')


def compute_similarities(input_a, input_b, chunk_id=0):
    start_time = time.time()
    size_a, size_b = len(input_a)//32, len(input_b)//32

    similarities = cp.zeros((size_a, size_b), dtype=cp.float32)

    a_threads_per_block = 16
    b_threads_per_block = 16
    threads_per_block = (a_threads_per_block, b_threads_per_block)

    nblocks_a = size_a // a_threads_per_block
    nblocks_b = size_b // b_threads_per_block
    nblocks = (nblocks_a, nblocks_b)

    dice_kernel(nblocks, threads_per_block, (
        similarities,
        input_a,
        input_b,
        size_a,
        size_b
    ))

    cp.cuda.Stream.null.synchronize()

    compute_time = time.time() - start_time
    # Copy data back from device to host
    start_time = time.time()
    data = similarities.get()
    transfer_time = time.time() - start_time

    comparisons = size_a * size_b / 2
    cmp_per_sec = comparisons / compute_time
    print(f"{chunk_id}: Comparisons: {humanize.intword(size_a*size_b)}  ({humanize.intword(cmp_per_sec)}) cmp/s. Computation: {compute_time:.3f} Result transfer: {transfer_time:.3f}s")

    return data


def test_dice_kernel(size_a=2**10, size_b=2**14):
    input_a = generate_random_encoding(size_a)
    input_b = generate_random_encoding(size_b)

    results = []
    for i, (c1, c2) in enumerate(chunk(input_a, input_b)):
        # {'datasetIndex': 0, 'range': [8192, 16384]}
        a_start, a_end = c1['range']
        b_start, b_end = c2['range']
        res = compute_similarities(
            input_a[a_start:a_end],
            input_b[b_start:b_end],
            chunk_id=i
        )
        results.append(res)

    return results


for i in range(1):
    size = 2**15
    s = test_dice_kernel(size, size)




