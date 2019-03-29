import humanize
import time
import cupy as cp
import numpy as np
import anonlink
from bitarray import bitarray
from chunking import chunk
from filtering import apply_threshold
from sorting import sort_sparse_similarities

encoding_size_in_bits = 1024
assert encoding_size_in_bits % 32 == 0, "nbits in encoding must be divisible by 32"
bflen = encoding_size_in_bits // 32


dice_kernel = cp.RawKernel(f'''
extern "C" __global__
void dice(float *out, 
          const unsigned *A, 
          const unsigned *B, 
          int asize, 
          int bsize,
          float threshold
          ) {{

    __shared__ unsigned shared_A[16 * 32];
    __shared__ unsigned shared_B[16 * 32];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > asize || j > bsize )
        return;

    for (int index = threadIdx.y * 16 + threadIdx.x; index < 16 * 32; index += 256)
    {{
        shared_A[index] = A[blockIdx.x * blockDim.x * 32 + index];
        shared_B[index] = B[blockIdx.y * blockDim.y * 32 + index];
    }}

    __syncthreads();

    unsigned a, b, a_and_b;
    unsigned pop_a = 0, pop_b = 0, pop_ab = 0;
    for (int k = 0; k < 32; ++k) {{
        a = shared_A[threadIdx.x * 32 + k];
        b = shared_B[threadIdx.y * 32 + k];
        a_and_b = a & b;
        pop_a += __popc(a);
        pop_b += __popc(b);
        pop_ab += __popc(a_and_b);
    }}

    float score = 2.0 * pop_ab / (pop_a + pop_b);
    score = score > threshold ? score : 0.0;
    out[i * bsize + j] = score;
}}
''', 'dice')


def generate_random_encoding(number_of_encodings=2**20):
    maxval = 2 ** 32 - 1
    size = (number_of_encodings, bflen)
    return np.random.randint(0, maxval, size=size, dtype='uint32')


def compute_similarities(input_a, input_b, chunk_id, threshold):
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
        size_b,
        cp.float32(threshold)
    ))

    sparse_similarities, num_results = apply_threshold(similarities, size_a, size_b, threshold=threshold)

    sort_sparse_similarities(sparse_similarities)

    cp.cuda.Stream.null.synchronize()

    compute_time = time.time() - start_time
    # Copy data back from device to host
    start_time = time.time()

    # Transfer sparse matrix back to host
    data = sparse_similarities.get()

    transfer_time = time.time() - start_time

    comparisons = size_a * size_b
    cmp_per_sec = comparisons / compute_time

    if chunk_id % 1 == 0:
        print(f"{chunk_id}: Comparisons: {humanize.intword(comparisons)}, Rate: {humanize.intword(cmp_per_sec)} cmp/s. Computation: {compute_time:.3f} Result transfer: {transfer_time:.6f}s")

    return data, num_results


def numpy_array_to_bitarray(a):
    ba = bitarray()
    ba.frombytes(a.tobytes())
    return ba


def test_dice_kernel(size_a=2**10, size_b=2**14, skip_anonlink=True):
    input_a = generate_random_encoding(size_a)
    input_b = generate_random_encoding(size_b)

    a_ba = [numpy_array_to_bitarray(a) for a in input_a]
    b_ba = [numpy_array_to_bitarray(a) for a in input_b]

    input_a = cp.asarray(input_a.ravel())
    input_b = cp.asarray(input_b.ravel())

    t = 0.55

    if not skip_anonlink:
        print("Computing in anonlink")
        start = time.time()
        z = anonlink.similarities.dice_coefficient([a_ba, b_ba], threshold=t)
        end = time.time()
        print(f"Anonlink computing {humanize.intword(size_a*size_b)} comparisons took a total time {end - start:.2f}")

    results = []
    num_results = 0

    for i, (c1, c2) in enumerate(chunk(input_a, input_b)):
        a_start, a_end = c1['range']
        b_start, b_end = c2['range']

        res, count = compute_similarities(
            input_a[a_start:a_end],
            input_b[b_start:b_end],
            chunk_id=i,
            threshold=t
        )
        results.append(res)
        num_results += count

    # TODO need to merge sort the results...
    return results


for i in range(1):
    size = 2**16
    start = time.time()
    s = test_dice_kernel(size, size)
    end = time.time()

    print(f"CUDA computing {humanize.intword(size**2)} comparisons took a total time {end - start:.2f}")

