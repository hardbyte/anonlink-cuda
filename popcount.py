import humanize
import time
import cupy as cp

encoding_size_in_bits = 1024
bflen = encoding_size_in_bits // 32

popcnt_kernel = cp.RawKernel(f'''
extern "C" __global__
void popcnt(unsigned *out, const unsigned *x, int outsz) {{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= outsz)
        return;
    const unsigned *y = x + tid * {bflen};
    unsigned pc = 0, t;
    for (int i = 0; i < {bflen}; ++i) {{
        asm ("popc.b32 %0, %1;" : "=r"(t) : "r"(y[i]));
        pc += t;
    }}
    out[tid] = pc;
}}
''', 'popcnt')

dice_kernel = cp.RawKernel(f'''
extern "C" __global__
void dice(float *out, 
          const unsigned *A, 
          const unsigned *B, 
          const unsigned *Pa, 
          const unsigned *Pb, 
          int asize, 
          int bsize) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (i > asize || j > bsize || i > j)
        return;
    
    // bitwise and for all {bflen} elements
    // a[i] & b[j]

    unsigned pc = 0, t;
    for (int k = 0; k < {bflen}; ++k) {{
        unsigned a_and_b = A[i*{bflen} + k] & B[j*{bflen} + k];
        asm ("popc.b32 %0, %1;" : "=r"(t) : "r"(a_and_b));
        pc += t;
    }}

    out[i * bsize + j] = 2.0*pc/(Pa[i] + Pb[j]);
}}
''', 'dice')


def generate_random_encoding(number_of_encodings=2**20, n_bits=1024):
    maxval = 2 ** 32 - 1
    assert n_bits % 32 == 0, "nbits must be divisible by 32"
    size = number_of_encodings * bflen

    return cp.random.randint(0, maxval, size=size, dtype='uint32')


def popcount_vector(encoding_vector):
    threads_per_block = 512
    input_size = len(encoding_vector)
    output_size = input_size//bflen
    assert input_size % threads_per_block == 0, 'nthreads must divide array size'
    nblocks = input_size // threads_per_block
    output_popcount = cp.zeros(output_size, dtype='uint32')
    popcnt_kernel((nblocks,), (threads_per_block,), (output_popcount, encoding_vector, output_size))
    return output_popcount


def test_dice_kernel(size_a=2**10, size_b=2**14):
    input_a = generate_random_encoding(size_a)
    input_b = generate_random_encoding(size_b)

    popcnt_a = popcount_vector(input_a)
    popcnt_b = popcount_vector(input_b)

    similarities = cp.zeros((size_a, size_b), dtype=cp.float32)

    a_threads_per_block = 32
    b_threads_per_block = 32
    threads_per_block = (a_threads_per_block, b_threads_per_block)

    nblocks_a = size_a // a_threads_per_block
    nblocks_b = size_b // b_threads_per_block
    nblocks = (nblocks_a, nblocks_b)

    dice_kernel(nblocks, threads_per_block, (
        similarities,
        input_a,
        input_b,
        popcnt_a,
        popcnt_b,
        size_a,
        size_b
    ))

    return similarities


for size in [2**10, 2**12]:
    start_time = time.time()
    s = test_dice_kernel(size, size).get()
    elapsed = time.time() - start_time
    comparisons = size*size/2

    cmp_per_sec = comparisons/elapsed
    print(f"{size}^2  {cmp_per_sec:.0f} ({humanize.intword(cmp_per_sec)}) cmp/s")


# popcnt_shared_memory = cp.RawKernel(f'''
#
#  __global__
# void mypopcnt(unsigned *out, const unsigned *x, int outsz) {{
#     __shared__ unsigned y[{bflen} * {threads_per_block}];
#
#     int tid = blockDim.x * blockIdx.x + threadIdx.x;
#     if (tid >= outsz)
#         return;
#
#     for (int i = 0; i < {bflen} * {threads_per_block}; i += {threads_per_block})
#         y[i] = x[i + tid * {bflen}];
#
#     //const unsigned *y = x + tid * {bflen};
#     unsigned pc = 0, t;
#     for (int i = 0; i < {bflen}; ++i) {{
#         asm ("popc.b32 %0, %1;" : "=r"(t) : "r"(y[tid * {bflen} + i]));
#         pc += t;
#     }}
#     out[tid] = pc;
# }}
# ''', 'mypopcnt')

# Test
def test_dice(n=2**10):
    maxval = 2**32 - 1
    input_a = cp.random.randint(0, maxval, size=n, dtype='uint32')
    input_b = cp.random.randint(0, maxval, size=n, dtype='uint32')
    assert n % bflen == 0, 'bflen must divide array size'
    output_size = n // bflen

    assert n % threads_per_block == 0, 'nthreads must divide array size'
    nblocks = n // threads_per_block

    start_time = time.time()
    popcnt_a = cp.zeros(output_size, dtype='uint32')
    popcnt_b = cp.zeros(output_size, dtype='uint32')
    popcnt_a_and_b = cp.zeros(output_size, dtype='uint32')
    similarities = cp.zeros(output_size*output_size, dtype='uint32')

    popcnt_kernel((nblocks,), (threads_per_block,), (popcnt_a, input_a, output_size))
    popcnt_kernel((nblocks,), (threads_per_block,), (popcnt_b, input_b, output_size))
    popcount_time = time.time()

    a_and_b = cp.bitwise_and(input_a, input_b)
    popcnt_kernel((nblocks,), (threads_per_block,), (popcnt_a_and_b, a_and_b, output_size))

    # compute the similarities for the diag
    #similarities = 2*popcnt_a_and_b/(popcnt_a+popcnt_b)


    dice_time = time.time()
    print(f"Compared {n} encodings of {encoding_size_in_bits} in {dice_time - start_time}s")
    print(f"{n/(dice_time - start_time):.1f} cmp/s")


    return similarities


#
# x = test_dice(1<<20).get()
# print(len(x))
# print(x[x>1])
# print(x[:10])
#test_dice(n=(1<<28))

#print(timeit.timeit(stmt=test_dice, number=10))

