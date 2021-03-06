import humanize
import time
import cupy as cp
from cupy import prof

encoding_size_in_bits = 1024
bflen = encoding_size_in_bits // 32


threads_per_block = 128
popcnt_shared_memory = cp.RawKernel(f'''
extern "C" __global__
void mypopcnt(unsigned *out, const unsigned *x, int outsz) {{
    __shared__ unsigned int y[{bflen} * {threads_per_block}];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid >= outsz)
        return;

    for (int i = threadIdx.x; i < {bflen} * {threads_per_block}; i += {threads_per_block}) {{
        int index = i + blockDim.x * blockIdx.x * {bflen};
        if (i >= {bflen} * {threads_per_block})
            printf("OUT OF RNGE: %d, %d\\n", i, threadIdx.x);
        
        // So y[i] is broken.
        y[i] = x[index];
        //y[0] = 1;
    }}
    
    __syncthreads();
    
    //const unsigned *y = x + tid * {bflen};
    unsigned pc = 0, t;
    for (int i = 0; i < {bflen}; ++i) {{
        asm ("popc.b32 %0, %1;" : "=r"(t) : "r"(y[threadIdx.x * {bflen} + i]));
        pc += t;
    }}
    out[tid] = pc;
}}
''', 'mypopcnt')

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

    unsigned pc = 0, t;
    for (int k = 0; k < {bflen}; ++k) {{
        unsigned a_and_b = A[i*{bflen} + k] & B[j*{bflen} + k];
        asm ("popc.b32 %0, %1;" : "=r"(t) : "r"(a_and_b));
        pc += t;
    }}

    out[i * bsize + j] = 2.0*pc/(Pa[i] + Pb[j]);
}}
''', 'dice')


dice_kernel_all_in_one = cp.RawKernel(f'''
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


def popcount_vector(encoding_vector):
    threads_per_block = 128
    input_size = len(encoding_vector)
    output_size = input_size//bflen
    assert input_size % threads_per_block == 0, 'nthreads must divide array size'
    nblocks = input_size // threads_per_block
    output_popcount = cp.zeros(output_size, dtype='uint32')
    popcnt_kernel((nblocks,), (threads_per_block,), (output_popcount, encoding_vector, output_size))
    #popcnt_shared_memory((nblocks,), (threads_per_block,), (output_popcount, encoding_vector, output_size))
    return output_popcount


@cp.prof.TimeRangeDecorator()
def test_dice_kernel(size_a=2**10, size_b=2**14):
    input_a = generate_random_encoding(size_a)
    input_b = generate_random_encoding(size_b)

    #popcnt_a = popcount_vector(input_a)
    #popcnt_b = popcount_vector(input_b)
    start_time = time.time()

    similarities = cp.zeros((size_a, size_b), dtype=cp.float32)


    a_threads_per_block = 16
    b_threads_per_block = 16
    threads_per_block = (a_threads_per_block, b_threads_per_block)

    nblocks_a = size_a // a_threads_per_block
    nblocks_b = size_b // b_threads_per_block
    nblocks = (nblocks_a, nblocks_b)

    dice_kernel_all_in_one(nblocks, threads_per_block, (
        similarities,
        input_a,
        input_b,
        size_a,
        size_b
    ))

    # dice_kernel(nblocks, threads_per_block, (
    #     similarities,
    #     input_a,
    #     input_b,
    #     popcnt_a,
    #     popcnt_b,
    #     size_a,
    #     size_b
    # ))

    cp.cuda.Stream.null.synchronize()
    #data = similarities.get()
    elapsed = time.time() - start_time
    comparisons = size_a * size_b / 2

    cmp_per_sec = comparisons / elapsed
    print(f"{size}^2  ({humanize.intword(cmp_per_sec)}) cmp/s")

    return similarities


for i in range(5):
    size = 2**15

    #cp.cuda.profiler.start()
    s = test_dice_kernel(size, size).get()
    #cp.cuda.profiler.stop()



# Test
def test_popcounting(n):

    maxval = 2**32 - 1
    input_a = cp.random.randint(0, maxval, size=n, dtype='uint32')
    assert n % bflen == 0, 'bflen must divide array size'
    output_size = n // bflen

    assert n % threads_per_block == 0, 'nthreads must divide array size'
    nblocks = (output_size + threads_per_block - 1) // threads_per_block

    popcnt_a = cp.zeros(output_size, dtype='uint32')

    start_time = time.time()

    # popcnt_shared_memory((nblocks,),
    #                      (threads_per_block,),
    #                      (popcnt_a, input_a, output_size))
    #

    popcnt_kernel((nblocks,), (threads_per_block,), (popcnt_a, input_a, output_size))
    popcount_time = time.time()

    #a_and_b = cp.bitwise_and(input_a, input_b)
    #popcnt_kernel((nblocks,), (threads_per_block,), (popcnt_a_and_b, a_and_b, output_size))

    print(f"Popcount {n} encodings of {encoding_size_in_bits} in {popcount_time - start_time}s")
    return popcnt_a




#print(timeit.timeit(stmt=test_dice, number=10))


