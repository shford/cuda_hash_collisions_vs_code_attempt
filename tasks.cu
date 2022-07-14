#pragma once

#include "tasks.cuh"


// modified WjCryptLib_Md5 library
typedef struct
{
    uint32_t     lo;
    uint32_t     hi;
    uint32_t     a;
    uint32_t     b;
    uint32_t     c;
    uint32_t     d;
    uint8_t      buffer[64];
    uint32_t     block[16];
} Md5Context_dev;

#define MD5_HASH_SIZE           ( 128 / 8 )

typedef struct
{
    uint8_t      bytes[MD5_HASH_SIZE];
} MD5_HASH_dev;

__device__ void memcpy_dtod(char* dst, char* src, int size)
{
    for (int byte = 0; byte <= size; ++byte)
    {
        dst[byte] = src[byte];
    }
}

__device__ void memset_dtod(char* dst, uint8_t src, int size)
{
    for (int byte = 0; byte <= size; ++byte)
    {
        dst[byte] = src;
    }
}

#define F( x, y, z )            ( (z) ^ ((x) & ((y) ^ (z))) )
#define G( x, y, z )            ( (y) ^ ((z) & ((x) ^ (y))) )
#define H( x, y, z )            ( (x) ^ (y) ^ (z) )
#define I( x, y, z )            ( (y) ^ ((x) | ~(z)) )
#define STEP( f, a, b, c, d, x, t, s )                          \
    (a) += f((b), (c), (d)) + (x) + (t);                        \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s))));  \
    (a) += (b);

__device__ static void* TransformFunction_dev(Md5Context_dev* ctx, void const* data, uintmax_t size)
{
    uint8_t* ptr;
    uint32_t     a;
    uint32_t     b;
    uint32_t     c;
    uint32_t     d;
    uint32_t     saved_a;
    uint32_t     saved_b;
    uint32_t     saved_c;
    uint32_t     saved_d;

#define GET(n) (ctx->block[(n)])
#define SET(n) (ctx->block[(n)] =             \
            ((uint32_t)ptr[(n)*4 + 0] << 0 )      \
        |   ((uint32_t)ptr[(n)*4 + 1] << 8 )      \
        |   ((uint32_t)ptr[(n)*4 + 2] << 16)      \
        |   ((uint32_t)ptr[(n)*4 + 3] << 24) )

    ptr = (uint8_t*)data;

    a = ctx->a;
    b = ctx->b;
    c = ctx->c;
    d = ctx->d;

    do
    {
        saved_a = a;
        saved_b = b;
        saved_c = c;
        saved_d = d;

        // Round 1
        STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
            STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
            STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
            STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
            STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
            STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
            STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
            STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
            STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
            STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
            STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
            STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
            STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
            STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
            STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
            STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

            // Round 2
            STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
            STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
            STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
            STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
            STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
            STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
            STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
            STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
            STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
            STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
            STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
            STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
            STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
            STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
            STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
            STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

            // Round 3
            STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
            STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
            STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
            STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
            STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
            STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
            STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
            STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
            STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
            STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
            STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
            STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
            STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
            STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
            STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
            STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)

            // Round 4
            STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
            STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
            STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
            STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
            STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
            STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
            STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
            STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
            STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
            STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
            STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
            STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
            STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
            STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
            STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
            STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

            a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;

        ptr += 64;
    } while (size -= 64);

    ctx->a = a;
    ctx->b = b;
    ctx->c = c;
    ctx->d = d;

#undef GET
#undef SET

    return ptr;
}

__device__ void Md5Initialise_dev(Md5Context_dev* Context)
{
    Context->a = 0x67452301;
    Context->b = 0xefcdab89;
    Context->c = 0x98badcfe;
    Context->d = 0x10325476;

    Context->lo = 0;
    Context->hi = 0;
}

__device__ void Md5Update_dev(Md5Context_dev* Context, void const* Buffer, uint32_t BufferSize)
{
    uint32_t    saved_lo;
    uint32_t    used;
    uint32_t    free;

    saved_lo = Context->lo;
    if ((Context->lo = (saved_lo + BufferSize) & 0x1fffffff) < saved_lo)
    {
        Context->hi++;
    }
    Context->hi += (uint32_t)(BufferSize >> 29);

    used = saved_lo & 0x3f;

    if (used)
    {
        free = 64 - used;

        if (BufferSize < free)
        {
            memcpy_dtod((char*)&Context->buffer[used], (char*)Buffer, BufferSize);
            return;
        }

        memcpy_dtod((char*)&Context->buffer[used], (char*)Buffer, free);
        Buffer = (uint8_t*)Buffer + free;
        BufferSize -= free;
        TransformFunction_dev(Context, Context->buffer, 64);
    }

    if (BufferSize >= 64)
    {
        Buffer = TransformFunction_dev(Context, Buffer, BufferSize & ~(unsigned long)0x3f);
        BufferSize &= 0x3f;
    }

    memcpy_dtod((char*)Context->buffer, (char*)Buffer, BufferSize);
}

__device__ void Md5Finalise_dev(Md5Context_dev* Context, MD5_HASH_dev* Digest)
{
    uint32_t    used;
    uint32_t    free;

    used = Context->lo & 0x3f;

    Context->buffer[used++] = 0x80;

    free = 64 - used;

    if (free < 8)
    {
        memset_dtod((char*)&Context->buffer[used], 0, free);
        TransformFunction_dev(Context, Context->buffer, 64);
        used = 0;
        free = 64;
    }

    memset_dtod((char*)&Context->buffer[used], 0, free - 8);

    Context->lo <<= 3;
    Context->buffer[56] = (uint8_t)(Context->lo);
    Context->buffer[57] = (uint8_t)(Context->lo >> 8);
    Context->buffer[58] = (uint8_t)(Context->lo >> 16);
    Context->buffer[59] = (uint8_t)(Context->lo >> 24);
    Context->buffer[60] = (uint8_t)(Context->hi);
    Context->buffer[61] = (uint8_t)(Context->hi >> 8);
    Context->buffer[62] = (uint8_t)(Context->hi >> 16);
    Context->buffer[63] = (uint8_t)(Context->hi >> 24);

    TransformFunction_dev(Context, Context->buffer, 64);

    uint8_t test_seg = (uint8_t)(Context->a);
    Digest->bytes[0] = test_seg;
    Digest->bytes[1] = (uint8_t)(Context->a >> 8);
    Digest->bytes[2] = (uint8_t)(Context->a >> 16);
    Digest->bytes[3] = (uint8_t)(Context->a >> 24);
    Digest->bytes[4] = (uint8_t)(Context->b);
    Digest->bytes[5] = (uint8_t)(Context->b >> 8);
    Digest->bytes[6] = (uint8_t)(Context->b >> 16);
    Digest->bytes[7] = (uint8_t)(Context->b >> 24);
    Digest->bytes[8] = (uint8_t)(Context->c);
    Digest->bytes[9] = (uint8_t)(Context->c >> 8);
    Digest->bytes[10] = (uint8_t)(Context->c >> 16);
    Digest->bytes[11] = (uint8_t)(Context->c >> 24);
    Digest->bytes[12] = (uint8_t)(Context->d);
    Digest->bytes[13] = (uint8_t)(Context->d >> 8);
    Digest->bytes[14] = (uint8_t)(Context->d >> 16);
    Digest->bytes[15] = (uint8_t)(Context->d >> 24);
}

__device__ void Md5Calculate_dev(void  const* Buffer, uint32_t BufferSize, MD5_HASH_dev* Digest)
{
    Md5Context_dev context;

    Md5Initialise_dev(&context);
    Md5Update_dev(&context, Buffer, BufferSize);
    Md5Finalise_dev(&context, Digest);
}




// statically initialized global variables
__device__ uint8_t d_num_collisions_found = 0;              // track number of collisions found by active kernel
__device__ unsigned long long d_collision_attempts = 0;     // track total number of attempts per collision
__device__ int d_global_mutex = UNLOCKED;                   // signal mutex to other threads (globally)
__device__ int d_collision_flag = FALSE;                    // signal host to read

// dynamically initialized in host
__constant__ __device__ MD5_HASH_dev d_const_md5_digest;    // store digest on L2 or L1 cache (on v8.6)
__device__ unsigned long long d_collision_size;             // track # of characters in collision

__global__ void find_collisions(volatile char* collision) {
    //===========================================================================================================
    // DECLARATIONS & INITIALIZATION
    //===========================================================================================================

    // create warp synchronization semaphore
    __shared__ int sync_warp_flag;

    // force each thread to get its "warp id"
    __shared__ int warp_group[CUDA_CORES_PER_MULTIPROCESSOR];

    for (int tid = 0; tid < CUDA_CORES_PER_MULTIPROCESSOR; ++tid) {
        
    }

    // allocate local buffer and keep track of size in case of resizing
    char local_collision[ARBITRARY_MAX_BUFF_SIZE];
    //unsigned long long local_buff_size = ARBITRARY_MAX_BUFF_SIZE;
    //cudaError_t ret = cudaMalloc((void**)&local_collision, local_buff_size);
    //if (ret != cudaSuccess) {
    //  printf("Err: %d. Local buffer allocation failed.\n", (int)ret);
    //}
    

    // initialize local buffer
    unsigned long long local_collision_size = d_collision_size;
    for (int byte_index = 0; byte_index <= local_collision_size; ++byte_index) {
        local_collision[byte_index] = collision[byte_index];
    }

    // allocate room for new digest
    MD5_HASH_dev local_md5_digest;

    // allocate storage for random character
    unsigned long long random_index = NUM_8BIT_RANDS;
    uint8_t randoms[NUM_8BIT_RANDS];

    //===========================================================================================================
    // COMPUTATIONS - GENERATE RANDS, RESIZE BUFFER, APPEND CHAR, HASH, COMPARE { EXIT }
    //===========================================================================================================

    do
    {
        // generate a new batch of random numbers as needed
        if (random_index == NUM_8BIT_RANDS) {
            random_index = 0;
            for (int i = 0; i < NUM_32BIT_RANDS; ++i)
            {
                int id = threadIdx.x + blockIdx.x * blockDim.x;
                curandStatePhilox4_32_10_t state;
                curand_init(i, id, 0, &state);
                // effectively assigns 4 random uint8_t's per execution
                *((uint32_t*)(randoms + i*4)) = curand(&state);
            }
        }

        /*
        // resize local_collision
        if (local_collision_size == ARBITRARY_MAX_BUFF_SIZE) {
            // retain ptr to old buffer
            char* old_buff = local_collision;

            // reassign local_collision ptr to new buffer
            local_buff_size *= 2;
            cudaMalloc(&local_collision, local_buff_size);

            // copy data from old buffer to new buffer
            for (int i = 0; i < ARBITRARY_MAX_BUFF_SIZE; ++i) {
                local_collision[i] = old_buff[i];
            }

            // free original buffer
            //cudaFree(old_buff);
        }
        */

        // append random char
        uint8_t character = randoms[random_index];
        local_collision[local_collision_size - 1] = character ? character : 1; // no premature null terminators
        local_collision[local_collision_size] = '\0';
        ++random_index;
        ++local_collision_size;

        // generate new hash
        Md5Calculate_dev((const void*)local_collision, local_collision_size, &local_md5_digest);

        // terminate all threads if first 20 bits of digest match (and prevent warp deadlock)
        if ((*((uint32_t*)&d_const_md5_digest) >> 12 == *((uint32_t*)&local_md5_digest) >> 12) || sync_warp_flag == TRUE)
        {
            // setup warp sync before setting mutex b/c loops can cause deadlock
            designatedThreadId = threadIdx.x;
            sync_warp_flag = TRUE;
            __syncwarp(); // causes the executing thread to wait until all threads specified in mask have executed a __syncwarp()

            // ensure threads converge so that critical thread is not indefinetly suspended
            while (sync_warp_flag)
            {
                // do mutex operations
                if (threadIdx.x == designatedThreadId)
                {
                    // set mutex lock
                    do {} while (atomicCAS(&d_global_mutex, UNLOCKED, LOCKED));

                    // enter critical section - writing for host polls and signalling when ready
                    for (int byte_index = 0; byte_index <= local_collision_size; ++byte_index) {
                        collision[byte_index] = local_collision[byte_index];
                    }
                    d_collision_size = local_collision_size;
                    ++d_num_collisions_found;

                    // signal host to read
                    d_collision_flag = TRUE;

                    // free lock only once host signals finished reading (e.g. d_collision_flag = FALSE)
                    do {} while (d_collision_flag);

                    // safely unlock mutex by writing to flag - remember relaxed ordering doesn't matter here
                    atomicExch(&d_global_mutex, UNLOCKED);

                    // release non-critical warp threads and reset flag
                    sync_warp_flag = FALSE;
                }
                else
                {
                    // have non-critical warp threads read check for 
                }
            }
        }
        // increment hash attempts
        atomicAdd(&d_collision_attempts, 1);
    } while(d_num_collisions_found < TARGET_COLLISIONS);
    //cudaFree(local_collision);
}

void task1() {
    //===========================================================================================================
    // SEQUENTIAL TASKS (Initial)
    //===========================================================================================================

    // todo v5 cudaMallocHost - code chunk has been tested
    // char* h_page_locked_data;
    // gpuErrchk( cudaMallocHost(&h_page_locked_data, ARBITRARY_MAX_BUFF_SIZE) );
    // cudaMemset(&h_page_locked_data, 0x00, sizeof(char) * ARBITRARY_MAX_BUFF_SIZE);

    // read file data
    char sampleFile_path[] = "C:/Users/shford/CLionProjects/cuda_hashing/sample.txt";
    char* h_sampleFile_buff;
    uint32_t h_sampleFile_buff_size = 0; // handle files up to ~4GiB (2^32-1 bytes) -- may be 1 byte too small
    get_file_data((char*)sampleFile_path, &h_sampleFile_buff, &h_sampleFile_buff_size);

    // get hash md5_digest
    MD5_HASH md5_digest;
    Md5Calculate((const void*)h_sampleFile_buff, h_sampleFile_buff_size, &md5_digest);

    // format and print digest as a string of hex characters
    char hash[MD5_HASH_SIZE_B + 1]; //MD5 len is 16B, 1B = 2 chars

    char tiny_hash[TINY_HASH_SIZE_B + 1];
    for (int i = 0; i < MD5_HASH_SIZE / 2; ++i)
    {
        sprintf(hash + i * 2, "%2.2x", md5_digest.bytes[i]);
    }
    hash[sizeof(hash)-1] = '\0';
    strncpy_s(tiny_hash, sizeof(tiny_hash), hash, _TRUNCATE);
    tiny_hash[sizeof(tiny_hash)-1] = '\0';

    printf("Full MD5 md5_digest is: %s\n", hash);
    printf("TinyHash md5_digest is: %s\n\n", tiny_hash);

    //===========================================================================================================
    // BEGIN CUDA PARALLELIZATION
    //===========================================================================================================

    // allocate storage for collisions once found
    char* h_collisions[TARGET_COLLISIONS];
    unsigned long long h_collision_sizes[TARGET_COLLISIONS];
    unsigned long long h_collision_attempts = 0;
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        h_collisions[i] = (char*)calloc(1, ARBITRARY_MAX_BUFF_SIZE);
        h_collision_sizes[i] = 0;
    }
    uint8_t h_collision_index = 0;
    int h_collision_flag = FALSE;

    // initialize device
    cudaSetDevice(0);

    // allocate global mem for collision - initialized in loop
    volatile char* d_collision;
    gpuErrchk( cudaMalloc((void **)&d_collision, ARBITRARY_MAX_BUFF_SIZE) );

    // parallelization setup - initialize device globals
    gpuErrchk( cudaMemcpyToSymbol(d_const_md5_digest, &md5_digest, sizeof(md5_digest), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(d_collision_size, &h_sampleFile_buff_size, sizeof(h_sampleFile_buff_size), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy((void*)d_collision, h_sampleFile_buff, h_sampleFile_buff_size, cudaMemcpyHostToDevice) );

    // execution configuration (sync device)
    find_collisions <<<MULTIPROCESSORS-5, 1>>> (d_collision);
    //find_collisions<<<MULTIPROCESSORS-5, CUDA_CORES_PER_MULTIPROCESSOR>>>(d_collision);

    // run kernel
    while (h_collision_index < TARGET_COLLISIONS)
    {
        // poll collision flag
        while (!h_collision_flag)
        {
            // NOTE: errors on this line are most likely from the kernel
            gpuErrchk(cudaMemcpyFromSymbol(&h_collision_flag, d_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyDeviceToHost));
        }

        // read updated collision count, collision size, collision, and hash attempts
        gpuErrchk(cudaMemcpyFromSymbol(&h_collision_sizes[h_collision_index], d_collision_size, sizeof(h_sampleFile_buff_size), 0, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_collisions[h_collision_index], (const void*)d_collision, h_collision_sizes[h_collision_index], cudaMemcpyDeviceToHost) );

        // reset flags to release mutex and reset kernel
        h_collision_flag = FALSE;
        //cudaMemcpyToSymbol(d_global_mutex, &h_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_collision_flag, &h_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyHostToDevice);

        // increment collision index
        ++h_collision_index;
        
        // on final loop read the number of cumulative attempts
        if (h_collision_index == TARGET_COLLISIONS)
        {
            gpuErrchk(cudaMemcpyFromSymbol(&h_collision_attempts, d_collision_attempts,
                sizeof(h_collision_attempts), 0, cudaMemcpyDeviceToHost));
        }
    }
    gpuErrchk( cudaDeviceSynchronize() );

    // free collisions
    cudaFree((void*)d_collision);
    free(h_sampleFile_buff);

    printf("\nCalculated %d collisions in %lld attempts... Success!/\n", TARGET_COLLISIONS, h_collision_attempts);

    //===========================================================================================================
    // WRITE COLLISIONS TO DISK
    //===========================================================================================================

    printf("Original string: %s\n", h_sampleFile_buff);
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        printf("Collision %d: %s\n", i, h_collisions[i]);

        // todo write collision

        // free collision once written
        free(h_collisions[i]);
    }
}
