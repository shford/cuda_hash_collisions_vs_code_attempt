/*
 * Constants for the RTX 3060-Ti obtained from Device Query (sample program)
 */
#pragma once

#ifndef CUDA_HASHING_CUDA_CONSTS_CUH
#define CUDA_HASHING_CUDA_CONSTS_CUH

#include <stdio.h>


#define COMPUTE_CAPABILITY 86
//===========================================================================================================
// CUDA CAPABILITY VERSION CONSTANTS
//===========================================================================================================
#if (COMPUTE_CAPABILITY == 86)
    #define MAX_RESIDENT_GRIDS_PER_DEVICE 128
    #define MAX_DIMENSIONALITY_OF_GRID_OF_THREADBLOCKS 3
    #define MAX_X_DIM_OF_A_GRID_OF_THREADBLOCKS 2147483647 // -eq 2**31-1
    #define MAX_Y_OR_Z_DIM_OF_A_GRID_OF_THREADBLOCKS 65535
    #define MAX_DIMENSIONALITY_OF_A_THREADBLOCK 3
    #define MAX_X_OR_Y_DIM_OF_A_THREADBLOCK 1024
    #define MAX_Z_DIM_OF_A_THREADBLOCK 64
    #define MAX_THREADS_PER_BLOCK 1024
    #define WARP_SIZE 32
    #define MAX_RESIDENT_BLOCKS_PER_SM 16
    #define MAX_RESIDENT_WARPS_PER_SM 48
    #define MAX_RESIDENT_THREADS_PER_SM 1536
    #define B32_REGISTERS_PER_SM 64000
    #define MAX_B32_REGISTERS_PER_THREADBLOCK 64000
    #define MAX_B32_REGISTERS_PER_THREAD 255
    #define MAX_SHAREDMEM_PER_SM (100*1024)
    #define MAX_STATIC_SHAREDMEM_PER_THREADBLOCK (48*1024) // see footnote 33 in Prog Manual
    #define MAX_SHAREDMEM_PER_THREADBLOCK (99*1024)
    #define MAX_SHARED_MEMBANKS 32
    #define MAX_LOCALMEM_PER_THREAD (512*1024)
    #define CONSTANT_MEM_SIZE (64*1024)
    #define CACHE_WORKING_SET_PER_SM_FOR_CONSTANT_MEM (8*1024)
    #define MIN_CACHE_WORKING_SET_PER_SM_FOR_TEXTURE_MEM (28*1024) // hardware specific b/w 28 & 128
    #define MAXIMUM_WIDTH_FOR_A_1D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 131072
    #define MAXIMUM_WIDTH_FOR_A_1D_TEXTURE_REFERENCE_BOUND_TO_LINEAR_MEMORY 268435456 // -eq 2^28
    #define MAXIMUM_WIDTH_FOR_A_1D_LAYERED_TEXTURE_REFERENCE 32768
    #define MAXIMUM_NUMBER_OF_LAYERS_FOR_A_1D_LAYERED_TEXTURE_REFERENCE 2048
    #define MAXIMUM_WIDTH_FOR_A_2D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 131072
    #define MAXIMUM_HEIGHT_FOR_A_2D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 65536
    #define MAXIMUM_WIDTH_FOR_A_2D_TEXTURE_REFERENCE_BOUND_TO_LINEAR_MEMORY 131072
    #define MAXIMUM_HEIGHT_FOR_A_2D_TEXTURE_REFERENCE_BOUND_TO_LINEAR_MEMORY 65000
    #define MAXIMUM_WIDTH_FOR_A_2D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY_SUPPORTING_TEXTURE_GATHER 32768
    #define MAXIMUM_HEIGHT_FOR_A_2D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY_SUPPORTING_TEXTURE_GATHER 32768
    #define MAXIMUM_WIDTH_FOR_A_2D_LAYERED_TEXTURE_REFERENCE 32768
    #define MAXIMUM_HEIGHT_FOR_A_2D_LAYERED_TEXTURE_REFERENCE 32768
    #define MAXIMUM_NUMBER_OF_LAYERS_FOR_A_2D_LAYERED_TEXTURE_REFERENCE 2048
    #define MAXIMUM_WIDTH_FOR_A_3D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 16384
    #define MAXIMUM_HEIGHT_FOR_A_3D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 16384
    #define MAXIMUM_DEPTH_FOR_A_3D_TEXTURE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 16384
    #define MAXIMUM_WIDTH_FOR_A_CUBEMAP_TEXTURE_REFERENCE 32768
    #define MAXIMUM_HEIGHT_FOR_A_CUBEMAP_TEXTURE_REFERENCE 32768
    #define MAXIMUM_WIDTH_FOR_A_CUBEMAP_LAYERED_TEXTURE_REFERENCE 32768
    #define MAXIMUM_HEIGHT_FOR_A_CUBEMAP_LAYERED_TEXTURE_REFERENCE 32768
    #define MAXIMUM_NUMBER_OF_LAYERS_FOR_A_CUBEMAP_LAYERED_TEXTURE_REFERENCE 2046
    #define MAXIMUM_NUMBER_OF_TEXTURES_THAT_CAN_BE_BOUND_TO_A_KERNEL 256
    #define MAXIMUM_WIDTH_FOR_A_1D_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 32768
    #define MAXIMUM_WIDTH_AND_NUMBER_OF_LAYERS_FOR_A_1D_LAYERED_SURFACE_REFERENCE 32768
    #define MAXIMUM_WIDTH_AND_NUMBER_OF_LAYERS_FOR_A_1D_LAYERED_SURFACE_REFERENCE 2048
    #define MAXIMUM_WIDTH_FOR_A_2D_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 131072
    #define MAXIMUM_HEIGHT_FOR_A_2D_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 65536
    #define MAXIMUM_WIDTH_FOR_A_2D_LAYERED_SURFACE_REFERENCE 32768
    #define MAXIMUM_HEIGHT_FOR_A_2D_LAYERED_SURFACE_REFERENCE 32768
    #define MAXIMUM_NUMBER_OF_LAYERS_FOR_A_2D_LAYERED_SURFACE_REFERENCE 2048
    #define MAXIMUM_WIDTH_FOR_A_3D_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 16384
    #define MAXIMUM_HEIGHT_FOR_A_3D_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 16384
    #define MAXIMUM_DEPTH_FOR_A_3D_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 16384
    #define MAXIMUM_WIDTH_FOR_A_CUBEMAP_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 32768
    #define MAXIMUM_HEIGHT_FOR_A_CUBEMAP_SURFACE_REFERENCE_BOUND_TO_A_CUDA_ARRAY 32768
    #define MAXIMUM_WIDTH_FOR_A_CUBEMAP_LAYERED_SURFACE_REFERENCE 32768
    #define MAXIMUM_HEIGHT_FOR_A_CUBEMAP_LAYERED_SURFACE_REFERENCE 32768
    #define MAXIMUM_NUMBER_OF_LAYERS_FOR_A_CUBEMAP_LAYERED_SURFACE_REFERENCE 2046
    #define MAXIMUM_NUMBER_OF_SURFACES_THAT_CAN_BE_BOUND_TO_A_KERNEL 32
#endif // (CUDA_COMPUTE_CAPABILITY == 86)

//===========================================================================================================
// DEVICE CONSTANTS (copied or derived from DeviceQuery.exe)
//===========================================================================================================
#define GLOBAL_MEMORY (8192*1024*1024)
#define CUDA_CORES 4864
#define MULTIPROCESSORS 38
#define CUDA_CORES_PER_MULTIPROCESSOR 128
#define MAX_MEM_PER_ACTIVE_THREAD (GLOBAL_MEMORY_MB*1024*1024 / CUDA_CORES)
#define WARP_SIZE 32
#define WARPS_PER_MULTIPROCESSOR (CUDA_CORES_PER_MULTIPROCESSOR / WARP_SIZE)
#define MEMORY_BUS_WIDTH 256
#define L2_CACHE_SIZE 3145728
#define MAX_TEXTURE_DIMENSION_SIZE_1D 131072
#define MAX_X_TEXTURE_DIMENSION_SIZE_2D 131072
#define MAX_Y_TEXTURE_DIMENSION_SIZE_2D 65536
#define MAX_X_Y_Z_TEXTURE_DIMENSION_SIZE_3D  16384
#define MAX_LAYERED_TEXTURE_SIZE_1D 32768
#define MAX_NUM_LAYERS_1D 2048
#define MAX_LAYERED_TEXTURE_SIZE_2D (32768*32768)
#define MAX_NUM_LAYERS_2D 2048
#define TOTAL_NUMBER_OF_REGISTERS_AVAILABLE_PER_BLOCK 65536
#define MAX_NUMBER_OF_THREADS_PER_MULTIPROCESSOR 1536
#define MAX_NUMBER_OF_THREADS_PER_BLOCK 1024
#define MAX_X_Y_DIMENSION_SIZE_OF_A_THREAD_BLOCK 1024 
#define MAX_Z_DIMENSION_SIZE_OF_A_THREAD_BLOCK 64
#define MAX_X_DIMENSION_SIZE_OF_A_GRID_SIZE 2147483647
#define MAX_Y_Z_DIMENSION_SIZE_OF_A_GRID_SIZE 65535

//===========================================================================================================
// USER DEFINED INPUTS - MODIFY FREELY
//===========================================================================================================
#define TARGET_COLLISIONS (1)
#define ENCODING_SIZE (256)
#define ARBITRARY_MAX_BUFF_SIZE (20000)     // MUST BE MULTIPLE OF 8 expected 4000, getting 10000... not sure why
#define BLOCKS_PER_KERNEL (MULTIPROCESSORS) // Default: HW Limit of 1 Active Block / 1 MP
#define THREADS_PER_BLOCK (CUDA_CORES_PER_MULTIPROCESSOR) // Default: HW Limit of 1 Active Thread / 1 Core

//===========================================================================================================
// USER DEFINED DERIVED INPUTS
//===========================================================================================================
#define THREADS_PER_KERNEL (BLOCKS_PER_KERNEL * THREADS_PER_BLOCK)

//===========================================================================================================
// PROGRAM CONSTRAINTS - DO NOT CHANGE (based off program limitations such as memory indexing)
//===========================================================================================================
#define NUM_8BIT_RANDS (48)
#define NUM_32BIT_RANDS (NUM_8BIT_RANDS / 4)

#define FALSE (0)
#define TRUE (1)
#define UNLOCKED (0)
#define LOCKED (1)

//===========================================================================================================
// USER INPUT SANITY CHECKS - DO NOT CHANGE (type literals correspond to program or hardware limitations)
//===========================================================================================================
#define THREADS_PER_KERNEL_leq_

#define TARGET_COLLISION_leq_THREADS_PER_KERNEL ((TARGET_COLLISION <= THREADS_PER_KERNEL))


#define IS_LESSOREQ_MAX_SIZE_for_MPs_PER_KERNEL ((MPs_PER_KERNEL <= MULTIPROCESSORS) ? TRUE : FALSE)
#define IS_MULT_OF_32_KERNEL_THREADBLOCK (() ? TRUE : FALSE)
#define IS_MULT_OF_8_ARBITRARY_MAX_BUFF_SIZE ((ARBITRARY_MAX_BUFF_SIZE % 8) ? TRUE : FALSE)



//===========================================================================================================
// CUDA ERROR CATCHING MACRO
// NOTE: ONLY RUN THIS BEFORE <<<>>> || AFTER FINAL SYNCHRONIZATION (IMPLICIT OR EXPLICIT)
//===========================================================================================================
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: Err no: %d, code: %s %s %d\n", code, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //CUDA_HASHING_CUDA_CONSTS_CUH
