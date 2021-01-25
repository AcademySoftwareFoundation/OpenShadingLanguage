// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#define CUDA_CHECK(call)                                              \
    {                                                                 \
        cudaError_t error = call;                                     \
        if (error != cudaSuccess) {                                   \
            std::stringstream ss;                                     \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << cudaGetErrorString(error) << "' (" __FILE__ << ":"  \
               << __LINE__ << ")\n";                                  \
            fprintf(stderr, "[CUDA ERROR]  %s", ss.str().c_str());    \
            exit(1);                                                  \
        }                                                             \
    }

#define CU_CHECK(call)                                                        \
    {                                                                         \
        CUresult error = call;                                                \
        if (error != CUDA_SUCCESS) {                                          \
            std::stringstream ss;                                             \
            const char* estr;                                                 \
            cuGetErrorString(error, &estr);                                   \
            ss << "CUDA call (" << #call << " ) failed with error: '" << estr \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                 \
            fprintf(stderr, "[CU ERROR]  %s", ss.str().c_str());              \
            exit(1);                                                          \
        }                                                                     \
    }

#define CUDA_SYNC_CHECK()                                                    \
    {                                                                        \
        cudaDeviceSynchronize();                                             \
        cudaError_t error = cudaGetLastError();                              \
        if (error != cudaSuccess) {                                          \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                              \
            exit(1);                                                         \
        }                                                                    \
    }

#define NVRTC_CHECK(call)                                              \
    {                                                                  \
        nvrtcResult error = call;                                      \
        if (error != NVRTC_SUCCESS) {                                  \
            std::stringstream ss;                                      \
            ss << "NVRTC call (" << #call << " ) failed with error: '" \
               << nvrtcGetErrorString(error) << "' (" __FILE__ << ":"  \
               << __LINE__ << ")\n";                                   \
            fprintf(stderr, "[NVRTC ERROR]  %s", ss.str().c_str());    \
            exit(1);                                                   \
        }                                                              \
    }
