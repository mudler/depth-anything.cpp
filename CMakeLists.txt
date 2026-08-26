cmake_minimum_required(VERSION 3.18)
project(depth_anything VERSION 0.0.1 LANGUAGES C CXX)

# Absolute path to this project's own root. Use this (not CMAKE_SOURCE_DIR)
# for every in-tree path so the build works both standalone AND when embedded
# via add_subdirectory() (e.g. the LocalAI backend), where CMAKE_SOURCE_DIR
# points at the parent project instead. CMake propagates this to subdirs.
set(DA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DA_BUILD_TESTS  "Build tests"                              OFF)
option(DA_BUILD_CLI    "Build da3-cli"                            ON)
option(DA_SHARED       "Build libdepthanything as a shared lib"   OFF)
option(DA_GGML_CUDA    "ggml CUDA backend"                        OFF)
option(DA_GGML_METAL   "ggml Metal backend"                       OFF)
option(DA_GGML_VULKAN  "ggml Vulkan backend"                      OFF)

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
# tinyBLAS (llamafile) — a hand-tuned AVX-512/AVX2 sgemm/hgemm compiled INTO
# ggml-cpu (static, no new runtime dep). ~28% faster inference on the matmul/
# conv-bound forward, parity-preserving. Toggle with -DDA_GGML_LLAMAFILE=OFF.
option(DA_GGML_LLAMAFILE "ggml tinyBLAS (llamafile) matmul kernels" ON)
set(GGML_LLAMAFILE ${DA_GGML_LLAMAFILE} CACHE BOOL "" FORCE)
set(GGML_CUDA   ${DA_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL  ${DA_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN ${DA_GGML_VULKAN} CACHE BOOL "" FORCE)
# CUDA target arch is left to ggml's own ggml-cuda/CMakeLists, which picks a
# sensible default (a virtual-arch list covering Maxwell..Blackwell, incl.
# 120a/121a-real for GB10) when CMAKE_CUDA_ARCHITECTURES is unset, and guards
# against a GPU-less builder. Do NOT force "native" here: it needs CMake >= 3.24
# and a GPU present at configure time, so it breaks cross/CI builds (e.g. the
# l4t/Jetson image on CMake 3.22 yields an empty "compute_" arch). For a local
# GPU-targeted build pass -DCMAKE_CUDA_ARCHITECTURES=native explicitly.
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

set(DA_SOURCES
    src/model_loader.cpp
    src/backend.cpp
    src/image_io.cpp
    src/preprocess.cpp
    src/rope2d.cpp
    src/uv_posembed.cpp
    src/attention.cpp
    src/vit_block.cpp
    src/dino_backbone.cpp
    src/dpt_blocks.cpp
    src/winograd.cpp
    src/dpt_head.cpp
    src/gs_head.cpp
    src/gs_adapter.cpp
    src/ply_export.cpp
    src/reconstruct.cpp
    src/glb_export.cpp
    src/colmap_export.cpp
    src/cam_pose.cpp
    src/linalg.cpp
    src/sim3.cpp
    src/stream.cpp
    src/fuse.cpp
    src/tsdf.cpp
    src/voxel_mesh.cpp
    src/icp.cpp
    src/posegraph.cpp
    src/ray_pose.cpp
    src/nested.cpp
    src/depth_export.cpp
    src/pose_export.cpp
    src/engine.cpp
    src/da_capi.cpp
    src/cli.cpp
    src/quantize.cpp
)
if(DA_SHARED)
    add_library(depthanything SHARED ${DA_SOURCES})
else()
    add_library(depthanything STATIC ${DA_SOURCES})
endif()
target_include_directories(depthanything
    PUBLIC  ${DA_ROOT}/include
    PRIVATE ${DA_ROOT}/src ${DA_ROOT}/third_party ${DA_ROOT}/third_party/stb)

# Winograd's hot GEMV uses AVX-512 intrinsics. ggml gets these via GGML_NATIVE,
# but our library target does not, so enable them for just winograd.cpp when the
# host compiler supports them (falls back to a scalar path otherwise).
include(CheckCXXCompilerFlag)
if(MSVC)
    # MSVC spells it /arch:AVX512 (implies BW/DQ/VL + FMA) and defines
    # __AVX512F__ under it, so the same intrinsics path lights up.
    check_cxx_compiler_flag("/arch:AVX512" DA_HAS_AVX512F)
    if(DA_HAS_AVX512F)
        set_source_files_properties(src/winograd.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX512")
    endif()
else()
    check_cxx_compiler_flag("-mavx512f" DA_HAS_AVX512F)
    if(DA_HAS_AVX512F)
        set_source_files_properties(src/winograd.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma")
    endif()
endif()
target_link_libraries(depthanything PUBLIC ggml)

# OpenMP for the single-threaded CPU geometry post-processing (surface fusion,
# per-seam ICP, normal estimation). ggml already pulls OpenMP into the toolchain;
# link it into OUR target too so the #pragma omp loops in fuse.cpp/icp.cpp use all
# cores instead of one. Optional: without it the pragmas compile to no-ops.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(depthanything PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "depth-anything.cpp: OpenMP enabled for CPU geometry")
else()
    message(STATUS "depth-anything.cpp: OpenMP NOT found — CPU geometry stays single-threaded")
endif()

if(DA_BUILD_CLI AND EXISTS ${DA_ROOT}/examples/cli/CMakeLists.txt)
    add_subdirectory(examples/cli)
endif()
if(DA_BUILD_TESTS AND EXISTS ${DA_ROOT}/tests/CMakeLists.txt)
    enable_testing()
    add_subdirectory(tests)
endif()
message(STATUS "depth-anything.cpp configured")
