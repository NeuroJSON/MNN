# CPU

if(MNN_SUPPORT_RENDER)
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/compute/* ${CMAKE_CURRENT_LIST_DIR}/render/*)
else()
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/compute/*)
endif()
add_library(MNNCPU OBJECT ${MNN_CPU_SRC})

# Native 3D convolution executors: the hot inner loops benefit
# significantly from AVX2 + FMA (8-wide SIMD) vs the SSE4.1 default.
# Apply per-file so the rest of MNNCPU stays SSE-compatible.
#
# Gate on the target being x86: MNN_USE_SSE / MNN_AVX2 default ON even on
# arm64 (e.g. Apple Silicon), where -mavx2/-mfma are invalid options and
# /arch:AVX2 is meaningless. On non-x86 the Conv3D sources compile their
# scalar fallback (CPUConvolution3D.cpp guards the AVX2 path behind
# __AVX2__ && __FMA__), so no flags are needed there.
if (MNN_USE_SSE AND MNN_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x86|i[3-6]86")
    if(NOT MSVC)
        set_source_files_properties(
            "${CMAKE_CURRENT_LIST_DIR}/CPUConvolution3D.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/CPUDeconvolution3D.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma")
    else()
        set_source_files_properties(
            "${CMAKE_CURRENT_LIST_DIR}/CPUConvolution3D.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/CPUDeconvolution3D.cpp"
            PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    endif()
endif()

if (MNN_SUPPORT_BF16)
    include(${CMAKE_CURRENT_LIST_DIR}/bf16/CMakeLists.txt)
    list(APPEND MNN_TARGETS MNN_BF16)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_BF16>)
    target_compile_options(MNNCPU PRIVATE -DMNN_SUPPORT_BF16)
endif()
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCPU>)
list(APPEND MNN_TARGETS MNNCPU)


if(MNN_USE_SPARSE_COMPUTE)
  target_compile_options(MNNCPU PRIVATE -DMNN_USE_SPARSE_COMPUTE)
endif()

if(MNN_CPU_WEIGHT_DEQUANT_GEMM)
  target_compile_options(MNNCPU PRIVATE -DMNN_CPU_WEIGHT_DEQUANT_GEMM)
endif()

# X86_64 AVX/SSE
if (MNN_USE_SSE)
    include(${CMAKE_CURRENT_LIST_DIR}/x86_x64/CMakeLists.txt)
endif()

# AArch32/64 Assemblies
include(${CMAKE_CURRENT_LIST_DIR}/arm/CMakeLists.txt)
if(MNN_USE_RVV)
    include(${CMAKE_CURRENT_LIST_DIR}/riscv/CMakeLists.txt)
endif()
IF(NOT DEFINED IOS_ARCH)
    set(IOS_ARCH "")
ENDIF()

# ARM82 Assemblies
IF(MNN_ARM82)
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64")
        target_compile_options(MNNCPU PRIVATE -DENABLE_ARMV82)
        include(${CMAKE_CURRENT_LIST_DIR}/../arm82/CMakeLists.txt)
        list(APPEND MNN_TARGETS MNN_Arm82)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Arm82>)
    ENDIF()
ENDIF()