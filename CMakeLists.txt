cmake_minimum_required(VERSION 3.10)
project(Explosion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get Conda environment path
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Using Conda environment at: ${CONDA_PREFIX}")
else()
    message(STATUS "CONDA_PREFIX not set, using Homebrew paths")
    set(CONDA_PREFIX "/opt/homebrew")
endif()

# Set paths for packages
set(TBB_INCLUDE_DIR "${CONDA_PREFIX}/include")
set(OpenMP_CXX_INCLUDE_DIR "${CONDA_PREFIX}/include")
set(OpenMP_CXX_LIBRARY "${CONDA_PREFIX}/Cellar/libomp/20.1.2/lib/libomp.dylib")

# Set OpenMP flags for macOS
if(APPLE)
    set(OpenMP_C_FLAGS "-Xclang -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${CONDA_PREFIX}/Cellar/libomp/20.1.2/lib/libomp.dylib")
endif()

# Find required packages
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

# Make OpenVDB optional but enabled by default
option(USE_OPENVDB "Use OpenVDB for VDB file support" ON)

if(USE_OPENVDB)
    set(OpenVDB_INCLUDE_DIR "${CONDA_PREFIX}/include/openvdb")
    set(OpenVDB_LIBRARY "${CONDA_PREFIX}/lib/libopenvdb.dylib")
    if(NOT EXISTS ${OpenVDB_INCLUDE_DIR} OR NOT EXISTS ${OpenVDB_LIBRARY})
        message(FATAL_ERROR "OpenVDB not found. Please install it using: brew install openvdb")
    endif()
endif()

# Add all source files
set(SOURCES
    main.cpp
    src/sampler.cpp
    src/grid_lookup.cpp
)

# Add executable
add_executable(explosion ${SOURCES})

# Include directories
target_include_directories(explosion PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/Eigen
    ${TBB_INCLUDE_DIRS}
    ${CONDA_PREFIX}/Cellar/libomp/20.1.2/include
)

# Link directories
link_directories(${CONDA_PREFIX}/lib ${CONDA_PREFIX}/Cellar/libomp/20.1.2/lib)

# Link libraries
target_link_libraries(explosion PRIVATE
    OpenMP::OpenMP_CXX
    TBB::tbb
)

if(USE_OPENVDB)
    target_compile_definitions(explosion PRIVATE USE_OPENVDB)
    target_include_directories(explosion PRIVATE ${OpenVDB_INCLUDE_DIR})
    target_link_libraries(explosion PRIVATE ${OpenVDB_LIBRARY})
endif() 