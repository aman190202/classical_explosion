cmake_minimum_required(VERSION 3.10)
project(Explosion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# OpenMP configuration for macOS
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_CXX_LIBRARIES "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    find_package(OpenMP REQUIRED)
else()
    find_package(OpenMP REQUIRED)
endif()

# Make OpenVDB optional but enabled by default
option(USE_OPENVDB "Use OpenVDB for VDB file support" ON)

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
    /opt/homebrew/include
)

# Link directories
link_directories(/opt/homebrew/lib)

# Link libraries
target_link_libraries(explosion 
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

if(USE_OPENVDB)
    add_definitions(-DUSE_OPENVDB)
    target_link_libraries(explosion 
        openvdb
        TBB::tbb
    )
endif() 