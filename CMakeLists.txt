cmake_minimum_required(VERSION 3.10)
project(Explosion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# Make OpenVDB optional
option(USE_OPENVDB "Use OpenVDB for VDB file support" OFF)

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
)

if(USE_OPENVDB)
    add_definitions(-DUSE_OPENVDB)
    target_link_libraries(explosion 
        openvdb
        TBB::tbb
    )
endif() 