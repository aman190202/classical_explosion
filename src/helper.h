#include <unordered_set>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <string>
#include "sampler.h"
#include "grid_lookup.h"

using namespace Eigen;
struct Vec3Hash {
    std::size_t operator()(const Vector3d& v) const {
        std::size_t h1 = std::hash<double>()(v.x());
        std::size_t h2 = std::hash<double>()(v.y());
        std::size_t h3 = std::hash<double>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Custom equality function for Vector3d
struct Vec3Equal {
    bool operator()(const Vector3d& v1, const Vector3d& v2) const {
        return v1.x() == v2.x() && v1.y() == v2.y() && v1.z() == v2.z();
    }
};

void getBoundingBox(const std::string& vdbFilePath, Vector3d& min, Vector3d& max) {
#ifdef USE_OPENVDB
    // Initialize OpenVDB
    openvdb::initialize();
    
    // Open the VDB file
    openvdb::io::File file(vdbFilePath);
    file.open();
    
    // Get the grids
    openvdb::GridPtrVecPtr grids = file.getGrids();
    
    // Find density grid (assuming it has the same bounds as temperature grid)
    openvdb::FloatGrid::Ptr densityGrid;
    for (openvdb::GridBase::Ptr grid : *grids) {
        if (grid->getName() == "density") {
            densityGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid);
            break;
        }
    }
    
    if (!densityGrid) {
        throw std::runtime_error("Could not find density grid in VDB file");
    }
    
    // Get the bounding box in world space
    openvdb::math::CoordBBox bbox = densityGrid->evalActiveVoxelBoundingBox();
    openvdb::math::Vec3d worldMin = densityGrid->indexToWorld(bbox.min());
    openvdb::math::Vec3d worldMax = densityGrid->indexToWorld(bbox.max());
    
    // Convert to Eigen vectors
    min = Vector3d(worldMin.x(), worldMin.y(), worldMin.z());
    max = Vector3d(worldMax.x(), worldMax.y(), worldMax.z());
    
    // Close the file
    file.close();
#else
    throw std::runtime_error("OpenVDB support not enabled");
#endif
}

void populateSetsFromVDB(const std::string& vdbFilePath,
                        std::unordered_set<Vector3d, Vec3Hash, Vec3Equal>& densityLocations,
                        std::unordered_set<Vector3d, Vec3Hash, Vec3Equal>& temperatureLocations) {
    // Initialize the sampler
    initializeSampler(vdbFilePath);
    
    // Get the bounding box of the volume
    Vector3d min, max;
    try {
        getBoundingBox(vdbFilePath, min, max);
        std::cout << "Bounding Box Coordinates:" << std::endl;
        std::cout << "Min: (" << min.x() << ", " << min.y() << ", " << min.z() << ")" << std::endl;
        std::cout << "Max: (" << max.x() << ", " << max.y() << ", " << max.z() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error getting bounding box: " << e.what() << std::endl;
        return;
    }
    
    // Initialize grid lookup
    initializeGridLookup(min, max);
    
    // Sample points in the volume
    for (double x = min.x(); x <= max.x(); x += 0.1) {
        for (double y = min.y(); y <= max.y(); y += 0.1) {
            for (double z = min.z(); z <= max.z(); z += 0.1) {
                Vector3d position(x, y, z);
                
                // Get values at this position
                float temperature, density;
                getValues(position, temperature, density);
                
                // If there's a non-zero value, add to the appropriate set
                if (density > 0.0) {
                    densityLocations.insert(position);
                }
                if (temperature > 0.0) {
                    temperatureLocations.insert(position);
                }
            }
        }
    }
}