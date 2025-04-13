#include <unordered_set>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <string>
#include "sampler.h"
#include "grid_lookup.h"
#include "Plane.h"
#include "Light.h"

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


Vector3d CornellBox(const Vector3d& rayOrigin, const Vector3d& rayDir, float& t, std::vector<Light>& lights)
{

    // Constants for Cornell box
    // const double BOX_SIZE = 1.0;  // Reduced from 10.0 to 5.0
    // const Vector3d BOX_MIN(-BOX_SIZE/2, -BOX_SIZE/2, -BOX_SIZE/2);
    // const Vector3d BOX_MAX(BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2);

    // Wall colors
    const Vector3d RED(0.8, 0.1, 0.1);
    const Vector3d GREEN(0.1, 0.8, 0.1);
    const Vector3d WHITE(0.8, 0.8, 0.8);
    const Vector3d LIGHT_COLOR(1.0, 1.0, 0.9);

    Vector3d finalColor(0.0, 0.0, 0.0);

    // Create Cornell box walls
    // FLOOR
    Plane floor(Vector3d(0, -10, 0), Vector3d(0, 1, 0), WHITE);
    // CEILING
    Plane ceiling(Vector3d(0, 10, 0), Vector3d(0, -1, 0), WHITE);
    // BACK WALL
    Plane back(Vector3d(0, 0, -5), Vector3d(0, 0, 1), WHITE);
    // Left wall (red
    Plane left(Vector3d(-5, 0, 0), Vector3d(1, 0, 0), RED);
    // Right wall (green)
    Plane right(Vector3d(5, 0, 0), Vector3d(-1, 0, 0), GREEN);


    double minT = std::numeric_limits<double>::infinity();
    const Plane* closestPlane = nullptr;

    Vector3d color;

    // Check intersections with all walls
    double intersectionT;
    if (floor.intersect(rayOrigin, rayDir, intersectionT) && intersectionT < minT) {
        minT = intersectionT;
        closestPlane = &floor;
    }
    if (ceiling.intersect(rayOrigin, rayDir, intersectionT) && intersectionT < minT) {
        minT = intersectionT;
        closestPlane = &ceiling;
    }
    if (back.intersect(rayOrigin, rayDir, intersectionT) && intersectionT < minT) {
        minT = intersectionT;
        closestPlane = &back;
    }
    if (left.intersect(rayOrigin, rayDir, intersectionT) && intersectionT < minT) {
        minT = intersectionT;
        closestPlane = &left;
    }
    if (right.intersect(rayOrigin, rayDir, intersectionT) && intersectionT < minT) {
        minT = intersectionT;
        closestPlane = &right;
    }

    if (closestPlane) {
        // Calculate lighting
        Vector3d intersectionPoint = rayOrigin + minT * rayDir;
        Vector3d normal = closestPlane->getNormal();
        Vector3d baseColor = closestPlane->getColor();
        Vector3d finalColor(0.0, 0.0, 0.0);

        // Add diffuse lighting from all lights and distance attenuation ; as attenuation is close to 0, pixels will be black
        // randomly select 10 lights to light the intersection point
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, lights.size() - 1);
        
        #pragma omp parallel for
        for (int i = 0; i < lights.size()/100; i++) {
            int randomIndex = dis(gen);
            const auto& light = lights[randomIndex];    
            Vector3d lightDir = (light.position - intersectionPoint).normalized();
            float diffuse = std::max(0.0f, static_cast<float>(normal.dot(lightDir)));
            // Add light contribution to base color
            float distance = (light.position - intersectionPoint).norm();
            float attenuation = 1.0f / (distance * distance);
            finalColor += baseColor.cwiseProduct(light.color) * light.intensity * diffuse * attenuation;
        }


        return finalColor;
    }

    return finalColor;
}

