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

Vector3d interpolateColor(float temperature) 
{
    // temperature is already normalized between 0 and 1
    // Interpolate between red (1,0,0) and white (1,1,1)
    float r = 1.0f;  // Red component stays at 1
    float g = temperature;  // Green and blue interpolate from 0 to 1
    float b = temperature;
    return Vector3d(r, g, b);
}

// Custom equality function for Vector3d
struct Vec3Equal {
    bool operator()(const Vector3d& v1, const Vector3d& v2) const {
        return v1.x() == v2.x() && v1.y() == v2.y() && v1.z() == v2.z();
    }
};

void getBoundingBox(const std::string& vdbFilePath, Vector3d& min_density, Vector3d& max_density, Vector3d& min_temperature, Vector3d& max_temperature) {
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
    openvdb::FloatGrid::Ptr temperatureGrid;
    for (openvdb::GridBase::Ptr grid : *grids) {
        if (grid->getName() == "density") {
            densityGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid);
            continue;
        }
        if (grid->getName() == "temperature") {
            temperatureGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid);
            continue;
        }
    }
    
    if (!densityGrid) {
        throw std::runtime_error("Could not find density grid in VDB file");
    }
    if (!temperatureGrid) {
        throw std::runtime_error("Could not find temperature grid in VDB file");
    }
    
    // Get the bounding box in world space
    openvdb::math::CoordBBox bbox_density = densityGrid->evalActiveVoxelBoundingBox();
    openvdb::math::Vec3d worldMin_density = densityGrid->indexToWorld(bbox_density.min());
    openvdb::math::Vec3d worldMax_density = densityGrid->indexToWorld(bbox_density.max());

    openvdb::math::CoordBBox bbox_temperature = temperatureGrid->evalActiveVoxelBoundingBox();
    openvdb::math::Vec3d worldMin_temperature = temperatureGrid->indexToWorld(bbox_temperature.min());
    openvdb::math::Vec3d worldMax_temperature = temperatureGrid->indexToWorld(bbox_temperature.max());
    
    // Convert to Eigen vectors
    min_density = Vector3d(worldMin_density.x(), worldMin_density.y(), worldMin_density.z());
    max_density = Vector3d(worldMax_density.x(), worldMax_density.y(), worldMax_density.z());

    min_temperature = Vector3d(worldMin_temperature.x(), worldMin_temperature.y(), worldMin_temperature.z());
    max_temperature = Vector3d(worldMax_temperature.x(), worldMax_temperature.y(), worldMax_temperature.z());
    
    // Close the file
    file.close();
#else
    throw std::runtime_error("OpenVDB support not enabled");
#endif
}



bool checkCornellBoxIntersect(const Vector3d& rayOrigin, const Vector3d& rayDir, float& t)
{
    // Constants for Cornell box
    // Wall colors
    const Vector3d RED(0.8, 0.1, 0.1);
    const Vector3d GREEN(0.1, 0.8, 0.1);
    const Vector3d WHITE(0.8, 0.8, 0.8);
    const Vector3d LIGHT_COLOR(1.0, 1.0, 0.9);

    Vector3d finalColor(0.0, 0.0, 0.0);

    // Create Cornell box walls
    // FLOOR
    Plane floor(Vector3d(0, -30, 0), Vector3d(0, 1, 0), WHITE);
    // CEILING
    Plane ceiling(Vector3d(0, 30, 0), Vector3d(0, -1, 0), WHITE);
    // BACK WALL
    Plane back(Vector3d(0, 0, -15), Vector3d(0, 0, 1), WHITE);
    // Left wall (red
    Plane left(Vector3d(-15, 0, 0), Vector3d(1, 0, 0), RED);
    // Right wall (green)
    Plane right(Vector3d(15, 0, 0), Vector3d(-1, 0, 0), GREEN);

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
        t = minT;
        return true;
    }

    return false;
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
    Plane floor(Vector3d(0, -30, 0), Vector3d(0, 1, 0), WHITE);
    // CEILING
    Plane ceiling(Vector3d(0, 30, 0), Vector3d(0, -1, 0), WHITE);
    // BACK WALL
    Plane back(Vector3d(0, 0, -15), Vector3d(0, 0, 1), WHITE);
    // Left wall (red
    Plane left(Vector3d(-15, 0, 0), Vector3d(1, 0, 0), RED);
    // Right wall (green)
    Plane right(Vector3d(15, 0, 0), Vector3d(-1, 0, 0), GREEN);


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
        for (int i = 0; i <= lights.size()/100; i++) {
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

