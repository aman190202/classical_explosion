#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>
#include "sampler.h"

class GridLookup {
private:
    // Grid dimensions and spacing
    Eigen::Vector3d minCorner;
    Eigen::Vector3d maxCorner;
    Eigen::Vector3d cellSize;
    int resolution;
    
    // 3D grid storage
    std::vector<float> temperatureGrid;
    std::vector<float> densityGrid;
    
    // Helper function to convert 3D position to grid index
    int positionToIndex(const Eigen::Vector3d& pos) const {
        Eigen::Vector3d normalized = (pos - minCorner).cwiseQuotient(maxCorner - minCorner);
        normalized = normalized.cwiseMax(Eigen::Vector3d::Zero()).cwiseMin(Eigen::Vector3d::Ones());
        
        int x = static_cast<int>(normalized.x() * (resolution - 1));
        int y = static_cast<int>(normalized.y() * (resolution - 1));
        int z = static_cast<int>(normalized.z() * (resolution - 1));
        
        return x + y * resolution + z * resolution * resolution;
    }
    
    // Helper function for trilinear interpolation
    float interpolate(const std::vector<float>& grid, const Eigen::Vector3d& pos) const {
        Eigen::Vector3d normalized = (pos - minCorner).cwiseQuotient(maxCorner - minCorner);
        normalized = normalized.cwiseMax(Eigen::Vector3d::Zero()).cwiseMin(Eigen::Vector3d::Ones());
        
        // Get the 8 surrounding grid points
        int x0 = static_cast<int>(normalized.x() * (resolution - 1));
        int y0 = static_cast<int>(normalized.y() * (resolution - 1));
        int z0 = static_cast<int>(normalized.z() * (resolution - 1));
        
        int x1 = std::min(x0 + 1, resolution - 1);
        int y1 = std::min(y0 + 1, resolution - 1);
        int z1 = std::min(z0 + 1, resolution - 1);
        
        // Get the fractional parts
        float xd = normalized.x() * (resolution - 1) - x0;
        float yd = normalized.y() * (resolution - 1) - y0;
        float zd = normalized.z() * (resolution - 1) - z0;
        
        // Get the 8 values
        float c000 = grid[x0 + y0 * resolution + z0 * resolution * resolution];
        float c100 = grid[x1 + y0 * resolution + z0 * resolution * resolution];
        float c010 = grid[x0 + y1 * resolution + z0 * resolution * resolution];
        float c110 = grid[x1 + y1 * resolution + z0 * resolution * resolution];
        float c001 = grid[x0 + y0 * resolution + z1 * resolution * resolution];
        float c101 = grid[x1 + y0 * resolution + z1 * resolution * resolution];
        float c011 = grid[x0 + y1 * resolution + z1 * resolution * resolution];
        float c111 = grid[x1 + y1 * resolution + z1 * resolution * resolution];
        
        // Interpolate along x
        float c00 = c000 * (1 - xd) + c100 * xd;
        float c01 = c001 * (1 - xd) + c101 * xd;
        float c10 = c010 * (1 - xd) + c110 * xd;
        float c11 = c011 * (1 - xd) + c111 * xd;
        
        // Interpolate along y
        float c0 = c00 * (1 - yd) + c10 * yd;
        float c1 = c01 * (1 - yd) + c11 * yd;
        
        // Interpolate along z
        return c0 * (1 - zd) + c1 * zd;
    }

public:
    GridLookup(const Eigen::Vector3d& min, const Eigen::Vector3d& max, int res = 100)
        : minCorner(min), maxCorner(max), resolution(res) {
        cellSize = (max - min) / (resolution - 1);
        temperatureGrid.resize(resolution * resolution * resolution);
        densityGrid.resize(resolution * resolution * resolution);
    }
    
    // Initialize the grid by sampling from the VDB file
    void initializeFromVDB() {
        #pragma omp parallel for collapse(3)
        for (int z = 0; z < resolution; ++z) {
            for (int y = 0; y < resolution; ++y) {
                for (int x = 0; x < resolution; ++x) {
                    Eigen::Vector3d pos = minCorner + Eigen::Vector3d(x, y, z).cwiseProduct(cellSize);
                    int index = x + y * resolution + z * resolution * resolution;
                    temperatureGrid[index] = volumeSampler->sampleTemperature(pos);
                    densityGrid[index] = volumeSampler->sampleDensity(pos);
                }
            }
        }
    }
    
    // Get temperature at a 3D position
    float getTemperature(const Eigen::Vector3d& pos) const {
        return interpolate(temperatureGrid, pos);
    }
    
    // Get density at a 3D position
    float getDensity(const Eigen::Vector3d& pos) const {
        return interpolate(densityGrid, pos);
    }
    
};

// Global grid lookup instance
std::unique_ptr<GridLookup> gridLookup;

// Initialize the grid lookup system
void initializeGridLookup(const Eigen::Vector3d& min, const Eigen::Vector3d& max, int resolution = 100) {
    gridLookup = std::make_unique<GridLookup>(min, max, resolution);
    gridLookup->initializeFromVDB();
}

// Get temperature at a 3D position
float getTemperature(const Eigen::Vector3d& pos) {
    if (!gridLookup) {
        throw std::runtime_error("Grid lookup not initialized");
    }
    return gridLookup->getTemperature(pos);
}

// Get density at a 3D position
float getDensity(const Eigen::Vector3d& pos) {
    if (!gridLookup) {
        throw std::runtime_error("Grid lookup not initialized");
    }
    return gridLookup->getDensity(pos);
}

