#include "sampler.h"
#include <iostream>
#include <stdexcept>

// Initialize the global sampler
std::unique_ptr<VDBVolumeSampler> volumeSampler;

void initializeSampler(const std::string& vdbFilePath) {
    try {
        volumeSampler = std::make_unique<VDBVolumeSampler>(vdbFilePath);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize sampler: " + std::string(e.what()));
    }
}

Eigen::Vector3d densitySampler(const Eigen::Vector3d& position) {
    if (!volumeSampler || !volumeSampler->isInitialized()) {
        return Eigen::Vector3d::Zero();
    }
    float density = volumeSampler->sampleDensity(position);
    // Return density as a grayscale color
    return Eigen::Vector3d(density, density, density);
}

Eigen::Vector3d colorSampler(const Eigen::Vector3d& position) {
    if (!volumeSampler || !volumeSampler->isInitialized()) {
        return Eigen::Vector3d::Zero();
    }
    float temperature = volumeSampler->sampleTemperature(position);
    
    // Map temperature to color using a simple color ramp
    // Cold (blue) -> Hot (red)
    float t = std::min(std::max(temperature, 0.0f), 1.0f);
    
    // Blue to red color ramp
    Eigen::Vector3d color;
    color.x() = t;                    // Red increases with temperature
    color.y() = 0.0;                  // No green
    color.z() = 1.0 - t;              // Blue decreases with temperature
    
    return color;
} 