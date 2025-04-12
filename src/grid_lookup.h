#pragma once

#include <Eigen/Dense>

// Initialize the grid lookup system
void initializeGridLookup(const Eigen::Vector3d& min, const Eigen::Vector3d& max, int resolution = 100);

// Get temperature at a 3D position
float getTemperature(const Eigen::Vector3d& pos);

// Get density at a 3D position
float getDensity(const Eigen::Vector3d& pos);

// Get both temperature and density at a 3D position
void getValues(const Eigen::Vector3d& pos, float& temperature, float& density); 