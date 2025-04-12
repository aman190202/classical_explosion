#ifndef BOX_H
#define BOX_H

#include "Eigen/Dense"

// Box struct for axis-aligned bounding box
struct Box {
    Eigen::Vector3d min;
    Eigen::Vector3d max;
    
    Box(const Eigen::Vector3d& min_, const Eigen::Vector3d& max_) : min(min_), max(max_) {}
};

// Ray-box intersection test
bool intersectBox(const Eigen::Vector3d& rayOrigin, const Eigen::Vector3d& rayDir,
                 const Box& box, double& tMin, double& tMax) {
    Eigen::Vector3d invDir(1.0/rayDir.x(), 1.0/rayDir.y(), 1.0/rayDir.z());
    
    Eigen::Vector3d t1 = (box.min - rayOrigin).cwiseProduct(invDir);
    Eigen::Vector3d t2 = (box.max - rayOrigin).cwiseProduct(invDir);
    
    Eigen::Vector3d tMin3 = t1.cwiseMin(t2);
    Eigen::Vector3d tMax3 = t1.cwiseMax(t2);
    
    tMin = tMin3.maxCoeff();
    tMax = tMax3.minCoeff();
    
    return tMax >= tMin && tMax > 0;
}

#endif // BOX_H