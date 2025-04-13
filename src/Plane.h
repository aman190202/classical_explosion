#ifndef PLANE_H
#define PLANE_H

#include "Eigen/Dense"

class Plane {
private:
    Eigen::Vector3d point;    // A point on the plane
    Eigen::Vector3d normal;   // Normal vector of the plane
    Eigen::Vector3d color;    // Color of the plane

public:
    Plane(const Eigen::Vector3d& p, const Eigen::Vector3d& n, const Eigen::Vector3d& c) 
        : point(p), normal(n.normalized()), color(c) {}

    // Ray-plane intersection
    bool intersect(const Eigen::Vector3d& rayOrigin, const Eigen::Vector3d& rayDir, double& t) const {
        double denom = normal.dot(rayDir);
        
        // If denominator is close to zero, ray is parallel to plane
        if (std::abs(denom) < 1e-6) {
            return false;
        }
        
        t = (point - rayOrigin).dot(normal) / denom;
        
        // Only return true if intersection is in front of ray origin
        return t >= 0;
    }

    // Get the point on the plane
    const Eigen::Vector3d& getPoint() const { return point; }
    
    // Get the normal vector
    const Eigen::Vector3d& getNormal() const { return normal; }

    // Get the color
    const Eigen::Vector3d& getColor() const { return color; }
};

#endif // PLANE_H 