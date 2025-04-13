#ifndef LIGHT_H
#define LIGHT_H

#include <Eigen/Dense>

using namespace Eigen;

struct Light
{
    Vector3d position;
    Vector3d color;
    float intensity;

    Light(const Vector3d& pos, const Vector3d& col, float intensity) : position(pos), color(col), intensity(intensity) {}
};

#endif // LIGHT_H

