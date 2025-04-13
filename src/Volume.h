#include "Eigen/Dense"
#include "sampler.h"


// create a function that takes in ray origin, ray direction, min and max of volume and returns whether the ray intersects the volume 

bool rayIntersectsVolume(const Vector3d& rayOrigin, const Vector3d& rayDirection, const Vector3d& min, const Vector3d& max, float& t) {
    // Initialize t values for near and far intersections
    double tNear = -std::numeric_limits<double>::infinity();
    double tFar = std::numeric_limits<double>::infinity();

    // Check each axis (x, y, z)
    for (int i = 0; i < 3; ++i) {
        if (rayDirection[i] == 0) {
            // Ray is parallel to the slab
            if (rayOrigin[i] < min[i] || rayOrigin[i] > max[i]) {
                return false;
            }
        } else {
            // Calculate intersection distances
            double t1 = (min[i] - rayOrigin[i]) / rayDirection[i];
            double t2 = (max[i] - rayOrigin[i]) / rayDirection[i];

            // Ensure t1 is the near intersection and t2 is the far intersection
            if (t1 > t2) {
                std::swap(t1, t2);
            }

            // Update near and far intersection distances
            tNear = std::max(tNear, t1);
            tFar = std::min(tFar, t2);

            // Check if the ray misses the volume
            if (tNear > tFar) {
                return false;
            }

            // Check if the volume is behind the ray
            if (tFar < 0) {
                return false;
            }
        }
    }

    t = tNear;

    // If we get here, the ray intersects the volume
    return true;
}

Vector3d cL(const Vector3d& location, float og1)
{
    // i made the lower bound of the volume -10 and pushed down the upper bound so that the volume remains the same size
    // so i need to convert the location to the original volume, og1 is the original lower bound and og2 is the original upper bound
    float difference = og1 + 10;

    return Vector3d(location[0], location[1] + difference, location[2]);
}

float BeerLambert(float absorption, float distance)
{
    return exp(-absorption * distance);
}

Vector3d getVolumeColor(const Vector3d& rayOrigin, const Vector3d& rayDirection, const Vector3d& m, const Vector3d& n, float og1, float og2, float minTemperature, float maxTemperature, float minDensity, float maxDensity, float t, std::vector<Light>& lights) 
{
    Vector3d min = m;
    Vector3d max = n;

    Vector3d intersection = rayOrigin + t * rayDirection;

    float opaqueVisibility = 1.0;
    float marchSize = 0.1f;
    float volumeDepth = 0.0f;

    float depth = m[2] - n[2];

    // grey color
    Vector3d color(.5, .5, .5);
    Vector3d accumulatedColor(0.0, 0.0, 0.0);
    float acc_density = 0.0;

    for(int i = 0; i < 20; i++)
    {
        
    }   

    if(acc_density < 1.0)
    {
        float c_t;
        accumulatedColor = accumulatedColor.cwiseProduct(CornellBox(rayOrigin, rayDirection, c_t, lights));
    }

    return accumulatedColor;
}

