#include "Eigen/Dense"
#include "sampler.h"
#include "Light.h"
#include <algorithm>
#include <iostream>
#include <omp.h>


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
    float difference = og1 + 30;

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



    // grey color
    Vector3d color(.5, .5, .5);
    Vector3d accumulatedColor(0.0, 0.0, 0.0);
    float acc_density = 0.0;

    float step_size = 0.1;
    
    for(int i = 1; i < 100; i++)
    {
        Vector3d position = intersection + i * step_size * rayDirection;

        position = cL(position, og1);
        float temperature = getTemperature(position);

        acc_density += temperature;
    }


    if(acc_density < 0.1)
    {
        float c_t;
        accumulatedColor =  CornellBox(rayOrigin, rayDirection, c_t, lights);
    }
    else
    {
        accumulatedColor = acc_density * Vector3d(.5, .5, .5);
    }

    return accumulatedColor;
}

std::vector<Light> fillLights(Vector3d min, Vector3d max)
{
    int m1 = min[0];
    int m2 = min[1];
    int m3 = min[2];
    int M1 = max[0];
    int M2 = max[1];
    int M3 = max[2];
    std::vector<Light> lights;

    #pragma omp parallel
    {
        std::vector<Light> local_lights;
        #pragma omp for collapse(3) nowait
        for(int i = m1; i <= M1; i++)
        {
            for(int j = m2; j <= M2; j++)
            {
                for(int k = m3; k <= M3; k++)
                {
                    float temperature = getTemperature(Vector3d(i, j, k));
                    if(temperature < 0.6f)
                        continue;
                    Vector3d color;
                    // Enhanced color mapping for explosion simulation
                    // Temperature ranges from 0.6 to 1.3
                    if (temperature < 0.8f) {
                        // 0.6-0.8: Dark red to bright red
                        float t = (temperature - 0.6f) / 0.2f;
                        color = Vector3d(0.5f + 0.5f * t, 0.0f, 0.0f);
                    }
                    else if (temperature < 0.9f) {
                        // 0.8-0.9: Red to orange
                        float t = (temperature - 0.8f) / 0.1f;
                        color = Vector3d(1.0f, t * 0.7f, 0.0f);
                    }
                    else if (temperature < 1.0f) {
                        // 0.9-1.0: Orange to yellow
                        float t = (temperature - 0.9f) / 0.1f;
                        color = Vector3d(1.0f, 0.7f + 0.3f * t, t * 0.5f);
                    }
                    else {
                        // 1.0-1.3: Yellow to white hot
                        float t = (temperature - 1.0f) / 0.3f;
                        if (t > 1.0f) t = 1.0f; // Manual clamping
                        color = Vector3d(1.0f, 1.0f, 0.5f + 0.5f * t);
                    }
                    // Intensity also increases with temperature
                    float intensity = 0.5f + 0.5f * temperature;
                    if (intensity > 1.0f) intensity = 1.0f; // Manual clamping
                    local_lights.push_back(Light(Vector3d(i, j, k), color, intensity));
                }
            }
        }
        #pragma omp critical
        {
            lights.insert(lights.end(), local_lights.begin(), local_lights.end());
        }
    }

    std::cout<<" "<<lights.size()<<" ";
    return lights;
}
