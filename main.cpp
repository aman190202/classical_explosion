#include <unordered_set>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <string>
#include <omp.h>
#include "src/sampler.h"
#include "src/grid_lookup.h"
#include "src/helper.h"
#include "src/Image.h"
#include "src/Camera.h"
#include "src/Box.h"

using namespace Eigen;

struct Light
{
    Vector3d position;
    Vector3d color;
    
    Light(const Vector3d& pos, const Vector3d& col) : position(pos), color(col) {}
};

// Constants for volumetric rendering
const double ABSORPTION_COEFFICIENT = 0.5;
const double SCATTERING_COEFFICIENT = 0.3;
const double MARCH_SIZE = 0.1;
const int MAX_VOLUME_MARCH_STEPS = 100;
const double MIN_TRANSMITTANCE = 0.01;

// Beer-Lambert law for light absorption
double beerLambert(double absorption, double distance) {
    return exp(-absorption * distance);
}

// Light attenuation based on distance
double lightAttenuation(double distance) {
    return 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vdb_file_path>" << std::endl;
        return 1;
    }
    
    std::string vdbFilePath = argv[1];
    
    // Unordered sets for density and temperature locations
    std::unordered_set<Vector3d, Vec3Hash, Vec3Equal> densityLocations;
    std::unordered_set<Vector3d, Vec3Hash, Vec3Equal> temperatureLocations;
    
    // Populate sets from VDB file
    populateSetsFromVDB(vdbFilePath, densityLocations, temperatureLocations);
    
    // Print the number of points in each set
    std::cout << "Number of density points: " << densityLocations.size() << std::endl;
    std::cout << "Number of temperature points: " << temperatureLocations.size() << std::endl;


    // check for density and temperature at a specific point
    Vector3d point(0.0, 0.0, 0.0);
    float density, temperature;
    getValues(point, temperature, density);
    
    // get maximum density and temperature and minimum density and temperature
    float maxDensity = 0.0;
    float minDensity = 1.0;
    float maxTemperature = 0.0;
    float minTemperature = 1.0;     

    for (const auto& location : densityLocations) {
        float density, temperature;
        getValues(location, temperature, density);
        maxDensity = std::max(maxDensity, density);
        minDensity = std::min(minDensity, density);
    }

    for (const auto& location : temperatureLocations)
    {
        float temperature;
        getValues(location, temperature, density);
        maxTemperature = std::max(maxTemperature, temperature);
        minTemperature = std::min(minTemperature, temperature);
    }

    std::cout << "Max density: " << maxDensity << std::endl;
    std::cout << "Min density: " << minDensity << std::endl;
    std::cout << "Max temperature: " << maxTemperature << std::endl;
    std::cout << "Min temperature: " << minTemperature << std::endl;

    // get bounding box of the volume
    Vector3d min, max;
    getBoundingBox(vdbFilePath, min, max);
    std::cout << "Bounding Box Coordinates:" << std::endl;
    std::cout << "Min: (" << min.x() << ", " << min.y() << ", " << min.z() << ")" << std::endl;
    std::cout << "Max: (" << max.x() << ", " << max.y() << ", " << max.z() << ")" << std::endl;

    std::vector<Light> lights;
    // go through all temperature points and get the light position and color and add to the vector, where color is white for 1 and red for 0.5 and if temperature is less than 0.5 then dont add it to the vector
    for (const auto& location : temperatureLocations)
    {
        float temperature;
        getValues(location, temperature, density);
        temperature = (temperature - minTemperature) / (maxTemperature - minTemperature);

        if(temperature > 0.5)
        {
            Vector3d loc{location.x(), location.y(), location.z()};
            // interpolate between white and red based on temperature
            Vector3d color = Vector3d(1.0, 0.0, 0.0) * temperature + Vector3d(0.0, 0.0, 1.0) * (1.0 - temperature);
            lights.push_back(Light(loc, color));
        }
    }

    std::cout << "Number of lights: " << lights.size() << std::endl;


    // Create image
    const int width = 1920;
    const int height = 1080;
    Image image(width, height);
    
    // Create camera
    Camera camera(
        Eigen::Vector3d(0, 25, 15),  // Position camera above and behind the ground
        Eigen::Vector3d(0, 0, 0),    // Look at the origin
        Eigen::Vector3d(0, 1, 0),    // Up vector
        60.0,                        // Field of view
        static_cast<double>(width) / height, // Aspect ratio
        0.1,                         // Near plane
        1000.0                       // Far plane
    );

    // Set number of threads (optional, OpenMP will use all available by default)
    // omp_set_num_threads(4);

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) 
    {
        for (int x = 0; x < width; ++x) 
        {
            double u = static_cast<double>(x) / width;
            double v = 1.0 - static_cast<double>(y) / height;
            Eigen::Vector3d rayDir = camera.generateRay(u, v);
            Eigen::Vector3d rayOrigin = camera.getPosition();
            Eigen::Vector3d finalColor(0.0, 0.0, 0.0);
            float acc_density = 0.0;
            double tMin, tMax;
            Box box(min, max);
            if (intersectBox(rayOrigin, rayDir, box, tMin, tMax)) 
            {
                Eigen::Vector3d hitPoint = rayOrigin + tMin * rayDir;
                Eigen::Vector3d exitPoint = rayOrigin + tMax * rayDir;

                // sample ray while transmittance is significant
                double transmittance = 1.0;
                Eigen::Vector3d accumulatedColor(0.0, 0.0, 0.0);
                double volumeDepth = 0.0;
                
                while (transmittance > MIN_TRANSMITTANCE && volumeDepth < (tMax - tMin)) 
                {
                    double previousTransmittance = transmittance;
                    Eigen::Vector3d samplePosition = hitPoint + volumeDepth * (exitPoint - hitPoint).normalized();
                    getValues(samplePosition, temperature, density);
                    
                    //normalize density and temperature
                    density = (density - minDensity) / (maxDensity - minDensity);
                    temperature = (temperature - minTemperature) / (maxTemperature - minTemperature);
                    
                    // Calculate absorption using Beer-Lambert law
                    double absorption = beerLambert(ABSORPTION_COEFFICIENT * density, MARCH_SIZE);
                    transmittance *= absorption;
                    
                    // Calculate light contribution from each light source
                    if (density > 0.0) {
                        Eigen::Vector3d lightContribution(0.0, 0.0, 0.0);
                        
                        // Add contribution from each light
                        for (const auto& light : lights) {
                            Eigen::Vector3d lightDir = (light.position - samplePosition).normalized();
                            double lightDistance = (light.position - samplePosition).norm();
                            double attenuation = lightAttenuation(lightDistance);
                            
                            // Simple phase function (isotropic scattering)
                            double phase = 1.0 / (4.0 * M_PI);
                            
                            // Calculate light contribution
                            lightContribution += light.color * attenuation * phase * density;
                        }
                        
                        // Add ambient light contribution
                        Eigen::Vector3d ambientLight(0.1, 0.1, 0.1);
                        lightContribution += ambientLight * density;
                        
                        // Add the light contribution to the accumulated color
                        double absorptionFromMarch = previousTransmittance - transmittance;
                        accumulatedColor += absorptionFromMarch * lightContribution;
                    }
                    
                    volumeDepth += MARCH_SIZE;
                }
                
                finalColor = accumulatedColor;
            }
            #pragma omp critical
            {
                image.setPixel(x, y, finalColor.x(), finalColor.y(), finalColor.z());
            }
        }
    }
    
    // Save image to file with vdb file name - remove the path and .vdb extension   
    std::string fileName = vdbFilePath.substr(vdbFilePath.find_last_of("/") + 1);
    fileName = fileName.substr(0, fileName.find_last_of("."));
    image.savePPM("output/" + fileName + ".ppm");
    
    return 0;
}
