#include <unordered_set>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <string>
#include "src/sampler.h"
#include "src/grid_lookup.h"
#include "src/helper.h"
#include "src/Image.h"
#include "src/Camera.h"
#include "src/Box.h"

using namespace Eigen;

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

    // // normalize the density and temperature between 0 and 1
    // for (const auto& location : densityLocations) {
    //     float density, temperature;
    //     getValues(location, temperature, density);
    //     density = (density - minDensity) / (maxDensity - minDensity);
    //     temperature = (temperature - minTemperature) / (maxTemperature - minTemperature);
    // }

    // // print the normalized density and temperature at a specific point
    // getValues(point, temperature, density);
    // std::cout << "Normalized density at (0,0,0): " << density << std::endl;
    // std::cout << "Normalized temperature at (0,0,0): " << temperature << std::endl;


    // // create a vector<vector3f> of points whose normalized temperature is greater than 0.5
    // std::vector<Vector3d> points;
    // for (const auto& location : temperatureLocations) {
    //     float temperature;
    //     getValues(location, temperature, density);
    //     density = (density - minDensity) / (maxDensity - minDensity);
    //     temperature = (temperature - minTemperature) / (maxTemperature - minTemperature);
    //     if (temperature > 0.5) {
    //         points.push_back(location);
    //     }
    // }
    // std::cout << "Number of points whose normalized temperature is greater than 0.5: " << points.size() << std::endl;



    // Create image
    const int width = 800;
    const int height = 600;
    Image image(width, height);
    
    // Create camera
    Camera camera(
        Eigen::Vector3d(0, 20, 50),  // Position camera above and behind the ground
        Eigen::Vector3d(0, 0, 0),    // Look at the origin
        Eigen::Vector3d(0, 1, 0),    // Up vector
        60.0,                        // Field of view
        static_cast<double>(width) / height, // Aspect ratio
        0.1,                         // Near plane
        1000.0                       // Far plane
    );


    for (int y = 0; y < height; ++y) 
    {
        for (int x = 0; x < width; ++x) 
        {
            double u = static_cast<double>(x) / width;
            double v = 1.0 - static_cast<double>(y) / height;
            Eigen::Vector3d rayDir = camera.generateRay(u, v);
            Eigen::Vector3d rayOrigin = camera.getPosition();
            Eigen::Vector3d finalColor(0.0, 0.0, 0.0);
            double tMin, tMax;
            Box box(min, max);
            if (intersectBox(rayOrigin, rayDir, box, tMin, tMax)) 
            {
                Eigen::Vector3d hitPoint = rayOrigin + tMin * rayDir;
                Eigen::Vector3d exitPoint = rayOrigin + tMax * rayDir;

                // sample ray while transmittance is significant
                double transmittance = 1.0;
                Eigen::Vector3d accumulatedColor(0.0, 0.0, 0.0);
                const double minTransmittance = 0.01; // Stop when transmittance drops below 1%
                int maxSteps = 100; // Maximum number of steps to prevent infinite loops
                int step = 0;
                
                while (transmittance > minTransmittance && step < maxSteps) 
                {
                    double t = static_cast<double>(step) / maxSteps;
                    Eigen::Vector3d samplePosition = hitPoint + (exitPoint - hitPoint) * t;
                    getValues(samplePosition, temperature, density);
                    
                    //normalize density and temperature
                    density = (density - minDensity) / (maxDensity - minDensity);
                    temperature = (temperature - minTemperature) / (maxTemperature - minTemperature);
                    
                    // Calculate step transmittance
                    double stepTransmittance = exp(-density * 0.1); // 0.1 is step size factor
                    
                    // Add color contribution with current transmittance
                    accumulatedColor += transmittance * density * Eigen::Vector3d(0.5, 0.5, 0.5);
                    
                    // Update transmittance for next step
                    transmittance *= stepTransmittance;
                    
                    step++;
                }
                finalColor = accumulatedColor;
            }
            image.setPixel(x, y, finalColor.x(), finalColor.y(), finalColor.z());
        }
    }
    
    // Save image to file
    image.savePPM("output.ppm");
    
    return 0;
}
