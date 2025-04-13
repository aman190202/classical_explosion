#include <unordered_set>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include "src/sampler.h"
#include "src/grid_lookup.h"
#include "src/helper.h"
#include "src/Image.h"
#include "src/Camera.h"
#include "src/Box.h"
#include "src/Plane.h"
#include "src/Light.h"
#include "src/Volume.h"
#include <atomic>

using namespace Eigen;

int main(int argc, char* argv[]) 
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vdb_file_path>" << std::endl;
        return 1;
    }
    
    std::string vdbFilePath = argv[1];
    // Extract filename from vdb file path
    std::string fileName = vdbFilePath.substr(vdbFilePath.find_last_of("/") + 1);
    fileName = fileName.substr(0, fileName.find_last_of("."));

    // Create image
    const int width = 1920;
    const int height = 1080;
    Image image(width, height);
    
    // Create camera inside the Cornell box
    Camera camera(
        Eigen::Vector3d(0, 0, 30),  // Moved camera closer
        Eigen::Vector3d(0, 0, 0),   // Look at the center
        Eigen::Vector3d(0, 1, 0),   // Up vector
        45.0,                       // Increased field of view
        static_cast<double>(width) / height, // Aspect ratio
        0.1,                        // Near plane
        1000.0                      // Far plane
    );

    std::unordered_set<Vector3d, Vec3Hash, Vec3Equal> densityLocations;
    std::unordered_set<Vector3d, Vec3Hash, Vec3Equal> temperatureLocations;
    
    // Populate sets from VDB file
    populateSetsFromVDB(vdbFilePath, densityLocations, temperatureLocations);  

    // get min and max temperature
    float minTemperature = std::numeric_limits<float>::infinity();
    float maxTemperature = -std::numeric_limits<float>::infinity();
    for (const auto& location : temperatureLocations) 
    {
        float temperature, density; 
        getValues(location, temperature, density);
        minTemperature = std::min(minTemperature, temperature);
        maxTemperature = std::max(maxTemperature, temperature);
    }

    float minDensity = std::numeric_limits<float>::infinity();
    float maxDensity = -std::numeric_limits<float>::infinity();
    for (const auto& location : densityLocations) 
    {
        float temperature, density; 
        getValues(location, temperature, density);
        minDensity = std::min(minDensity, density);
        maxDensity = std::max(maxDensity, density);
    }       

    // bounding box of the volume
    Vector3d min, max;
    getBoundingBox(vdbFilePath, min, max);
    std::cout << "Bounding Box: " << min << " " << max << std::endl;

    float og1 = min[1];
    float og2 = max[1];

    float difference = (max[1] - min[1]) ;
    max[1] = difference - 10;
    min[1] = -10;

    
    //OUTPUT NUMBER OF OPENMP THREADS
    std::cout << "Number of OpenMP threads: " << omp_get_max_threads() << std::endl;

    // fill light vector from temperature locations and get color based on temperature
    std::vector<Light> lights;

    for (const auto& location : temperatureLocations) 
    {
        float temperature, density;
        getValues(location, temperature, density);
        temperature = (temperature - minTemperature) / (maxTemperature - minTemperature);
        Vector3d color = interpolateColor(temperature);
        float intensity = 1.0f;  // Cap intensity at 1.0
        Vector3d location_copy = location;
        location_copy[1] -= 10;
        lights.push_back(Light(location_copy, color, intensity));
    }

    std::cout << "Lights: " << lights.size() << std::endl;

    // Add atomic counter for progress tracking
    std::atomic<int> completedRows(0);

    const int samplesPerPixel = 4;  // 4x4 grid = 16 samples per pixel
    const double sampleStep = 1.0 / samplesPerPixel;
    const double sampleOffset = sampleStep / 2.0;

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) 
    {
        for (int x = 0; x < width; ++x) 
        {
            Eigen::Vector3d finalColor(0.0, 0.0, 0.0);
            
            // Supersampling loop
            for (int sy = 0; sy < samplesPerPixel; ++sy) {
                for (int sx = 0; sx < samplesPerPixel; ++sx) {
                    // Calculate sub-pixel position
                    double u = (static_cast<double>(x) + sx * sampleStep + sampleOffset) / width;
                    double v = 1.0 - (static_cast<double>(y) + sy * sampleStep + sampleOffset) / height;

                    Eigen::Vector3d rayDir = camera.generateRay(u, v);
                    Eigen::Vector3d rayOrigin = camera.getPosition();

                    float c_t;
                    float t;
                    Eigen::Vector3d sampleColor;
                    if (rayIntersectsVolume(rayOrigin, rayDir, min, max, t)) 
                    {
                        sampleColor = getVolumeColor(rayOrigin, rayDir, min, max, og1, og2, minTemperature, maxTemperature, minDensity, maxDensity, t, lights);
                    }
                    else
                    {
                        sampleColor = CornellBox(rayOrigin, rayDir, c_t, lights);
                    }
                    
                    // Apply tone mapping to each sample
                    sampleColor = sampleColor.cwiseQuotient(sampleColor + Vector3d(1.0, 1.0, 1.0));
                    sampleColor = sampleColor.array().pow(1.0/2.2);
                    
                    finalColor += sampleColor;
                }
            }
            
            // Average the samples
            finalColor /= (samplesPerPixel * samplesPerPixel);

            #pragma omp critical
            {
                image.setPixel(x, y, finalColor.x(), finalColor.y(), finalColor.z());
            }
        }
        
        // Update progress after completing a row
        completedRows++;
        #pragma omp critical
        {
            float progress = (float)completedRows / height * 100.0f;
            std::cout << "\rRendering progress: " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
        }
    }
    
    // Save final image
    std::string finalFileName = "output/" + fileName + ".ppm";
    image.savePPM(finalFileName);
    
    return 0;
}
