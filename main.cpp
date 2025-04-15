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
        Eigen::Vector3d(0, 0, 100),  // Moved camera closer
        Eigen::Vector3d(0, 0, 0),   // Look at the center
        Eigen::Vector3d(0, 1, 0),   // Up vector
        45.0,                       // Increased field of view
        static_cast<double>(width) / height, // Aspect ratio
        0.1,                        // Near plane
        1000.0                      // Far plane
    );

    Vector3d min_density, max_density, min_temperature, max_temperature; // bounfing box of the volume
    getBoundingBox(vdbFilePath, min_density, max_density, min_temperature, max_temperature);

    
    float temperature = getTemperature((min_temperature + max_temperature) / 2);
    float density = getDensity((min_density + max_density) / 2);

    float og1 = min_temperature[1];
    float og2 = max_temperature[1];

    float difference = (max_temperature[1] - min_temperature[1]);
    max_temperature[1] = difference - 30;
    min_temperature[1] = -30;


    Vector3d min = min_temperature;
    Vector3d max = max_temperature;

    
    //OUTPUT NUMBER OF OPENMP THREADS
    std::cout << "Number of OpenMP threads: " << omp_get_max_threads() << std::endl;

    // fill light vector from temperature locations and get color based on temperature
    std::vector<Light> lights;
    lights.push_back(Light(Vector3d(0, 0, 0), Vector3d(1.0, 1.0, 1.0), 1.0));

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
                    Eigen::Vector3d sampleColor{0.0, 0.0, 0.0};
                    float cb, vb;
                    bool is_c = checkCornellBoxIntersect(rayOrigin, rayDir, cb);
                    bool is_v = rayIntersectsVolume(rayOrigin, rayDir, min, max, vb);
                    if(is_c && is_v)
                    {
                        if(cb < vb)
                        {
                            sampleColor = CornellBox(rayOrigin, rayDir, c_t, lights);
                        }
                        else
                        {
                            sampleColor = Vector3d(1.0, 0.0, 0.0); //getVolumeColor(rayOrigin, rayDir, min, max, og1, og2, minTemperature, maxTemperature, minDensity, maxDensity, t, lights);
                        }
                    }
                    else if(is_c)
                    {
                        sampleColor = CornellBox(rayOrigin, rayDir, c_t, lights);
                    }
                    else if(is_v)
                    {
                        sampleColor = Vector3d(0.0, 1.0, 0.0); //getVolumeColor(rayOrigin, rayDir, min, max, og1, og2, minTemperature, maxTemperature, minDensity, maxDensity, t, lights);
                    }
                    else
                    {
                        sampleColor = Vector3d(0.0, 0.0, 0.0);
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
