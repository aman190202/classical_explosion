#include <openvdb/openvdb.h>
#include <iostream>
#include <string>
#include <limits>
#include <fstream>
#include <vector>
#include <cmath>

// Structure to hold RGB color
struct Color {
    unsigned char r, g, b;
    Color(unsigned char r_ = 0, unsigned char g_ = 0, unsigned char b_ = 0) 
        : r(r_), g(g_), b(b_) {}
};



// Function to map value to color (using a heat map style coloring)
Color valueToColor(float value, float minVal, float maxVal) {
    // Normalize value between 0 and 1
    float normalized = (value - minVal) / (maxVal - minVal);
    normalized = std::max(0.0f, std::min(1.0f, normalized));
    
    // Create a heat map style coloring
    Color color;
    if (normalized < 0.25f) {
        // Black to Blue
        color.b = static_cast<unsigned char>(normalized * 4 * 255);
    } else if (normalized < 0.5f) {
        // Blue to Green
        color.b = static_cast<unsigned char>((0.5f - normalized) * 4 * 255);
        color.g = static_cast<unsigned char>((normalized - 0.25f) * 4 * 255);
    } else if (normalized < 0.75f) {
        // Green to Red
        color.g = static_cast<unsigned char>((0.75f - normalized) * 4 * 255);
        color.r = static_cast<unsigned char>((normalized - 0.5f) * 4 * 255);
    } else {
        // Red to White
        color.r = 255;
        color.g = color.b = static_cast<unsigned char>((normalized - 0.75f) * 4 * 255);
    }
    return color;
}

// Function to save a 2D slice as PPM image
void saveSliceToPPM(const std::string& filename, const std::vector<Color>& pixels, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& color : pixels) {
        file.write(reinterpret_cast<const char*>(&color.r), 1);
        file.write(reinterpret_cast<const char*>(&color.g), 1);
        file.write(reinterpret_cast<const char*>(&color.b), 1);
    }
}

// Function to create a slice visualization
void visualizeGridSlice(openvdb::FloatGrid::Ptr grid, const std::string& name, 
                       const openvdb::CoordBBox& bbox, float minVal, float maxVal) {
    // Get the dimensions of the bounding box
    openvdb::Coord dims = bbox.max() - bbox.min() + openvdb::Coord(1);
    int width = dims.x();
    int height = dims.y();
    
    // Create accessor for efficient value access
    auto accessor = grid->getConstAccessor();
    
    // Create pixel buffer
    std::vector<Color> pixels(width * height);
    
    // Get the middle Z coordinate for the slice
    int midZ = (bbox.min().z() + bbox.max().z()) / 2;
    
    // Fill the pixel buffer
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            openvdb::Coord xyz(x + bbox.min().x(), y + bbox.min().y(), midZ);
            float value = accessor.getValue(xyz);
            pixels[y * width + x] = valueToColor(value, minVal, maxVal);
        }
    }
    
    // Save the image
    std::string filename = name + "_slice.ppm";
    saveSliceToPPM(filename, pixels, width, height);
    std::cout << "Saved visualization to " << filename << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <vdb_file>" << std::endl;
        return 1;
    }

    // Initialize OpenVDB
    openvdb::initialize();
    
    try {
        std::string filename = argv[1];
        std::cout << "\nAnalyzing VDB file: " << filename << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        // Open the VDB file
        openvdb::io::File file(filename);
        file.open();

        // Get grid metadata
        openvdb::GridPtrVecPtr grids = file.getGrids();

        std::cout << "\nFile Information:" << std::endl;
        std::cout << "Number of grids: " << grids->size() << std::endl;

        // Analyze each grid
        for (openvdb::GridBase::Ptr grid : *grids) {
            std::cout << "\nGrid: " << grid->getName() << std::endl;
            std::cout << std::string(30, '-') << std::endl;

            // Grid type and other basic info
            std::cout << "Grid type: " << grid->type() << std::endl;
            std::cout << "Value type: " << grid->valueType() << std::endl;
            std::cout << "Class: " << grid->getGridClass() << std::endl;
            
            // Get transform info
            openvdb::math::Transform::Ptr transform = grid->transformPtr();
            std::cout << "Voxel size: " << transform->voxelSize()[0] << std::endl;

            // Get grid statistics
            std::cout << "\nGrid Statistics:" << std::endl;
            std::cout << "Active voxel count: " << grid->activeVoxelCount() << std::endl;
            std::cout << "Memory usage (bytes): " << grid->memUsage() << std::endl;

            // Get bounding box
            openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
            openvdb::Vec3d minWorld = transform->indexToWorld(bbox.min());
            openvdb::Vec3d maxWorld = transform->indexToWorld(bbox.max());
            
            std::cout << "\nBounding Box:" << std::endl;
            std::cout << "Min voxel index: " << bbox.min() << std::endl;
            std::cout << "Max voxel index: " << bbox.max() << std::endl;
            std::cout << "Min world position: " << minWorld << std::endl;
            std::cout << "Max world position: " << maxWorld << std::endl;

            // Check value at origin (0,0,0)
            openvdb::Vec3d origin(0.0, 0.0, 0.0);
            openvdb::Coord originCoord = transform->worldToIndexCellCentered(origin);
            std::cout << "\nValue at origin (0,0,0):" << std::endl;
            std::cout << "Voxel index: " << originCoord << std::endl;

            // Try to cast to specific grid types for more information
            if (auto floatGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid)) {
                std::cout << "\nFloat Grid Statistics:" << std::endl;
                float minVal = std::numeric_limits<float>::max();
                float maxVal = std::numeric_limits<float>::lowest();
                
                // Check value at origin
                auto accessor = floatGrid->getConstAccessor();
                float originValue = accessor.getValue(originCoord);
                std::cout << "Value at origin: " << originValue << std::endl;
                
                // Find min/max values
                for (auto iter = floatGrid->beginValueOn(); iter; ++iter) {
                    float val = *iter;
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
                
                std::cout << "\nValue Statistics:" << std::endl;
                std::cout << "Min value: " << minVal << std::endl;
                std::cout << "Max value: " << maxVal << std::endl;

                // Create visualization
                visualizeGridSlice(floatGrid, grid->getName(), bbox, minVal, maxVal);
            }
            else if (auto vec3Grid = openvdb::GridBase::grid<openvdb::Vec3SGrid>(grid)) {
                std::cout << "\nVector Grid detected" << std::endl;
                
                // Check value at origin
                auto accessor = vec3Grid->getConstAccessor();
                openvdb::Vec3f originValue = accessor.getValue(originCoord);
                std::cout << "Value at origin: " << originValue << std::endl;
                
                openvdb::Vec3f minVal(std::numeric_limits<float>::max());
                openvdb::Vec3f maxVal(std::numeric_limits<float>::lowest());
                
                // Find min/max values
                for (auto iter = vec3Grid->beginValueOn(); iter; ++iter) {
                    openvdb::Vec3f val = *iter;
                    minVal = openvdb::Vec3f(
                        std::min(minVal[0], val[0]),
                        std::min(minVal[1], val[1]),
                        std::min(minVal[2], val[2])
                    );
                    maxVal = openvdb::Vec3f(
                        std::max(maxVal[0], val[0]),
                        std::max(maxVal[1], val[1]),
                        std::max(maxVal[2], val[2])
                    );
                }
                
                std::cout << "\nValue Statistics:" << std::endl;
                std::cout << "Min values (x,y,z): " << minVal << std::endl;
                std::cout << "Max values (x,y,z): " << maxVal << std::endl;
            }

            std::cout << std::endl << std::string(50, '=') << std::endl;
        }

        file.close();
    }
    catch (const std::exception& e) {
        std::cerr << "Error analyzing VDB file: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 