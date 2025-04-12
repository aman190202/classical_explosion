#ifndef SAMPLER_H
#define SAMPLER_H

#include <Eigen/Dense>
#include <string>
#include <memory>
#include <optional>
#include <stdexcept>

#ifdef USE_OPENVDB
#include <openvdb/openvdb.h>
#endif

class VDBVolumeSampler {
private:
#ifdef USE_OPENVDB
    openvdb::FloatGrid::Ptr densityGrid;
    openvdb::FloatGrid::Ptr temperatureGrid;
    std::optional<openvdb::FloatGrid::ConstAccessor> densityAccessor;
    std::optional<openvdb::FloatGrid::ConstAccessor> temperatureAccessor;
    openvdb::math::Transform::Ptr transform;
#endif
    bool initialized;

public:
    VDBVolumeSampler(const std::string& vdbFilePath) {
#ifdef USE_OPENVDB
        // Initialize OpenVDB
        openvdb::initialize();
        
        // Open the VDB file
        openvdb::io::File file(vdbFilePath);
        file.open();
        
        // Get the grids
        openvdb::GridPtrVecPtr grids = file.getGrids();
        
        // Find density and temperature grids
        for (openvdb::GridBase::Ptr grid : *grids) {
            if (grid->getName() == "density") {
                densityGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid);
                densityAccessor.emplace(densityGrid->getConstAccessor());
            }
            else if (grid->getName() == "temperature") {
                temperatureGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid);
                temperatureAccessor.emplace(temperatureGrid->getConstAccessor());
            }
        }
        
        if (!densityGrid || !temperatureGrid) {
            throw std::runtime_error("Could not find density or temperature grid in VDB file");
        }
        
        transform = densityGrid->transformPtr();
        initialized = true;
#else
        throw std::runtime_error("OpenVDB support not enabled");
#endif
    }

    // Sample density at world position
    float sampleDensity(const Eigen::Vector3d& worldPos) const {
#ifdef USE_OPENVDB
        if (!densityAccessor) {
            throw std::runtime_error("Density accessor not initialized");
        }
        openvdb::Vec3d pos(worldPos.x(), worldPos.y(), worldPos.z());
        openvdb::Coord coord = transform->worldToIndexCellCentered(pos);
        return densityAccessor->getValue(coord);
#else
        // Return a default value when OpenVDB is not enabled
        return 0.0f;
#endif
    }

    // Sample temperature at world position
    float sampleTemperature(const Eigen::Vector3d& worldPos) const {
#ifdef USE_OPENVDB
        if (!temperatureAccessor) {
            throw std::runtime_error("Temperature accessor not initialized");
        }
        openvdb::Vec3d pos(worldPos.x(), worldPos.y(), worldPos.z());
        openvdb::Coord coord = transform->worldToIndexCellCentered(pos);
        return temperatureAccessor->getValue(coord);
#else
        // Return a default value when OpenVDB is not enabled
        return 0.0f;
#endif
    }

    bool isInitialized() const { return initialized; }
};

// Global sampler instance
extern std::unique_ptr<VDBVolumeSampler> volumeSampler;

// Initialize the sampler with a VDB file
void initializeSampler(const std::string& vdbFilePath);

// Density sampling function
Eigen::Vector3d densitySampler(const Eigen::Vector3d& position);

// Temperature sampling function
Eigen::Vector3d colorSampler(const Eigen::Vector3d& position);

#endif // SAMPLER_H
