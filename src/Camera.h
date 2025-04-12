#ifndef CAMERA_H
#define CAMERA_H

#include "Eigen/Dense"
#include <cmath>

class Camera {
private:
    Eigen::Vector3d position;    // Camera position in world space
    Eigen::Vector3d lookAt;      // Point the camera is looking at
    Eigen::Vector3d up;          // Up vector
    double fov;                  // Field of view in degrees
    double aspectRatio;          // Width/height ratio
    double nearPlane;            // Near clipping plane
    double farPlane;             // Far clipping plane

    // Derived vectors
    Eigen::Vector3d forward;     // Forward direction vector
    Eigen::Vector3d right;       // Right direction vector
    Eigen::Vector3d upVector;    // Up direction vector

    void updateVectors() {
        // Calculate forward vector
        forward = (lookAt - position).normalized();
        
        // Calculate right vector
        right = forward.cross(up).normalized();
        
        // Recalculate up vector to ensure orthogonality
        upVector = right.cross(forward).normalized();
    }

public:
    // Constructor
    Camera(const Eigen::Vector3d& pos = Eigen::Vector3d(0, 0, 0),
           const Eigen::Vector3d& target = Eigen::Vector3d(0, 0, -1),
           const Eigen::Vector3d& upDir = Eigen::Vector3d(0, 1, 0),
           double fieldOfView = 60.0,
           double ratio = 16.0/9.0,
           double near = 0.1,
           double far = 1000.0)
        : position(pos), lookAt(target), up(upDir),
          fov(fieldOfView), aspectRatio(ratio),
          nearPlane(near), farPlane(far) {
        updateVectors();
    }

    // Getters
    const Eigen::Vector3d& getPosition() const { return position; }
    const Eigen::Vector3d& getLookAt() const { return lookAt; }
    const Eigen::Vector3d& getUp() const { return up; }
    const Eigen::Vector3d& getForward() const { return forward; }
    const Eigen::Vector3d& getRight() const { return right; }
    const Eigen::Vector3d& getUpVector() const { return upVector; }
    double getFOV() const { return fov; }
    double getAspectRatio() const { return aspectRatio; }
    double getNearPlane() const { return nearPlane; }
    double getFarPlane() const { return farPlane; }

    // Setters
    void setPosition(const Eigen::Vector3d& pos) {
        position = pos;
        updateVectors();
    }

    void setLookAt(const Eigen::Vector3d& target) {
        lookAt = target;
        updateVectors();
    }

    void setUp(const Eigen::Vector3d& upDir) {
        up = upDir;
        updateVectors();
    }

    void setFOV(double fieldOfView) {
        fov = fieldOfView;
    }

    void setAspectRatio(double ratio) {
        aspectRatio = ratio;
    }

    void setNearPlane(double near) {
        nearPlane = near;
    }

    void setFarPlane(double far) {
        farPlane = far;
    }

    // Generate a ray from camera position through a pixel
    Eigen::Vector3d generateRay(double u, double v) const {
        // Convert FOV to radians
        double fovRad = fov * M_PI / 180.0;
        
        // Calculate viewport dimensions
        double viewportHeight = 2.0 * tan(fovRad / 2.0);
        double viewportWidth = viewportHeight * aspectRatio;
        
        // Calculate pixel position in viewport space
        double pixelX = (u - 0.5) * viewportWidth;
        double pixelY = (v - 0.5) * viewportHeight;
        
        // Calculate ray direction
        Eigen::Vector3d rayDir = forward + 
                                right * pixelX + 
                                upVector * pixelY;
        
        return rayDir.normalized();
    }

    // Get view matrix (for use in shaders or other transformations)
    Eigen::Matrix4d getViewMatrix() const {
        Eigen::Matrix4d view = Eigen::Matrix4d::Identity();
        
        // Calculate the basis vectors
        Eigen::Vector3d z = -forward;
        Eigen::Vector3d x = right;
        Eigen::Vector3d y = upVector;
        
        // Set the rotation part of the view matrix
        view.block<1,3>(0,0) = x.transpose();
        view.block<1,3>(1,0) = y.transpose();
        view.block<1,3>(2,0) = z.transpose();
        
        // Set the translation part
        view.block<3,1>(0,3) = -position;
        
        return view;
    }

    // Get projection matrix (for use in shaders or other transformations)
    Eigen::Matrix4d getProjectionMatrix() const {
        Eigen::Matrix4d projection = Eigen::Matrix4d::Zero();
        
        double fovRad = fov * M_PI / 180.0;
        double tanHalfFov = tan(fovRad / 2.0);
        
        projection(0,0) = 1.0 / (aspectRatio * tanHalfFov);
        projection(1,1) = 1.0 / tanHalfFov;
        projection(2,2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        projection(2,3) = -2.0 * farPlane * nearPlane / (farPlane - nearPlane);
        projection(3,2) = -1.0;
        
        return projection;
    }
};

#endif // CAMERA_H 