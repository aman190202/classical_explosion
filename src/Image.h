#ifndef IMAGE_H
#define IMAGE_H

#include "Eigen/Dense"
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

class Image {
private:
    int width;
    int height;
    Eigen::MatrixXd red;
    Eigen::MatrixXd green;
    Eigen::MatrixXd blue;

public:
    // Constructor
    Image(int w, int h) : width(w), height(h) {
        red = Eigen::MatrixXd::Zero(height, width);
        green = Eigen::MatrixXd::Zero(height, width);
        blue = Eigen::MatrixXd::Zero(height, width);
    }

    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Set color at specific pixel
    void setPixel(int x, int y, double r, double g, double b) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            red(y, x) = std::clamp(r, 0.0, 1.0);
            green(y, x) = std::clamp(g, 0.0, 1.0);
            blue(y, x) = std::clamp(b, 0.0, 1.0);
        }
    }

    // Get color at specific pixel
    void getPixel(int x, int y, double& r, double& g, double& b) const {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            r = red(y, x);
            g = green(y, x);
            b = blue(y, x);
        }
    }

    // Fill the entire image with a color
    void fill(double r, double g, double b) {
        red.setConstant(std::clamp(r, 0.0, 1.0));
        green.setConstant(std::clamp(g, 0.0, 1.0));
        blue.setConstant(std::clamp(b, 0.0, 1.0));
    }

    // Save image to PPM file
    bool savePPM(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        // Write PPM header
        file << "P3\n" << width << " " << height << "\n255\n";

        // Write pixel data
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Convert from [0,1] to [0,255] range
                int r = static_cast<int>(red(y, x) * 255);
                int g = static_cast<int>(green(y, x) * 255);
                int b = static_cast<int>(blue(y, x) * 255);
                
                file << r << " " << g << " " << b << " ";
            }
            file << "\n";
        }

        return true;
    }

    // Get direct access to color channels
    Eigen::MatrixXd& getRed() { return red; }
    Eigen::MatrixXd& getGreen() { return green; }
    Eigen::MatrixXd& getBlue() { return blue; }
};

#endif // IMAGE_H 