#!/bin/bash



# Use ffmpeg to create a video from the frames
# -pattern_type glob: use glob pattern matching
# -framerate 24: set frame rate to 24 fps
# -i "frames/*.ppm": input all PPM files from frames directory
# -c:v libopenh264: use OpenH264 codec
# -pix_fmt yuv420p: use YUV 4:2:0 pixel format for compatibility
# -b:v 5M: set bitrate to 5 Mbps for good quality
ffmpeg -pattern_type glob -framerate 24 -i "frames/*.ppm" \
    -c:v libopenh264 -pix_fmt yuv420p -b:v 5M \
    output/explosion_animation.mp4

echo "Video created: output/explosion_animation.mp4" 