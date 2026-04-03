# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS package implementing a high-precision online markerless stereo extrinsic calibration system based on the research paper "High-Precision Online Markerless Stereo Extrinsic Calibration" (see `document/` folder).

The system performs real-time calibration of stereo camera extrinsic parameters without requiring calibration markers or patterns.

## Build System

This is a ROS package using CMake:

```bash
# Build the package (from workspace root)
catkin_make

# Or with catkin build
catkin build stereo_extrinsic_calibration
```

## Project Status

Currently in initial development phase. The repository contains:
- Empty CMakeLists.txt and package.xml (need to be populated)
- Reference documentation (PDF paper and system diagram)
- No source code implementation yet

