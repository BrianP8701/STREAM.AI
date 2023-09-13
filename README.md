# STREAM AI - Tip Tracker & Anomaly Detection
## Overview

This project aims to provide a real-time solution to common 3D printing errors, such as over-extrusion, under-extrusion, and inconsistencies caused by bubbles in the extruder. Our system's end goal is to use real-time video, gcode and data from the digital twin sensors to monitor the printing process, track the extruder tip's position, identify printing errors and make real-time adjustments to the print based on the measured width of the extruded material.

There are many sources of error in tracking the tip. To counter these we created methods to detect, process and correct tips. Further details can be found:
  - [Short Written Report](https://docs.google.com/document/d/1vwumKr0Bu93e1cg_Y6qv2G6o-ez8gFCLIyPrFaPOw8E/edit?usp=sharing)
  - [Video Presentation](https://youtu.be/gBybSietDuw)

## Pre-requisites
- Python >= 3.x
- pip

## Installation Steps

### Clone the Repository
```bash
git clone https://github.com/yourusername/yourproject.git](https://github.com/BrianP8701/STREAM.AI.git
cd [Path to this project]
```
### Create a Virtual Environment

For macOS and Linux:
```bash
python3 -m venv myenv
```
For Windows:
```cmd
python -m venv myenv
```
### Activate the Virtual Environment
For macOS and Linux:
```bash
source myenv/bin/activate
```
For Windows:
```cmd
.\myenv\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
To run the system, simply choose your video, gcode and signals path on main.py and run.

## Contributing

Bug Reporting: Should you stumble upon any bugs or challenges, we appreciate detailed issue reports. Please create a new issue, outlining the encountered problem and the specific inputs you used.

Optimizations & Refinements: If you discover ways to enhance the system's efficiency, improve robustness, or streamline the code, kindly submit a pull request with your proposed changes.

Thank you!

