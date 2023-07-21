# STREAM AI - Tip Tracker & Measure Diameter 
## Overview

This project aims to provide a real-time solution to common 3D printing errors, such as over-extrusion, under-extrusion, and inconsistencies caused by bubbles in the extruder. Our system's end goal is to use real-time video, gcode and data from the digital twin sensors to monitor the printing process, track the extruder tip's position, identify printing errors and make real-time adjustments to the print based on the measured width of the extruded material.

There are many sources of error in tracking the tip. To counter these we created methods to detect, process and correct tips. Further details can be found at: https://docs.google.com/document/d/1JAA3Ad73O36-zbMMh2id8rbsgu4T1TKEHiiCEI0894E/edit#

## Installation
pip is a Python package installer. You can use it to install the required packages into a virtual environment. Here's how to do it:

Clone the project:
```
git clone https://github.com/BrianP8701/STREAM.AI
```

Navigate to the project directory:
```
cd STREAM.AI
```

Create a new virtual environment:
```
python3 -m venv env
```

Activate the new environment:
```
source env/bin/activate  # Use 'env\Scripts\activate' on Windows
```

Install the required packages from the requirements.txt file:
```
pip install -r requirements.txt
```

## Usage

## Contributing
