# STREAM AI - Tip Tracker & Measure Diameter 
## Overview

This project aims to provide a real-time solution to common 3D printing errors, such as over-extrusion, under-extrusion, and inconsistencies caused by bubbles in the extruder. Our system's end goal is to use real-time video, gcode and data from the digital twin sensors to monitor the printing process, track the extruder tip's position, identify printing errors and make real-time adjustments to the print based on the measured width of the extruded material.

There are many sources of error in tracking the tip. To counter these we created methods to detect, process and correct tips. Further details can be found at: https://docs.google.com/document/d/1JAA3Ad73O36-zbMMh2id8rbsgu4T1TKEHiiCEI0894E/edit#


## Setup Instructions

### Prerequisites

- Make sure you have `conda` installed. If not, [download and install Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Setting up the Environment

1. Clone this repository:
```bash
git clone https://github.com/BrianP8701/STREAM.AI
cd [STREAM_AI_REPO_DIRECTORY]
```

2. Recreate the conda environment from the provided `environment.yml`:
```bash
conda env create -f environment.yml
```

3. Activate the newly created environment:
```bash
conda activate STREAM_AI
```


## Running the Project


## Usage


## Contributing

Bug Reporting: Should you stumble upon any bugs or challenges, we appreciate detailed issue reports. Please create a new issue, outlining the encountered problem and the specific inputs you used.

Optimizations & Refinements: If you discover ways to enhance the system's efficiency, improve robustness, or streamline the code, kindly submit a pull request with your proposed changes.

Thank you!


