# STREAM AI - Tip Tracker & Measure Diameter 
## Overview

This project aims to provide a real-time solution to common 3D printing errors, such as over-extrusion, under-extrusion, and inconsistencies caused by bubbles in the extruder. Our system's end goal is to use real-time video, gcode and data from the digital twin sensors to monitor the printing process, track the extruder tip's position, identify printing errors and make real-time adjustments to the print based on the measured width of the extruded material.

There are many sources of error in tracking the tip. To counter these we created methods to detect, process and correct tips. Further details can be found at: https://docs.google.com/document/d/1JAA3Ad73O36-zbMMh2id8rbsgu4T1TKEHiiCEI0894E/edit#

Certainly. Here's a concise README for your project:

---

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

### Running the Project

[Specific instructions on how to run or use your project.]

---

Replace the placeholders (`[Your Project Name]`, `[Short description of your project.]`, `[YOUR_REPO_LINK]`, `[YOUR_REPO_DIRECTORY]`, and `[YOUR_ENV_NAME]`) with the appropriate information. The last section, "Running the Project", is a placeholder for any specific instructions you might have for someone to run or use your project after setting up the environment.
## Usage

## Contributing
