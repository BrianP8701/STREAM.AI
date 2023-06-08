# Setting up the enviroment
To run this project, you'll need to set up a Python environment and install the required packages. You can do this using either Conda or pip:

## Option 1: Using Conda
Conda is a package and environment manager. It can recreate the exact environment this project was developed in. Here's how to do it:

Clone the project:
```
git clone https://github.com/your_username/your_project.git
```

Navigate to the project directory:
```
cd your_project
```

Create a new Conda environment from the environment.yml file:
```
conda env create -f environment.yml
```

Activate the new environment:
```
conda activate your_env_name
```
## Option 2: Using pip
pip is a Python package installer. You can use it to install the required packages into a virtual environment. Here's how to do it:

Clone the project:
```
git clone https://github.com/your_username/your_project.git
```

Navigate to the project directory:
```
cd your_project
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
