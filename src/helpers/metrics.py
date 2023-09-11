'''
    This file contains helper functions for saving and analyzing metrics.
'''
import json
import matplotlib.pyplot as plt
import os

def update_json_dicts(json_path, data_to_add):
    """
    Update dictionaries inside a JSON file.

    Parameters:
    - json_path (str): The path to the JSON file to update.
    - data_to_add (dict): Dictionary containing data to add to the dictionaries inside the JSON file.
                          Keys are dictionary names within the JSON file, and values are key-value pairs to add to these dictionaries.

    Behavior:
    - Reads existing data from the JSON file if it exists, or creates a new empty JSON file if it doesn't.
    - Updates dictionaries within the JSON file with key-value pairs from data_to_add.
    - Writes the updated data back to the JSON file.

    Example:
    >>> update_json_dicts('data.json', {"list1": {"new_key1": "new_value1"}, "list2": {"new_key2": "new_value2"}})
    """

    # Check if the file exists
    if os.path.exists(json_path):
        # Read existing JSON data
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    else:
        # Create an empty dictionary if the file doesn't exist
        json_data = {}

    # Update dictionaries
    for dict_name, key_value_pairs in data_to_add.items():
        if dict_name in json_data:
            json_data[dict_name].update(key_value_pairs)
        else:
            json_data[dict_name] = key_value_pairs

    # Write updated JSON data back to file
    with open(json_path, 'w') as f:
        json.dump(json_data, f)


def plot_json_data(json_path, save_path, key, x_label, y_label, graph_title):
    """
    Plot data from a JSON file with a specific key.

    Parameters:
    - json_path (str): Path to the JSON file.
    - key (str): Key in the JSON dictionary containing the data to plot.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - graph_title (str): Title for the graph.

    Behavior:
    - Reads data associated with the provided key from the JSON file.
    - Plots the data on a graph with provided labels and title.
    """
    
    # Load data from JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if key not in data:
        raise ValueError(f"No data found for key: {key}")

    # Extract x and y values
    x_values = [item[0] for item in data[key]]
    y_values = [item[1] for item in data[key]]

    # Plot the data
    plt.plot(x_values, y_values, 'o-')  # 'o-' means to use dots for data points and lines to connect them
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(graph_title)
    
    # Automatically adjust the x and y limits
    plt.xlim(min(x_values), max(x_values))
    plt.ylim(min(y_values), max(y_values))

    plt.grid(True)
    plt.savefig(save_path, format='png', dpi=300)  # Adjust format and dpi as needed


