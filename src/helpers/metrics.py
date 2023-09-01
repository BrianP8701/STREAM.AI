import json
import os

def update_json_dicts(json_path, data_to_add):
    """
    Update dictionaries inside a JSON file.

    Parameters:
    - json_path (str): The path to the JSON file to update.
    - data_to_add (dict): Dictionary containing data to add to the dictionaries inside the JSON file.
                          Keys are dictionary names within the JSON file, and values are key-value pairs to add to these dictionaries.

    Behavior:
    - Reads existing data from the JSON file.
    - Updates dictionaries within the JSON file with key-value pairs from data_to_add.
    - Writes the updated data back to the JSON file.

    Example:
    >>> update_json_dicts('data.json', {"list1": {"new_key1": "new_value1"}, "list2": {"new_key2": "new_value2"}})
    """
    # Read existing JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Update dictionaries
    for dict_name, key_value_pairs in data_to_add.items():
        if dict_name in json_data:
            json_data[dict_name].update(key_value_pairs)

    # Write updated JSON data back to file
    with open(json_path, 'w') as f:
        json.dump(json_data, f)

# Test the function
json_path = 'data.json'
data_to_add = {
    "list1": {"new_key1": "new_value1"}, 
    "list2": {"new_key2": "new_value2"}
}
update_json_dicts(json_path, data_to_add)
