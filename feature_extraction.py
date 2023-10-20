import pandas as pd

def goal_difference(data):
    """Calculate the goal difference for each match."""
    data['goal_diff'] = data['gf'] - data['ga']
    return data

def shot_accuracy(data):
    """Calculate the ratio of shots on target to total shots."""
    data['shot_accuracy'] = data['sot'] / data['sh']
    return data

def extract_features(data):
    """Main function to extract features from the data."""
    data = goal_difference(data)
    data = shot_accuracy(data)
    # Add more feature extraction functions as needed
    return data
