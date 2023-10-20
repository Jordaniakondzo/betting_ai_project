import pandas as pd
from sklearn.model_selection import train_test_split

def convert_data_types(data):
    """Convert data types of columns for analysis."""
    data['gf'] = data['gf'].astype(int)
    data['ga'] = data['ga'].astype(int)
    return data

def handle_missing_values(data):
    """Check and handle missing values."""
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            data[column].fillna(data[column].mean(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    return data

def convert_date_features(data):
    """Convert date column to numerical features."""
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data.drop('date', axis=1, inplace=True) 
    return data

def encode_categorical_columns(data):
    """One-hot encode categorical columns."""
    # Drop non-numeric and potentially problematic columns
    data.drop(columns=['notes', 'match report'], errors='ignore', inplace=True)
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    data = pd.get_dummies(data, columns=categorical_columns)
    return data

def split_data(data):
    """Split data into training and testing sets."""
    if 'result_D' in data.columns and 'result_W' in data.columns:
        # Create a new target column based on result_D and result_W
        data['target'] = -1  # default to loss
        data.loc[data['result_D'] == 1, 'target'] = 0  # draw
        data.loc[data['result_W'] == 1, 'target'] = 1  # win
        # Now drop the one-hot encoded columns
        data.drop(['result_D', 'result_W'], axis=1, inplace=True)
        y = data["target"]
    else:
        y = data["result"]

    X = data.drop("target", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reset index for train and test data
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test

def preprocess_data(data):
    """Main function to preprocess the data."""
    data = convert_data_types(data)
    data = handle_missing_values(data)
    data = convert_date_features(data)
    data = encode_categorical_columns(data)
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Save data
    X_test.to_csv('X_test_saved.csv', index=False)
    X_train.to_csv('X_train_saved.csv', index=False)
    y_test.to_csv('y_test_saved.csv', index=False)
    y_train.to_csv('y_train_saved.csv', index=False)

    return X_train, X_test, y_train, y_test
   