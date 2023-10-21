import pandas as pd
import os
import logging
from data_loader import load_data
from data_preprocessing import preprocess_data
from feature_extraction import extract_features
from model import RandomForestModel, XGBoostModel

# Setup logging
logging.basicConfig(level=logging.INFO)

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """Train and evaluate a list of models."""
    results = []  # List to store evaluation results

    for model_name, model_instance in models:
        logging.info(f"Training {model_name}...")
        model_instance.train(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        
        logging.info(f"Results for {model_name}:")
        model_instance.evaluate(y_test, y_pred)
        logging.info("--------------------------------------------------")
        
        # Save the model after training
        model_path = os.path.join("trained_models", f"{model_name}_saved_model.pkl")
        model_instance.save_model(model_path)

        # Store the results for analysis
        results.append({
            "model": model_name,
            "evaluation": model_instance.evaluate(y_test, y_pred)
        })

    # Write the results to a file
    with open('results.txt', 'w') as f:
        for result in results:
            f.write(str(result) + '\n')

# The rest of your main code
data = load_data('matches.csv') 
logging.info(data.columns)

X_train, X_test, y_train, y_test = preprocess_data(data)

X_train = extract_features(X_train)
X_test = extract_features(X_test)

models = [
    ("Random Forest", RandomForestModel()),
    ("Gradient Boosting", XGBoostModel())
]

train_and_evaluate(models, X_train, y_train, X_test, y_test)
