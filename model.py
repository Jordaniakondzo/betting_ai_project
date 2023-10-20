from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from joblib import dump, load
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

class BaseModel:
    def train(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_
        print(f"Best hyperparameters for {type(self.model).__name__}: {self.grid_search.best_params_}")
        scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Accuracy for each fold: {scores}")
        print(f"Average accuracy with cross-validation: {scores.mean()}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def evaluate(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    def save_model(self, filename):
         dump(self.model, filename)
         print(f"{type(self.model).__name__} model saved to xgboost_model.jolib")

    def save_model(self, filename):
         dump(self.model, filename)
         print(f"{type(self.model).__name__} model saved to random_forest_model.joblibb")

    def bet_recommendation(self, probabilities):
        threshold_recommended = 0.7  # Modify as needed
        threshold_suspect = 0.5  # Modify as needed
        if probabilities[1] >= threshold_recommended:
            return "Recommended"
        elif probabilities[1] < threshold_suspect:
            return "Dangerous"
        else:
            return "Suspicious"

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier()
        params = {'n_estimators': [10, 50, 100, 200], 
                  'max_depth': [None, 10, 20, 30, 40, 50]}
        self.grid_search = GridSearchCV(self.model, params, cv=3, n_jobs=-1, verbose=2)

class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        params = {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9]
        }
        self.grid_search = GridSearchCV(self.model, params, cv=3, n_jobs=-1, verbose=2)

class MatchPredictor:
    def __init__(self, model):
        self.model = model

    def predict_and_recommend(self, X_test):
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        results = []
        for i in range(len(predictions)):
            result = ""
            if predictions[i] == 0:
                result = "Draw"
            elif predictions[i] == 1:
                result = "Win"
            else:
                result = "Loss"
            
            recommendation = self.model.bet_recommendation(probabilities[i])
            results.append((result, recommendation))

        return results
# Use MatchPredictor to predict match outcomes and get betting recommendations
if __name__ == "__main__":  # This ensures the code below is only executed if the file is run directly
    # Initialize the models
    rf_model = RandomForestModel()
    xgb_model = XGBoostModel()

    # Load data
    X_test = pd.read_csv('X_test_saved.csv')
    X_train = pd.read_csv('X_train_saved.csv')
    y_test = pd.read_csv('y_test_saved.csv')
    y_train = pd.read_csv('y_train_saved.csv')

    rf_model.train(X_train, y_train)
    xgb_model.train(X_train, y_train)

  # Save models after training 
    xgb_model.save_model("trained_models/xgboost_model.joblib")
    rf_model.save_model("trained_models/random_forest_model.joblib")
   
    # Use MatchPredictor to make predictions with RandomForest
    rf_predictor = MatchPredictor(rf_model)
    rf_predictions = rf_predictor.predict_and_recommend(X_test)
    for i, (result, recommendation) in enumerate(rf_predictions):
        print(f"Match {i + 1}: Predicted Outcome - {result}, Betting Recommendation - {recommendation}")

    print("\n")

    # Use MatchPredictor to make predictions with XGBoost
    xgb_predictor = MatchPredictor(xgb_model)
    xgb_predictions = xgb_predictor.predict_and_recommend(X_test)
    for i, (result, recommendation) in enumerate(xgb_predictions):
        print(f"Match {i + 1}: Predicted Outcome - {result}, Betting Recommendation - {recommendation}")