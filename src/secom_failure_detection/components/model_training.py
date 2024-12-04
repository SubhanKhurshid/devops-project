import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from secom_failure_detection import logger
import joblib
from sklearn.feature_selection import SelectFromModel


def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    logger.info(
        f"Training model with {n_estimators} estimators and max depth of {max_depth}..."
    )

    # Train the RandomForest model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Perform feature selection using the trained model
    logger.info("Performing feature selection using the trained model...")
    selector = SelectFromModel(
        model, max_features=50
    )  # Select top 50 features, adjust as needed
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Return the trained model and the selected data
    logger.info(f"Model trained with {X_train_selected.shape[1]} features selected.")

    return model, X_train_selected, selector


def save_model(model, scaler, model_path):
    """
    Saves the trained model and scaler to the specified path.
    """
    logger.info(
        f"Saving model to {model_path}/model.pkl and scaler to {model_path}/scaler.pkl."
    )
    joblib.dump(model, os.path.join(model_path, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))
    logger.info("Model and scaler saved successfully.")
