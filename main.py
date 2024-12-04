import os
from secom_failure_detection.components.data_preprocessing import (
    load_data,
    preprocess_data,
)
from secom_failure_detection.components.model_training import train_model
from secom_failure_detection.components.model_evaluation import evaluate_model
from secom_failure_detection.components.model_training import save_model
from secom_failure_detection.config.configuration import Config
from secom_failure_detection import logger


def main():
    logger.info("Starting the failure detection process...")

    config = Config()
    data_path = config.get("data_path")
    label_path = f"{data_path}/secom_labels.data"

    # Load and preprocess the data
    data, labels = load_data(f"{data_path}/secom.data", label_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        data, labels, test_size=0.2, random_state=42
    )

    # Train the model and get the selected features
    model, X_train_selected, selector = train_model(
        X_train,
        y_train,
        n_estimators=config.get("n_estimators"),
        max_depth=config.get("max_depth"),
    )

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model and scaler
    save_model(model, scaler, config.get("model_save_path"))


if __name__ == "__main__":
    main()
