from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from secom_failure_detection import logger
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance and generates evaluation metrics.
    """
    logger.info("Evaluating model performance...")

    # Convert y_test to a 1D numpy array
    y_test = np.array(y_test).ravel()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy}")

    # Classification report
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return cm
