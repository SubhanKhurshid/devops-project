import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.feature_selection import SelectFromModel
from secom_failure_detection import logger
from sklearn.ensemble import RandomForestClassifier


def load_data(data_path, label_path):
    try:
        # Load the sensor data (secom.data)
        data = pd.read_csv(data_path, delimiter=" ", header=None)

        # Load the labels (secom_labels.data), only taking the first column
        labels = pd.read_csv(label_path, delimiter=" ", header=None, usecols=[0])

        # Convert labels to numeric, replacing any non-numeric values (if any)
        labels = pd.to_numeric(labels[0], errors="coerce")

        # Check for NaN values in labels and fill them (e.g., with the mean)
        if labels.isnull().any():
            print("Labels contain NaN values, attempting to fill with mean...")
            labels.fillna(labels.mean(), inplace=True)

        # Ensure no NaN values remain after filling
        if labels.isnull().any():
            raise ValueError("Labels contain NaN values after imputation.")

        # Return data and labels
        return data, labels

    except Exception as e:
        # Log the error and raise an exception if it occurs
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(data, labels, test_size=0.2, random_state=42):
    """
    Preprocess the data: fill missing values, split, scale features.
    """
    logger.info("Starting data preprocessing...")

    # Handle missing values by filling them with the column mean
    logger.info("Handling missing values...")
    imputer = SimpleImputer(strategy="mean")
    data = imputer.fit_transform(data)

    # Check for infinity values and replace them with NaN
    logger.info("Checking for infinity values...")
    data[np.isinf(data)] = np.nan  # Replace infinite values with NaN
    data = imputer.fit_transform(data)  # Re-impute after handling infinity

    # Ensure that there are no NaNs in the data
    logger.info("Checking for NaN values after preprocessing...")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        logger.error("Data contains NaN or infinite values after preprocessing!")
        raise ValueError("Data contains NaN or infinite values after preprocessing.")

    # Ensure that there are no NaNs in labels after imputation
    logger.info("Checking for NaN values in the labels...")
    if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
        logger.error("Labels contain NaN or infinite values!")
        raise ValueError("Labels contain NaN or infinite values.")

    # Split the data into training and test sets
    logger.info(
        f"Splitting data into train and test sets with test_size={test_size}..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    logger.info(
        f"Data split into {X_train.shape[0]} train samples and {X_test.shape[0]} test samples."
    )

    # Normalize the data
    logger.info("Scaling the data using StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Print the shapes of the data at different stages
    print(f"Shape of X_train after preprocessing: {X_train.shape}")
    print(f"Shape of X_test after preprocessing: {X_test.shape}")
    print(f"Number of features in the training data: {X_train.shape[1]}")

    logger.info("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test, scaler  # Do not return the selector here
