import joblib
from pathlib import Path
from data_preprocessing import (
    load_invoice_data,
    split_data,
    scale_features,
    apply_labels,
)
from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier,
)


# List of input features used for training the classifier
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

# Target variable indicating whether an invoice is flagged as risky
TARGET = "flag_invoice"


def main():
    """
    Execute the complete machine learning pipeline.

    The pipeline performs the following steps:
    1. Load and prepare the dataset.
    2. Split the data into training and testing sets.
    3. Scale the feature values.
    4. Train a Random Forest classifier using Grid Search.
    5. Evaluate the best-performing model.
    6. Save the trained model for future predictions.
    """

    project_root = Path.cwd().parent
    db_path = project_root / "data/inventory.db"

    # Load the invoice dataset from the database
    df = load_invoice_data(db_path)

    # Generate the target labels for invoice risk
    df = apply_labels(df)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET,
    )

    # Standardize the feature values using StandardScaler
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        project_root / "models/scaler.pkl",
    )

    # Train the Random Forest classifier with hyperparameter tuning
    grid_search = train_random_forest(
        X_train_scaled,
        y_train,
    )

    # Evaluate the best model found during Grid Search
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier",
    )

    # Save the trained model for future inference
    joblib.dump(
        grid_search.best_estimator_,
        project_root / "models/predict_flag_invoice.pkl",
    )


# Execute the pipeline only when this file is run directly
if __name__ == "__main__":
    main()