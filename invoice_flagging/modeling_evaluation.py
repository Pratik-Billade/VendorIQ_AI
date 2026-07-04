from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using GridSearchCV to identify the
    optimal hyperparameter combination.

    Parameters:
        X_train (pd.DataFrame or np.ndarray): Training feature set.
        y_train (pd.Series or np.ndarray): Training target labels.

    Returns:
        GridSearchCV: Fitted GridSearchCV object containing the best model.
    """

    # Create the base Random Forest classifier
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    # Define the hyperparameter search space
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }

    # Use the F1-score as the evaluation metric during cross-validation
    scorer = make_scorer(f1_score)

    # Configure GridSearchCV to evaluate all parameter combinations
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    # Train the models and identify the best-performing configuration
    grid_search.fit(X_train, y_train)

    # Return the fitted GridSearchCV object
    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name):
    """
    Evaluate a classification model on the test dataset.

    Parameters:
        model: Trained classification model.
        X_test (pd.DataFrame or np.ndarray): Test feature set.
        y_test (pd.Series or np.ndarray): Actual class labels.
        model_name (str): Name of the model being evaluated.
    """

    # Generate predictions for the test data
    preds = model.predict(X_test)

    # Calculate the overall classification accuracy
    accuracy = accuracy_score(y_test, preds)

    # Generate a detailed classification report containing
    # precision, recall, F1-score, and support
    report = classification_report(y_test, preds)

    # Display the evaluation results
    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)