from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.

    Returns:
        LinearRegression: Trained Linear Regression model.
    """

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Return the trained model
    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree Regressor.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.
        max_depth (int): Maximum depth of the decision tree.

    Returns:
        DecisionTreeRegressor: Trained Decision Tree model.
    """

    # Create the Decision Tree model with a fixed random state
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Return the trained model
    return model


def train_random_forest(X_train, y_train, max_depth=6):
    """
    Train a Random Forest Regressor.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.
        max_depth (int): Maximum depth of each tree in the forest.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """

    # Create the Random Forest model
    model = RandomForestRegressor(
        max_depth=max_depth,
        random_state=42,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Return the trained model
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model using common performance metrics.

    Parameters:
        model: Trained regression model.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Actual target values.
        model_name (str): Name of the model being evaluated.

    Returns:
        dict: Dictionary containing MAE, RMSE, and R² score.
    """

    # Generate predictions on the test dataset
    preds = model.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, preds)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = root_mean_squared_error(
        y_test,
        preds,
    )

    # Calculate R² score and convert it to a percentage
    r2 = r2_score(y_test, preds) * 100

    # Display the evaluation results
    print(f"\n{model_name} Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}%")

    # Return the evaluation metrics as a dictionary
    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }