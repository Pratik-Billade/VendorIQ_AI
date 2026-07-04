import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str):
    """
    Load the vendor_invoice table from a SQLite database into a pandas DataFrame.

    Parameters:
        db_path (str): Path to the SQLite database file.

    Returns:
        pd.DataFrame: DataFrame containing all records from the vendor_invoice table.
    """

    # Establish a connection to the SQLite database
    conn = sqlite3.connect(db_path)

    # SQL query to retrieve all rows from the vendor_invoice table
    query = "SELECT * FROM vendor_invoice"

    # Execute the query and load the results into a pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Return the loaded data
    return df


def prepare_features(df: pd.DataFrame):
    """
    Separate the feature(s) and target variable for machine learning.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the dataset.

    Returns:
        tuple:
            X (pd.DataFrame): Feature DataFrame containing the 'Dollars' column.
            y (pd.Series): Target variable containing the 'Freight' column.
    """

    # Select the input feature(s)
    X = df[["Dollars"]]

    # Select the target variable
    y = df["Freight"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.
        test_size (float): Fraction of data to reserve for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple:
            X_train, X_test, y_train, y_test
    """

    # Randomly split the data into training and testing sets
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )