import sqlite3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_invoice_data(db_path: str):
    """
    Load invoice and purchase data from the SQLite database and create
    a feature dataset for model training.

    Returns:
        pd.DataFrame: Dataset containing invoice information along with
        aggregated purchase statistics.
    """

    # Establish a connection to the SQLite database
    conn = sqlite3.connect(db_path)

    # SQL query to aggregate purchase information and join it with invoice data
    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    # Execute the query and load the results into a DataFrame
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Return the prepared dataset
    return df


def create_invoice_risk_label(row):
    """
    Generate a binary risk label for an invoice.

    An invoice is flagged as risky if:
    - The invoice amount differs from the total purchase amount by more than $5.
    - The average receiving delay is greater than 10 days.

    Parameters:
        row (pd.Series): A single row from the dataset.

    Returns:
        int: 1 for risky invoice, 0 otherwise.
    """

    # Flag invoices with significant amount mismatches
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1

    # Flag invoices with long receiving delays
    if row["avg_receiving_delay"] > 10:
        return 1

    # Otherwise, mark the invoice as low risk
    return 0


def apply_labels(df):
    """
    Apply the invoice risk labeling function to every row in the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with the additional 'flag_invoice' column.
    """

    # Create the target label column
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)

    return df


def split_data(df, features, target):
    """
    Split the dataset into training and testing sets.

    Parameters:
        df (pd.DataFrame): Complete dataset.
        features (list): List of feature column names.
        target (str): Name of the target column.

    Returns:
        tuple:
            X_train, X_test, y_train, y_test
    """

    # Select the input features
    X = df[features]

    # Select the target variable
    y = df[target]

    # Split the dataset into training and testing sets
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def scale_features(X_train, X_test, scaler_path):
    """
    Standardize the training and testing features and save the fitted scaler.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        scaler_path (str): Path where the scaler will be saved.

    Returns:
        tuple:
            X_train_scaled, X_test_scaled
    """

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Learn scaling parameters from the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaling parameters to the test data
    X_test_scaled = scaler.transform(X_test)

    # Save the fitted scaler for future inference
    joblib.dump(scaler, scaler_path)

    # Return the scaled datasets
    return X_train_scaled, X_test_scaled