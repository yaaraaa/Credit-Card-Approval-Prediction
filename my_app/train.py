import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(data_path):
    """
    Loads and preprocesses the dataset, mapping categorical fields to numeric values,
    standardizing income, and splitting into training and test sets.

    Args:
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: A tuple containing the train and test DataFrames.

    Side Effects:
        Saves the fitted scaler to 'model/scaler.pkl' for later use during inference.
    """

    df = pd.read_csv(data_path)

    print(df['Credit_Card_Issuing'].value_counts())

    # map categorical data
    gender_mapping = {'Male': 1, 'Female': 0}
    car_mapping = {'Yes': 1, 'No': 0}
    housing_mapping = {'Yes': 1, 'No': 0}
    cc_mapping = {'Approved': 1, 'Denied': 0}

    df['Gender'] = df['Gender'].map(gender_mapping)
    df['Own_Car'] = df['Own_Car'].map(car_mapping)
    df['Own_Housing'] = df['Own_Housing'].map(housing_mapping)
    df['Credit_Card_Issuing'] = df['Credit_Card_Issuing'].map(cc_mapping)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # standardize numerical data
    scaler = StandardScaler()
    train_df['Income'] = scaler.fit_transform(train_df[['Income']])
    test_df['Income'] = scaler.transform(test_df[['Income']])

    # save the scaler for later use in inference
    joblib.dump(scaler, 'model/scaler.pkl')

    return train_df, test_df



def train_model(train_df, test_df, model_path):
    """
    Trains a logistic regression model on the provided training data and saves the model and data splits.

    Args:
        train_df (pd.DataFrame): Training data with features and target 'Credit_Card_Issuing'.
        test_df (pd.DataFrame): Test data with features and target.
        model_path (str): File path to save the trained model.

    Side Effects:
        Saves the trained model to `model_path`.
        Saves the training and test feature and target splits to the 'splits' directory as CSV files.
    """

    X_train = train_df.drop(columns=['Credit_Card_Issuing'])
    y_train = train_df['Credit_Card_Issuing']

    X_test = test_df.drop(columns=['Credit_Card_Issuing'])
    y_test = test_df['Credit_Card_Issuing']

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save model and split data for evaluation
    joblib.dump(model, model_path)
    X_train.to_csv('splits/X_train.csv', index=False)
    X_test.to_csv('splits/X_test.csv', index=False)
    y_train.to_csv('splits/y_train.csv', index=False)
    y_test.to_csv('splits/y_test.csv', index=False)

    print("Model trained and saved.")

if __name__ == "__main__":
    train_df, test_df = preprocess_data('data/credit_card_train.csv')
    train_model(train_df, test_df, 'model/model.pkl')
