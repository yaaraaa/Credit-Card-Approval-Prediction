from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionSerializer
import joblib
import pandas as pd
import json
import numpy as np

# Load the trained model and optimal thresholds
model = joblib.load("my_app/model/model.pkl")
with open("my_app/model/optimal_thresholds.json", "r") as f:
    optimal_thresholds = json.load(f)

def preprocess_input(data):
    """
    Converts input data to a DataFrame, maps categorical fields to numeric values,
    and standardizes numerical fields using the saved scaler.

    Args:
        data (dict or list of dict): Input data with fields for "Gender", "Own_Car", "Own_Housing", and "Income".

    Returns:
        pd.DataFrame: Processed DataFrame with categorical fields mapped and numerical fields standardized.
    """

    df = pd.DataFrame(data)

    # Map categorical fields
    gender_mapping = {'Male': 1, 'Female': 0}
    car_mapping = {'Yes': 1, 'No': 0}
    housing_mapping = {'Yes': 1, 'No': 0}

    df['Gender'] = df['Gender'].map(gender_mapping)
    df['Own_Car'] = df['Own_Car'].map(car_mapping)
    df['Own_Housing'] = df['Own_Housing'].map(housing_mapping)

    # Load the scaler and normalize the Income column
    scaler = joblib.load('my_app/model/scaler.pkl')
    df['Income'] = scaler.transform(df[['Income']])

    return df

@api_view(['POST'])
def predict(request):
    """
    Handles POST requests for predicting credit card approval using a trained model.

    The function validates the input data, preprocesses it, applies the model to predict probabilities,
    and returns predictions based on gender-specific thresholds.

    Args:
        request (HttpRequest): A POST request with JSON data containing input features.

    Returns:
        Response: A JSON response with a list of predictions if the input is valid,
                  or validation errors if the input is invalid.
    """

    serializer = PredictionSerializer(data=request.data)
    if serializer.is_valid():
        df = preprocess_input(serializer.validated_data)

        probabilities = model.predict_proba(df)[:, 1]

        # Apply gender-based thresholds for predictions
        predictions = []
        for i, prob in enumerate(probabilities):
            gender = df['Gender'].iloc[i]
            threshold = optimal_thresholds['male_threshold'] if gender == 1 else optimal_thresholds['female_threshold']
            prediction = int(prob >= threshold)
            predictions.append(prediction)

        return Response({"predictions": predictions}, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
