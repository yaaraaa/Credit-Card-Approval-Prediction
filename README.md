# Credit Card Approval Prediction App
This project is a Django-based web application that predicts credit card approvals using a trained logistic regression model. Beyond simple prediction, the system addresses real-world challenges like class imbalance, bias and variance trade-offs, and fair decision-making across demographics.

## Python Version
- 3.10.0

## Modules
- `/model`: Directory containing the saved trained model, calculated optimal prediction thresholds and scaler, used for inference.
- `/train_results`: Stores results of performance, fairness, and interpretability evaluations, including JSON files and plots.
- `train.py`: Script for preprocessing data, training the model, and saving the model and scaler.
- `evaluate.py`: Evaluates model performance, fairness, bias/variance, and interpretability, saving results in JSON and visual formats, also computes gender-specific thresholds to improve fairness, saved for use in inference.
- `serializers.py`: Define the structure and validation for incoming data.
- `views.py`: Django API endpoint for handling credit card approval prediction requests based on applicant data.
- `requirements.txt`: Contains all necessary dependencies for running the application.


## Setup

### Install required dependencies

```
pip install -r requirements.txt
```

### Run the Django App
```
python manage.py runserver
```

## API Endpoints
- `POST /predict`: Predicts credit card approval based on applicant details.
  - `Body`: 
    - Num_Children (int or list): Number of children of the applicant(s).
    - Gender (str or list): Gender of the applicant(s) ("Male" or "Female").
    - Income (float or list): Income of the applicant(s).
    - Own_Car (str or list): Car ownership status of the applicant(s) ("Yes" or "No").
    - Own_Housing (str or list): Housing ownership status of the applicant(s) ("Yes" or "No").
- `Response`: JSON with predictions field, which is a list of predicted approvals (1 for approved, 0 for denied).


