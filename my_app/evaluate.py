import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import warnings



def calculate_tpr_fpr(y_true, y_pred):
    """
    Calculates True Positive Rate (TPR) and False Positive Rate (FPR) based on the given true and predicted labels.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        tuple: TPR and FPR values.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 
    return tpr, fpr


def find_optimal_threshold(probs, true_labels, target_tpr, target_fpr):
    """
    Finds the optimal threshold for a given set of probabilities to achieve target TPR and FPR values.

    Args:
        probs (np.array): Array of predicted probabilities.
        true_labels (np.array): Array of true binary labels.
        target_tpr (float): Target true positive rate (TPR).
        target_fpr (float): Target false positive rate (FPR).

    Returns:
        float: The threshold that minimizes the difference between the calculated and target TPR/FPR.
    """

    best_threshold = 0.5
    best_diff = float("inf")

    # Test thresholds from 0.01 to 0.99
    for threshold in np.linspace(0.01, 0.99, 100):
        preds = (probs >= threshold).astype(int)

        tpr, fpr = calculate_tpr_fpr(true_labels, preds)
        
        # Compare TPR and FPR to the target values
        diff = abs(tpr - target_tpr) + abs(fpr - target_fpr)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

    return best_threshold

def find_optimal_thresholds(model, X_test, y_test, result_path='model/optimal_thresholds.json'):
    """
    Determines optimal gender-based thresholds for model predictions to balance TPR and FPR without 
    comprimising precision, and saves them to a JSON file.

    Args:
        model: Trained classification model with a `predict_proba` method.
        X_test (pd.DataFrame): Test set features, including "Gender".
        y_test (pd.Series): True labels for the test set.
        result_path (str): File path to save the optimal thresholds JSON.

    Side Effects:
        Saves a JSON file containing the optimal thresholds for each gender.
    """

    X_test_df = X_test.copy()
    X_test_df['y_true'] = y_test
    X_test_df['y_proba'] = model.predict_proba(X_test)[:, 1]

    # Separate probabilities by gender
    male_probs = X_test_df[X_test_df['Gender'] == 1]['y_proba']
    female_probs = X_test_df[X_test_df['Gender'] == 0]['y_proba']
    male_true = X_test_df[X_test_df['Gender'] == 1]['y_true']
    female_true = X_test_df[X_test_df['Gender'] == 0]['y_true']

    # Calculate initial precision for both groups
    initial_male_precision = precision_score(male_true, (male_probs >= 0.5).astype(int))
    initial_female_precision = precision_score(female_true, (female_probs >= 0.5).astype(int))
    total_precision = initial_male_precision + initial_female_precision

    # Calculate initial TPR and FPR for both groups at default threshold of 0.5
    male_tpr, male_fpr = calculate_tpr_fpr(male_true, (male_probs >= 0.5).astype(int))
    female_tpr, female_fpr = calculate_tpr_fpr(female_true, (female_probs >= 0.5).astype(int))

    # Calculate weighted TPR and FPR targets
    weighted_tpr = (initial_male_precision / total_precision) * male_tpr + \
                (initial_female_precision / total_precision) * female_tpr
    weighted_fpr = (initial_male_precision / total_precision) * male_fpr + \
                (initial_female_precision / total_precision) * female_fpr

    # Find optimal thresholds to match the weighted TPR and FPR for each group
    male_threshold = find_optimal_threshold(male_probs, male_true, weighted_tpr, weighted_fpr)
    female_threshold = find_optimal_threshold(female_probs, female_true, weighted_tpr, weighted_fpr)

    # Save the optimal thresholds
    thresholds = {
        'male_threshold': male_threshold,
        'female_threshold': female_threshold
    }

    with open(result_path, 'w') as f:
        json.dump(thresholds, f)

    print("Optimal thresholds determined and saved.")



def evaluate_performance(model, X_test, y_test, metrics_title, roc_title, apply_thresholds=None):
    """
    Evaluates model performance on the test set, calculates key metrics, and saves the results.

    Args:
        model: Trained classification model with a `predict_proba` method.
        X_test (pd.DataFrame): Test set features, including "Gender".
        y_test (pd.Series): True labels for the test set.
        metrics_title (str): Title for saving the metrics JSON file.
        roc_title (str): Title for saving the ROC curve image file.
        apply_thresholds (dict, optional): Custom thresholds for each gender ('male_threshold' and 'female_threshold').

    Side Effects:
        Saves performance metrics as JSON and the ROC curve as a PNG in 'train_results/performance/'.
    """

    X_eval = X_test.copy() 
    X_eval['y_proba'] = model.predict_proba(X_eval.drop(columns=['y_proba', 'y_pred'], errors='ignore'))[:, 1]

    # Apply default or custom thresholds
    if apply_thresholds:
        X_eval['y_pred'] = np.where(
            (X_eval['Gender'] == 1) & (X_eval['y_proba'] >= apply_thresholds['male_threshold']), 1,
            np.where((X_eval['Gender'] == 0) & (X_eval['y_proba'] >= apply_thresholds['female_threshold']), 1, 0)
        )
        y_pred = X_eval['y_pred']
    else:
        y_pred = (X_eval['y_proba'] >= 0.5).astype(int)

    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score
    }
    
    with open(f"train_results/performance/{metrics_title}.json", "w") as f:
        json.dump(metrics, f)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, X_eval['y_proba'])
    roc_auc = auc(fpr, tpr)

    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"train_results/performance/{roc_title}.png")

    print("Results for evaluation saved.")


def evaluate_fairness(model, X_test, y_test, title, apply_thresholds=None):
    """
    Evaluates fairness metrics for the model on the test set, calculating demographic parity, 
    equalized odds, and predictive parity differences by gender, and saves the results.

    Args:
        model: Trained classification model with a `predict_proba` method.
        X_test (pd.DataFrame): Test set features, including "Gender".
        y_test (pd.Series): True labels for the test set.
        title (str): Title for saving the fairness metrics JSON file.
        apply_thresholds (dict, optional): Custom thresholds for each gender ('male_threshold' and 'female_threshold').

    Side Effects:
        Saves fairness metrics as JSON in 'train_results/fairness/'.
    """

    # Predict on the entire test set
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Convert the test set and true labels to a DataFrame for easier analysis
    X_eval = X_test.copy()
    X_eval['y_true'] = y_test

    # Determine predictions based on thresholds
    if apply_thresholds:
        # Apply optimal thresholds per gender
        X_eval['y_pred'] = np.where(
            (X_eval['Gender'] == 1) & (y_proba >= apply_thresholds['male_threshold']), 1,
            np.where((X_eval['Gender'] == 0) & (y_proba >= apply_thresholds['female_threshold']), 1, 0)
        )
    else:
        # Use default threshold of 0.5
        X_eval['y_pred'] = (y_proba >= 0.5).astype(int)

    # Calculate Demographic Parity
    approval_rates = X_eval.groupby('Gender')['y_pred'].mean()
    demographic_parity_difference = abs(approval_rates[1] - approval_rates[0])

    # Calculate Equalized Odds (TPR and FPR differences)
    male_data = X_eval[X_eval['Gender'] == 1]
    female_data = X_eval[X_eval['Gender'] == 0]
    male_tpr, male_fpr = calculate_tpr_fpr(male_data['y_true'], male_data['y_pred'])
    female_tpr, female_fpr = calculate_tpr_fpr(female_data['y_true'], female_data['y_pred'])
    tpr_difference = abs(male_tpr - female_tpr)
    fpr_difference = abs(male_fpr - female_fpr)

    # Calculate Predictive Parity (Precision difference)
    male_precision = precision_score(male_data['y_true'], male_data['y_pred'])
    female_precision = precision_score(female_data['y_true'], female_data['y_pred'])
    predictive_parity_difference = abs(male_precision - female_precision)

    # Compile results
    fairness_metrics = {
        "Demographic Parity Difference": demographic_parity_difference,
        "Equalized Odds": {
            "TPR Difference": tpr_difference,
            "FPR Difference": fpr_difference
        },
        "Predictive Parity Difference": predictive_parity_difference,
        "Details": {
            "Approval Rates": {
                "Male": approval_rates[1],
                "Female": approval_rates[0]
            },
            "TPR": {
                "Male": male_tpr,
                "Female": female_tpr
            },
            "FPR": {
                "Male": male_fpr,
                "Female": female_fpr
            },
            "Precision": {
                "Male": male_precision,
                "Female": female_precision
            }
        }
    }

    with open(f'train_results/fairness/{title}.json', 'w') as f:
        json.dump(fairness_metrics, f, indent=4)


def evaluate_bias_variance(model, X_train, y_train, X_test, y_test, model_name="Model Learning Curve", save_path="train_results/bias_variance"):
    """
    Evaluates model bias and variance by plotting a learning curve and calculating cross-validation scores.

    Args:
        model: Trained classification model to evaluate.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name of the model for plotting.
        save_path (str): Directory path to save the learning curve and cross-validation results.

    Side Effects:
        Saves the learning curve as a PNG and cross-validation results as a JSON file in the specified path.
    """
    # Define the different sizes of the training set
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    validation_scores = []

    # Iterate over different training set sizes
    for train_size in train_sizes:
        subset_size = int(train_size * len(X_train))
        X_subset, y_subset = X_train[:subset_size], y_train[:subset_size]

        model.fit(X_subset, y_subset)

        train_score = model.score(X_subset, y_subset)
        validation_score = model.score(X_test, y_test)

        train_scores.append(train_score)
        validation_scores.append(validation_score)

    # Plot the learning curve
    plt.figure()
    plt.plot(train_sizes, train_scores, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, validation_scores, 'o-', color="red", label="Validation Score")
    plt.xlabel("Training Set Size (fraction)")
    plt.ylabel("Accuracy")
    plt.ylim(0.9, 1.0)  # Adjust as needed for more visible scaling
    plt.title(f"{model_name} - Learning Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f"{save_path}/learning_curve.png")
    plt.close()
    print(f"Learning curve saved")

    # Concatenate training and test data for cross-validation
    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)

    cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')

    cv_results = {
        "Cross-Validation Scores": cv_scores.tolist(),
        "Mean CV Accuracy": np.mean(cv_scores),
        "CV Standard Deviation": np.std(cv_scores)
    }

    with open(f"{save_path}/cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=4)

    print(f"Cross-validation results saved")



def evaluate_interpretability(model, X_train, path="train_results/interpretability"):
    """
    Evaluates the interpretability of the model by generating feature importance and SHAP values, 
    and saves the results.

    Args:
        model: Trained classification model with coefficients and SHAP compatibility.
        X_train (pd.DataFrame): Training set features used to calculate feature importance and SHAP values.
        path (str): Directory path to save interpretability results, including feature coefficients and SHAP plot.

    Side Effects:
        Saves model feature coefficients as JSON and a SHAP summary plot as a PNG file in the specified path.
    """
    
    # 1. Feature Importance via Logistic Regression Coefficients
    feature_names = X_train.columns 
    coefficients = {
        feature: coef for feature, coef in zip(feature_names, model.coef_[0])
    }
    
    # Save coefficients as JSON
    with open(path+"/coefficients.json", "w") as f:
        json.dump(coefficients, f, indent=4)
    print("Feature coefficients saved")
    
    # 2. SHAP Interpretability
    # Initialize SHAP explainer and calculate SHAP values
    explainer_shap = shap.Explainer(model, X_train)
    shap_values = explainer_shap(X_train)

    # Create and save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig(path+"/shap_summary.png")
    plt.close()
    print("SHAP summary plot saved")


def lime_interpretability(model, X_train, X_test, idx, path="train_results/interpretability"):
    """
    Generates a LIME explanation for a specified test instance and saves the explanation as an HTML file.

    Args:
        model: Trained classification model compatible with LIME.
        X_train (pd.DataFrame): Training set features used to initialize the LIME explainer.
        X_test (pd.DataFrame): Test set features containing the instance to explain.
        idx (int): Index of the instance in X_test to explain with LIME.
        path (str): Directory path to save the LIME explanation HTML file.

    Side Effects:
        Saves the LIME explanation for the specified instance as an HTML file in the specified path.

    Returns:
        None
    """
    # Suppress warnings for LIME
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Initialize LIME explainer
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=['Class 0', 'Class 1'],  # Adjust to reflect your actual class labels
            mode='classification'
        )

        # Explain an instance from X_test
        exp = explainer_lime.explain_instance(X_test.iloc[idx], model.predict_proba)
        
        # Save the LIME explanation as an HTML file
        exp.save_to_file(path+f"/lime_explanation_{idx}.html")
        print("LIME explanation saved")



if __name__ == "__main__":
    # Load model and data
    model = joblib.load("model/model.pkl")
    X_train = pd.read_csv("splits/X_train.csv")
    y_train = pd.read_csv("splits/y_train.csv").values.ravel()
    X_test = pd.read_csv("splits/X_test.csv")
    y_test = pd.read_csv("splits/y_test.csv").values.ravel()
    
    # Evaluate performance of model using default thresholds
    evaluate_performance(
        model, X_test, y_test, 
        metrics_title="metrics_default_threshold", roc_title="roc_default_threshold"
    )

    # Evaluate bias and variance of model
    evaluate_bias_variance(model, X_train, y_train, X_test, y_test, model_name="Logistic Regression")


    # Evaluate fairness of model
    evaluate_fairness(model, X_test, y_test, "fairness_default_thresholds")
    
    # Find and save optimal thresholds
    find_optimal_thresholds(model, X_test, y_test)
    
    # Load optimal thresholds
    with open("model/optimal_thresholds.json", "r") as f:
        optimal_thresholds = json.load(f)

    evaluate_fairness(
        model, X_test, y_test, 
        "fairness_optimal_thresholds", optimal_thresholds
    )
    
    # Evaluate using optimal thresholds
    evaluate_performance(
        model, X_test, y_test, 
        metrics_title="metrics_optimal_thresholds", roc_title="roc_optimal_threshold", 
        apply_thresholds=optimal_thresholds
    )

    # Evaluate interpretability
    evaluate_interpretability(model, X_train)
    lime_interpretability(model, X_train, X_test, 1)
