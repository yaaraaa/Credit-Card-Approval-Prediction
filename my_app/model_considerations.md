### Perfomance
- Several key metrics were selected: accuracy, precision, sensitivity, specificity, F1 score, and AUC-ROC. These metrics were chosen to provide a comprehensive view of the model's performance, especially given the class imbalance in the data. Cost-sensitve learning was also used to give the minority class more importance.

- Performance on the validation set revealed an ROC curve with an AUC of 1.0, indicating that the model is performing exceptionally well at classification. The remaining metrics also confirm the model’s effectiveness in handling class imbalance.

### Bias/Variance
- Logistic regression was selected as it is a low variance model, so it reduces risk of overfitting.

- Bias and variance were evaluated using:
    - `Learning Curves:` To observe how accuracy changes on training and validation sets with different dataset sizes, identifying potential underfitting or overfitting.
    - `Cross-Validation:` To assess performance stability across different data splits.

- The learning curve shows that both training and validation scores are consistently high and very close, around 97-98% accuracy, across different training set sizes. This indicates a good balance between bias and variance.

- The cross-validation results further confirm this stability, with a mean accuracy of 97.24% and a very low standard deviation of 0.0004. This low standard deviation suggests low variance, while the high mean accuracy indicate low bias, as desired.

### Fairness
- We evaluated fairness using:
    - `Demographic Parity:` Checked similar approval rates across genders.
    - `Equalized Odds:` Checked that TPR and FPR were consistent between genders.
    - `Predictive Parity:` Verified that precision was balanced.

- The fairness results show a TPR gap of 3.5% and an FPR gap of 1.8% between genders, meaning females may be slightly disadvantaged in approvals. Additionally, a 37.3% disparity in approval rates suggests females face reduced approval chances.

- To improve fairness, we set separate approval thresholds for males and females. First, target true positive and false positive rates for each gender were calculated, weighted by how precise the model was for each group. Then, various thresholds were tested to match these targets, ensuring balanced approval rates across genders and reducing unfairness in the model’s decisions.

- With optimized thresholds, gender fairness improved significantly: TPR and FPR are nearly identical for males and females. Although female precision dropped slightly to 88.4%, sensitivity (TPR) rose to 96.7%, meaning fewer missed approvals. This trade-off is acceptable here, as improving sensitivity is crucial to providing fair opportunities across genders.

### Interpretability
- Logistic regression was chosen for its inherent interpretability, as feature coefficients reveal the impact of each input on predictions. To further enhance transparency, we used SHAP Values, and LIME. 

- The SHAP summary and feature importance coefficients shows Income as the most influential factor in approvals, followed by Gender and Own_Housing. Own_Car has a minor impact, and Num_Children is nearly negligible.

- Lime produced explaination at the instance level, so it varied for each sample.
























