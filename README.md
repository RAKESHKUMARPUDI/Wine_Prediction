# Wine Quality Classification Project

##Live Streamlit App Link
https://wineprediction-n79kbcapa6dwwdnohrmbp2.streamlit.app/

## a. Problem Statement
The goal of this project is to classify red wines as "Good" (Score >= 7) or "Bad" (Score < 7) 
based on their chemical properties using various Machine Learning algorithms.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository (Wine Quality - Red)
- **Instances:** 1,599
- **Features:** 12 (Fixed acidity, Volatile acidity, Citric acid, Residual sugar, Chlorides, 
  Free sulfur dioxide, Total sulfur dioxide, Density, pH, Sulphates, Alcohol, Quality)

## c. Models used: Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.865 | 0.887 | 0.590| 0.276| 0.376 | 0.340 |
| Decision Tree | 0.871 | 0.722 | 0.571 | 0.510 | 0.539 | 0.466 |
| kNN | 0.878 | 0.845 | 0.633 | 0.404 | 0.493 | 0.442 |
| Naive Bayes | 0.846 | 0.860 | 0.486 | 0.787 | 0.601 | 0.536 |
| Random Forest | 0.900 | 0.942 | 0.727 | 0.510 | 0.600 | 0.555 |
| XGBoost | 0.903 | 0.940 | 0.700 | 0.595 | 0.643 | 0.590 |


## d. Observations on Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Provides a solid baseline but struggles with non-linear chemical interactions. |
| Decision Tree | Captures local patterns well but shows signs of slight overfitting on noise. |
| kNN | Sensitive to feature scaling; performs moderately well by looking at similar wines. |
| Naive Bayes | Highest Recall; flags most "Good" wines but has many False Positives. |
| Random Forest | High AUC and Stability; excellent at handling outliers in chemical data. |
| XGBoost | Best overall performance (Accuracy/MCC); handles complex relationships most effectively. |
