# AI-Powered-Fraud-Detection-in-Auto-Insurance-Predictive-Modeling-for-Smarter-Claims-Management

This project aims to develop and implement an AI-powered predictive model to detect fraudulent claims in the auto insurance sector. By leveraging machine learning techniques, the goal is to enhance the efficiency and accuracy of claims management, reduce financial losses due to fraud, and streamline the investigation process for suspicious claims. This initiative directly addresses the problem of rising fraudulent activities impacting insurance companies' profitability and operational costs.

## Problem Statement

The core problem addressed is "AI-Powered Fraud Detection in Auto Insurance: Predictive Modeling for Smarter Claims Management". Auto insurance fraud is a significant challenge, leading to substantial financial losses for insurers and increased premiums for honest policyholders. Manual detection methods are often inefficient, time-consuming, and prone to human error. This project seeks to build a robust predictive model that can accurately identify fraudulent claims, enabling insurance companies to manage claims more intelligently and proactively.

## Dataset

The analysis is based on a comprehensive Auto Insurance Fraud Claims Dataset. The dataset is comprised of information from multiple sources.

**Data Dictionary Summary:**
The dataset includes a wide array of features crucial for fraud detection, encompassing:
* **Claim Information:** `Claim_ID`, `Claims_Date`, `Total_Claim`, `Injury_Claim`, `Property_Claim`, `Vehicle_Claim`.
* **Policy Details:** `Policy_Num`, `Bind_Date1`, `Customer_Life_Value1`, `Policy_State`, `Policy_Start_Date`, `Policy_Expiry_Date`, `Policy_BI`, `Policy_Ded`, `Policy_Premium`, `Umbrella_Limit`.
* **Insured Person Details:** `Age_Insured`, `Gender`, `Education`, `Occupation`, `Hobbies`, `Insured_Relationship`, `Capital_Gains`, `Capital_Loss`, `Insured_Zip`.
* **Accident Details:** `Accident_Date`, `Accident_Type`, `Collision_Type`, `Accident_Severity`, `authorities_contacted`, `Acccident_State`, `Acccident_City`, `Accident_Location`, `Accident_Hour`, `Num_of_Vehicles_Involved`, `Property_Damage`, `Bodily_Injuries`, `Witnesses`, `Police_Report`.
* **Vehicle Details:** `Auto_Make`, `Auto_Model`, `Auto_Year`, `Vehicle_Color`, `Vehicle_Cost`, `Annual_Mileage`, `DiffIN_Mileage`, `Vehicle_Registration`.
* **Discounts and Flags:** `Low_Mileage_Discount`, `Commute_Discount`.
* **Target Variable:** `Fraud_Ind` (Indicator if the claim is suspected or confirmed fraud - Y/N).
* **Internal Verification:** `Check_Point`.

The dataset was provided across four CSV files:
1.  `Auto_Insurance_Fraud_Claims_File01.csv`
2.  `Auto_Insurance_Fraud_Claims_File02.csv`
3.  `Auto_Insurance_Fraud_Claims_File03.csv`
4.  `Auto_Insurance_Fraud_Claims_Results_Submission.csv` (This file is likely for format reference or result submission).

**Training Data Construction:** The primary training dataset was constructed by merging `Auto_Insurance_Fraud_Claims_File01.csv` and `Auto_Insurance_Fraud_Claims_File02.csv` based on common identifiers. During this initial merge and preprocessing phase, **5 new variables were engineered and added** to enhance the feature set. [cite_start]These additions provide a richer context for predictive modeling. [cite: 1]

**Testing Data:** `Auto_Insurance_Fraud_Claims_File03.csv` serves as the independent test set for evaluating the trained models and predicting the "Fraud_Ind" target variable.

## Data Preprocessing Steps

The raw data underwent a rigorous preprocessing pipeline to ensure its quality and suitability for machine learning.

1.  **Data Loading and Initial Inspection:**
    * All relevant CSV files (`Auto_Insurance_Fraud_Claims_File01.csv`, `Auto_Insurance_Fraud_Claims_File02.csv`, `Auto_Insurance_Fraud_Claims_File03.csv`) were loaded into individual pandas DataFrames.
    * Initial checks were performed for `df.head()`, `df.info()`, `df.shape`, and duplicate rows to understand the raw data structure and identify immediate issues.

2.  **Data Merging for Training Set:**
    * `Auto_Insurance_Fraud_Claims_File01.csv` and `Auto_Insurance_Fraud_Claims_File02.csv` were merged into a single comprehensive DataFrame.
    * The merging strategy involved using common key columns, primarily `Claim_ID` and potentially `Policy_Num`, to accurately combine information across the different data segments. An `outer` merge was utilized to retain all possible records and minimize data loss during the integration process.

3.  **Handling Missing Values (Training Data):**
    * Identified the count and percentage of missing values for each column in the merged training DataFrame.
    * **Numerical Columns:** Missing values in columns like `Customer_Life_Value1`, `Umbrella_Limit`, `Capital_Gains`, `Capital_Loss`, `Vehicle_Cost`, `Annual_Mileage`, `DiffIN_Mileage`, and various claim amounts (`Total_Claim`, `Injury_Claim`, `Property_Claim`, `Vehicle_Claim`) were imputed using the **median** to maintain robustness against potential outliers.
    * **Categorical Columns:** Missing values in columns such as `Policy_State`, `Gender`, `Education`, `Occupation`, `Hobbies`, `Insured_Relationship`, `Garage_Location`, `Accident_Type`, `Collision_Type`, `Accident_Severity`, `authorities_contacted`, `Acccident_State`, `Acccident_City`, `Property_Damage`, `Police_Report`, `Auto_Make`, `Auto_Model`, `Vehicle_Color`, `Low_Mileage_Discount`, `Commute_Discount`, and `Fraud_Ind` were imputed using the **mode** (most frequent category).
    * **Date/Time Columns:** Columns like `Bind_Date1`, `Policy_Start_Date`, `Policy_Expiry_Date`, `Accident_Date`, `DL_Expiry_Date`, and `Claims_Date` were converted to datetime objects. Critical dates (`Accident_Date`, `Claims_Date`, `Policy_Start_Date`) with missing values were handled by dropping the corresponding rows if the count was small to maintain data integrity for temporal calculations. For `DL_Expiry_Date`, a flag `DL_Expiry_Date_IsMissing` was created, and missing values were filled with a placeholder date (e.g., the mode date or a future date) to allow for feature engineering.

4.  **Specific Data Type Corrections:**
    * A custom function `parse_bi_limit` was implemented to correctly handle `Policy_BI` values that might be in a "X/Y" string format (e.g., "100000/300000"). This involved splitting the string and summing the parts, with robust error handling to convert unparseable values to `NaN` and then impute them with the median.
    * Explicit steps were taken to ensure `Accident_Hour` and `Bodily_Injuries` columns are treated as numerical. Any non-numeric values were coerced to `NaN` and subsequently filled with the median (for numerical types) or zero (as a fallback).

5.  **Robust Date Feature Engineering (Training Data):**
    * Temporal features such as Year, Month, Day, Day of Week, and Quarter were extracted from all relevant date columns.
    * Crucial duration-based features were engineered:
        * `Policy_Duration_Days`: Calculated as the difference between `Policy_Expiry_Date` and `Policy_Start_Date`.
        * `Days_to_Accident_from_PolicyStart`: Measures the time from `Policy_Start_Date` to `Accident_Date`.
        * `Days_to_Claim_from_Accident`: Represents the duration between `Accident_Date` and `Claims_Date`.
        * `Time_Since_DL_Expiry_at_Accident`: Calculated as days from `DL_Expiry_Date` to `Accident_Date`, specifically capturing if the driver's license was expired at the time of the accident.
    * Missing values in date-derived features were handled by imputing with the median or zero, and a specific flag (`DL_Expiry_Date_IsMissing`) was created for missing `DL_Expiry_Date` values.

6.  **Advanced Numerical Ratio Features (Training Data):** Several important ratio features were engineered to capture proportional relationships, which are often highly indicative in fraud detection:
    * `Claim_to_Premium_Ratio`: `Total_Claim` divided by `Policy_Premium`.
    * `Injury_Claim_Ratio`: `Injury_Claim` as a proportion of `Total_Claim`.
    * `Property_Claim_Ratio`: `Property_Claim` as a proportion of `Total_Claim`.
    * `Vehicle_Claim_Ratio`: `Vehicle_Claim` as a proportion of `Total_Claim`.
    * `Veh_Cost_Claim_Ratio`: `Vehicle_Claim` relative to `Vehicle_Cost`.
    * Robust division by zero handling was implemented for all ratio calculations.

7.  **Outlier Handling via Capping (Training Data):** A basic outlier treatment strategy using **IQR-based capping** was applied to numerical features. Values falling outside the 1.5\*IQR range (below Q1 or above Q3) were clipped to the respective lower or upper bounds, mitigating the impact of extreme values. Identifier columns and inherently bounded temporal features were excluded from this process.

8.  **Feature Removal (Training Data):** Non-predictive identifier columns such as `Claim_ID`, `Policy_Num`, `Vehicle_Registration`, and `Check_Point` were explicitly dropped after their utility in merging was completed, to prevent data leakage and improve model efficiency.

9.  **Comprehensive Categorical Encoding (Training Data):** `pd.get_dummies` (One-Hot Encoding with `drop_first=True`) was applied to a broad set of identified categorical columns. A final check ensured all columns in the feature set `X` were converted to numeric type, coercing any lingering non-numeric values to `NaN` and filling them with zeros.

10. **Target Encoding (Training Data):** The `Fraud_Ind` target variable (Y/N) was transformed into a binary numerical format (1/0) using `LabelEncoder`.

11. **Feature Scaling (Training Data):** All numerical features in `X` were scaled using `StandardScaler` to bring them to a uniform scale (mean 0, variance 1). This is crucial for distance-based algorithms (like Logistic Regression and K-Nearest Neighbors) and can improve convergence for others.

12. **Processing of Test Data (`Auto_Insurance_Fraud_Claims_File03.csv`):**
    * `Auto_Insurance_Fraud_Claims_File03.csv` will undergo the **exact same preprocessing steps** as the training data (data type corrections, feature engineering, outlier handling, feature removal, categorical encoding using columns learned from training data, and feature scaling using the `StandardScaler` fitted on the training data). This ensures consistency between training and test sets.
    * The `Fraud_Ind` column in this test set will serve as the true labels for evaluating the model's predictions.

## Predictive Modeling

* **Diverse Model Selection:** A wide array of machine learning classification algorithms was employed to identify the most effective model for fraud detection, trained on the merged `File01` and `File02` dataset:
    * Logistic Regression
    * Decision Tree Classifier
    * Random Forest Classifier
    * Gradient Boosting Classifier
    * K-Nearest Neighbors (KNeighborsClassifier)
    * Naive Bayes (GaussianNB)
    * XGBoost Classifier
    * LightGBM Classifier
    * CatBoost Classifier
    * AdaBoost Classifier
* **Addressing Class Imbalance in Models:** To counteract the inherent imbalance where fraudulent claims are a minority, specific strategies were applied during model training:
    * For **Logistic Regression, Decision Tree, and Random Forest**, the `class_weight='balanced'` parameter was utilized.
    * For **XGBoost and LightGBM**, `scale_pos_weight` was dynamically calculated as the ratio of negative to positive samples in the training set and passed to the classifier.
    * For **CatBoost**, `auto_class_weights='Balanced'` was employed.
    * For models like Gradient Boosting, K-Nearest Neighbors, and Gaussian Naive Bayes that do not directly support `class_weight` parameters in the same way, future iterations might consider data resampling techniques (e.g., SMOTE) prior to training or custom cost-sensitive learning approaches.
* **Comprehensive Evaluation Metrics:** Each model's performance was rigorously evaluated using a suite of 8 key metrics, critical for assessing fraud detection models on the independent test set (`Auto_Insurance_Fraud_Claims_File03.csv`):
    * **Accuracy:** Overall correctness.
    * **Precision (weighted):** Proportion of positive identifications that were actually correct.
    * **Recall (weighted):** Proportion of actual positives that were identified correctly.
    * **F1-Score (weighted):** Harmonic mean of precision and recall.
    * **ROC AUC Score:** Measures the ability of the model to distinguish between classes. Particularly useful for imbalanced data.
    * **Confusion Matrix:** Provides a detailed breakdown of True Positives, True Negatives, False Positives, and False Negatives.
    * **Full Classification Report:** Offers precision, recall, and F1-score for each class (fraud and non-fraud).
    * **Cohen's Kappa Score:** A robust measure of agreement between predicted and actual classifications, accounting for chance agreement.

## Model Evaluation

Upon training and evaluating the various classification models on the combined `File01` and `File02` dataset and testing on `File03`, the performance varied across algorithms, particularly in their ability to detect the minority (fraudulent) class. Key findings include:

* **Exceptional Performance by Tree-Based and Ensemble Models (with a caveat):**
    * **Decision Tree Classifier, Random Forest Classifier, XGBoost Classifier, LightGBM Classifier, and CatBoost Classifier** all demonstrated **perfect (1.0000)** scores across Accuracy, Precision, Recall, F1-Score, and ROC AUC, with a Cohen's Kappa of 1.0000. While these metrics indicate ideal classification, such perfect scores in a real-world fraud detection scenario are highly unusual and **warrant further investigation for potential data leakage or a perfect correlation** between features and the target variable within the dataset. This might suggest an issue where the target variable is directly or indirectly determinable from one or more features, which would not generalize to new, unseen data in a real production environment.

* **Substantial Performance by Gradient Boosting:**
    * The **Gradient Boosting Classifier** achieved a strong **Accuracy of 0.9517**, with a **Recall of 0.87** for the fraud class and a **Precision of 0.94**, resulting in an **F1-Score of 0.90**. Its **ROC AUC of 0.9840** indicates excellent discriminative ability. This model provides a more realistic high-performance benchmark given the potential for data anomalies in the 1.0000 scores.

* **Moderate Performance by K-Nearest Neighbors and AdaBoost:**
    * **K-Nearest Neighbors** showed **Accuracy of 0.8297** and a fraud class **Recall of 0.54** (F1-score of 0.62), suggesting it captures some patterns but might be sensitive to the high dimensionality or feature scaling.
    * **AdaBoost Classifier** achieved an **Accuracy of 0.8127** with a fraud class **Recall of 0.45** (F1-score of 0.55), indicating moderate performance in identifying fraudulent claims.

* **Challenges with Simpler Models and Imbalance:**
    * **Logistic Regression** and **Naive Bayes (GaussianNB)** exhibited very poor performance in detecting the minority (fraudulent) class, with **Recall scores of 0.00** and **0.07** respectively for the 'Y' class. Their Cohen's Kappa scores near zero also reflect minimal agreement beyond chance. This highlights the severe impact of class imbalance on these models even with `class_weight='balanced'` for Logistic Regression, underscoring the necessity of more advanced imbalance handling techniques or robust models for this problem.

* **Specific Metrics:**

| Model Name                      | Accuracy | Precision (Fraud Class) | Recall (Fraud Class) | F1-Score (Fraud Class) | ROC AUC | Cohen's Kappa |
| :------------------------------ | :------- | :---------------------- | :------------------- | :--------------------- | :------ | :------------ |
| Logistic Regression             | 0.7467   | 0.0000                  | 0.0000               | 0.0000                 | 0.5579  | 0.0000        |
| Decision Tree Classifier        | 1.0000   | 1.0000                  | 1.0000               | 1.0000                 | 1.0000  | 1.0000        |
| Random Forest Classifier        | 1.0000   | 1.0000                  | 1.0000               | 1.0000                 | 1.0000  | 1.0000        |
| Gradient Boosting Classifier    | 0.9517   | 0.9400                  | 0.8700               | 0.9000                 | 0.9840  | 0.8692        |
| K-Nearest Neighbors (KNeighbors)| 0.8297   | 0.7200                  | 0.5400               | 0.6200                 | 0.8463  | 0.5098        |
| Naive Bayes (GaussianNB)        | 0.7394   | 0.4100                  | 0.0700               | 0.1100                 | 0.5680  | 0.0467        |
| XGBoost Classifier              | 1.0000   | 1.0000                  | 1.0000               | 1.0000                 | 1.0000  | 1.0000        |
| LightGBM Classifier             | 1.0000   | 1.0000                  | 1.0000               | 1.0000                 | 1.0000  | 1.0000        |
| CatBoost Classifier             | 1.0000   | 1.0000                  | 1.0000               | 1.0000                 | 1.0000  | 1.0000        |
| AdaBoost Classifier             | 0.8127   | 0.7000                  | 0.4500               | 0.5500                 | 0.9038  | 0.4399        |

*(Note: Precision, Recall, and F1-Score for the Fraud Class (Y) are extracted from the "Full Classification Report" for better focus on the minority class performance. Weighted averages are also available in the report.)*

## Conclusion and Future Work

This project successfully demonstrates the immense potential of AI-powered predictive modeling for auto insurance fraud detection. By meticulously preprocessing a diverse dataset and leveraging various machine learning algorithms with careful attention to class imbalance, we've built a foundational system for smarter claims management. The identification of fraudulent claims through predictive models offers a proactive approach, enabling insurers to allocate investigative resources more effectively, mitigate financial losses, and ultimately enhance operational efficiency. This shift from reactive to predictive detection marks a substantial improvement over traditional manual methods.

However, the perfect scores achieved by several advanced tree-based models (Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost) are highly indicative of a potential data leakage issue or a direct, non-generalizable correlation present within the current dataset. While exciting on paper, such performance rarely translates to real-world scenarios. It underscores the critical need for further data validation and careful feature engineering to ensure the model learns generalizable patterns rather than memorizing the training data. The more realistic, yet still strong, performance of Gradient Boosting provides a more reliable baseline for immediate application.

### Future Work and Uniqueness:

1.  **Anomaly Detection Integration for Novel Fraud (Uniqueness):** While supervised learning excels at detecting known fraud patterns, future work will integrate unsupervised anomaly detection techniques (e.g., Isolation Forest, Autoencoders, One-Class SVM). This would allow the system to identify novel or emerging fraud schemes that may not be represented in the historical labeled data, providing an extra layer of defense against sophisticated fraudsters and ensuring adaptability to evolving tactics.
2.  **Explainable AI (XAI) for Actionable Insights:** Implement Explainable AI (XAI) techniques (e.g., SHAP, LIME) to interpret model predictions. This is crucial for insurance investigators, allowing them to understand *why* a specific claim was flagged as fraudulent, providing actionable insights for their investigations (e.g., "This claim is suspicious due to unusually high vehicle claim amount relative to vehicle cost and a short time between accident and claim date"). This transparency is a key differentiator for practical deployment and building trust in the AI system.
3.  **Real-time Fraud Scoring and Dynamic Alerting:** Develop an architecture for real-time scoring of claims as they are filed. This would involve deploying the best-performing, validated model as an API endpoint, allowing claims to be scored instantly and triggering immediate, prioritized alerts for suspicious cases. This significantly reduces the window for fraudulent payouts and enables rapid response.
4.  **Graph Neural Networks (GNNs) for Organized Fraud Detection (Uniqueness):** Explore the application of Graph Neural Networks to detect organized fraud rings and collusion. By constructing a knowledge graph where nodes represent entities (insureds, vehicles, garages, repair shops, witnesses) and edges represent their relationships, GNNs can identify suspicious clusters, uncommon connections, or patterns of interconnected fraudulent activities that are extremely difficult to spot with traditional tabular methods. This offers a cutting-edge approach to expose complex, multi-party fraud networks.
5.  **Reinforcement Learning for Adaptive Policy Management:** Investigate using reinforcement learning to dynamically adjust insurance policy parameters or claim processing rules based on the evolving fraud landscape and the model's detection capabilities. As the fraud detection model learns and identifies new patterns, the system could suggest optimal policy interventions to proactively discourage fraudulent behavior, creating a self-improving, adaptive fraud prevention ecosystem.
6.  **Continuous Learning and Robust MLOps Pipeline:** Implement a comprehensive MLOps (Machine Learning Operations) pipeline for continuous monitoring of model performance in production and automated retraining. Fraud patterns evolve rapidly, so the model needs to be regularly updated with new data and retrained to maintain its efficacy and adapt to emerging threats. This ensures the AI remains cutting-edge, relevant, and effective over time, avoiding model degradation.

## Setup and Usage

To run this project, you will need a Python environment with the following libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost

```

# Powerbi Dashboard Download link : https://drive.google.com/file/d/1U1XrR-Z3VXRT6FDLu0KzJWq7_dsEwBdF/view?usp=sharing

<img width="1134" height="620" alt="image" src="https://github.com/user-attachments/assets/b6ee020f-45e1-481f-b88d-d3747f8bf69b" />
<img width="1142" height="637" alt="image" src="https://github.com/user-attachments/assets/48a503f4-46d9-40f7-9cc6-d91329d3c6d7" />
<img width="1129" height="639" alt="image" src="https://github.com/user-attachments/assets/580d16cb-da15-4963-8116-f8de0c154164" />
<img width="1117" height="634" alt="image" src="https://github.com/user-attachments/assets/1b6d6b1e-afa1-47cb-9d5d-c83e7c7e98f5" />


# please click on the download button to download the total file donot download it part wise.
<img width="1647" height="1026" alt="image" src="https://github.com/user-attachments/assets/2e050ea2-6fc6-4071-a97f-e7d518befd44" />


# Our video link : https://drive.google.com/file/d/15PTMHILuw0dYzKGzifQ6JmmdEBfNPNwi/view?usp=sharing

<img width="1879" height="1031" alt="image" src="https://github.com/user-attachments/assets/18f173b4-cdd1-4830-9ce5-5f2967d14bb8" />




