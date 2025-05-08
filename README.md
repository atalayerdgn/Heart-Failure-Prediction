# Heart Failure Prediction

## Overview
This project develops a machine learning model to predict the likelihood of heart disease based on various health metrics. It uses an XGBoost classifier trained on a clinical dataset containing features like age, sex, chest pain type, blood pressure, cholesterol, and other relevant health indicators.

## Dataset

The dataset (`heart.csv`) contains 918 records with 12 features:
- Age: Patient's age
- Sex: Patient's gender (M/F)
- ChestPainType: Type of chest pain (ATA, NAP, ASY, TA)
- RestingBP: Resting blood pressure (mm Hg)
- Cholesterol: Serum cholesterol level (mg/dl)
- FastingBS: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- RestingECG: Resting electrocardiogram results
- MaxHR: Maximum heart rate achieved
- ExerciseAngina: Exercise-induced angina (Y/N)
- Oldpeak: ST depression induced by exercise relative to rest
- ST_Slope: Slope of the peak exercise ST segment
- HeartDisease: Output class (1 = heart disease; 0 = normal)

## Project Structure

- `Heart_Failure_Prediction_Model.ipynb`: Jupyter notebook containing the entire workflow
- `heart.csv`: Dataset with heart disease information
- `heart_failure_xgb_model.pkl`: Saved XGBoost model
- `scaler.pkl`: Saved StandardScaler for feature normalization

## Methodology

The project follows these steps:

1. **Data Loading**: Import and examine the heart disease dataset
2. **Exploratory Data Analysis**: Visualize distributions, correlations, and class balance
3. **Preprocessing**: Scale features and perform any necessary encoding
4. **Model Training**: Use XGBoost with hyperparameter tuning (GridSearchCV and StratifiedKFold)
5. **Evaluation**: Calculate classification metrics (accuracy, ROC AUC), confusion matrix
6. **Feature Importance**: Identify the most influential factors for prediction
7. **Model Saving**: Persist the trained model and scaler for future use

## Results

The model achieves approximately 89% accuracy and 94% ROC AUC score on the test data. The model identifies several key factors that contribute to heart disease prediction, shown in the feature importance analysis.

## How to Use

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
   ```
3. Run the notebook or use the saved model for predictions:
   ```python
   import joblib
   
   # Load the saved model and scaler
   model = joblib.load('heart_failure_xgb_model.pkl')
   scaler = joblib.load('scaler.pkl')
   
   # Preprocess new data
   X_new_scaled = scaler.transform(X_new)
   
   # Make predictions
   predictions = model.predict(X_new_scaled)
   probabilities = model.predict_proba(X_new_scaled)
   ```

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

## License

This project is available for open use and modification. 
