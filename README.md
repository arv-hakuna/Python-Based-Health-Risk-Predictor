# Health Risk Prediction Model

This project aims to predict the health risk of patients based on various health-related attributes. The prediction is made using a Logistic Regression model.

## Dataset

The model is trained on the `processed_health_data.csv` dataset. This dataset contains various health metrics for patients, including:

- Age, Gender, Ethnicity
- Clinical measurements like blood pressure, heart rate, BMI, etc.
- Lifestyle information like diet, exercise, smoking, and alcohol consumption.
- Lab results like cholesterol levels, blood glucose, etc.
- The target variable is `Health_Risk`, which is a categorical representation of the patient's health risk.

## Model

A Logistic Regression model is used for the classification task. The model is trained to predict the `Health_Risk` category based on the other features in the dataset.

## Getting Started

### Prerequisites

- Python 3.x
- The libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <your-project-directory>
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run the prediction model, execute the following command:

```bash
python health_risk_predictor.py
```

This will train the model, make predictions on the test set, and print the model's accuracy and a classification report.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.