 
## Rock vs Mine Predictor

This is a simple logistic regression model that classifies sonar signals as either **rock** or **mine** using the [Sonar dataset from Kaggle](https://www.kaggle.com/datasets/rupakroy/sonarcsv).

### Model Used

* **Logistic Regression** from `scikit-learn`

### Accuracy

* Achieved an accuracy of **83%** on the test set.

###  Files

* `rock_vs_mine.ipynb`: Google Colab notebook with the full code.
* `requirements.txt`: List of required libraries.


### What the code does

Loads the CSV dataset using pandas
Splits features (X) and target (y)
Splits data into training and testing sets
Trains a logistic regression model using scikit-learn
Evaluates model performance using accuracy score

### 🛠 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### 📦 Dependencies

* pandas
* numpy
* scikit-learn

