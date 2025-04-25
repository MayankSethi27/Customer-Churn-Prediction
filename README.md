# Churn Prediction Using Artificial Neural Networks (ANN)

## Project Overview

This project aims to predict customer churn based on various customer attributes. The project uses an Artificial Neural Network (ANN) to classify whether a customer will churn or not. The dataset used contains features such as age, account balance, and services used by the customers. 

### Models Used
- **Artificial Neural Network (ANN)**: A deep learning model used for binary classification tasks. In this project, the ANN model captures the complex relationships between the input features to predict whether a customer will churn.

### Evaluation Metrics:
- **Accuracy Score**: The percentage of correctly classified customers (both churned and non-churned).
- **Confusion Matrix**: Used to evaluate the classification performance and visualize the number of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score metrics for each class (churned and non-churned customers).

Both the confusion matrix and classification report were created to compare the model's performance in predicting customer churn.

### Loss Function & Optimizer
- **Loss Function**: The **binary cross-entropy** loss function is used, which is ideal for binary classification problems. It calculates the difference between the actual and predicted values and penalizes the model more when the prediction is far from the actual value.
- **Optimizer**: The **Adam optimizer** is used, which combines the advantages of two other popular optimizers: AdaGrad and RMSProp. Adam is known for its efficiency and suitability for large datasets and high-dimensional data. It adjusts the learning rate for each param

## Libraries Used

The following libraries were used in this project:

- **TensorFlow**: An open-source deep learning library used to build and train the Artificial Neural Network (ANN).
- **Keras**: A high-level neural networks API, built on top of TensorFlow, used to define and train the ANN model.
- **NumPy**: A library for numerical computing, used for handling arrays and performing mathematical operations.
- **Pandas**: A data analysis library, used for data manipulation and cleaning.
- **Matplotlib**: A plotting library used to visualize the model's training history and performance.
- **Scikit-learn**: A machine learning library, used for preprocessing, data splitting, and model evaluation (e.g., confusion matrix, classification report).

## Features
- **Data Preprocessing**: Includes encoding categorical features and feature scaling.
- **ANN Model Building**: The model consists of input layers, hidden layers, and an output layer.
- **Dropout Regularization**: Added in hidden layers to prevent overfitting.
- **Hyperparameter Tuning**: Early stopping was implemented during training to avoid overfitting.
- **Model Evaluation**: The model's performance was evaluated using confusion matrix and classification report.

## How to Run the Code

1. Clone the repository:

    ```bash
    git clone https://github.com/MayankSethi27/Churn-Prediction-ANN.git
    ```

2. Navigate into the project directory:

    ```bash
    cd Churn-Prediction-ANN
    ```

3. Install the required libraries (use a virtual environment if needed):

    ```bash
    pip install -r requirements.txt
    ```

4. Open and run the Jupyter Notebook `Churn_Prediction_ANN.ipynb` to see the entire workflow, from data preprocessing to model evaluation.

## Data Description

The dataset contains the following features:

- **Age**: Age of the customer
- **Balance**: Account balance of the customer
- **NumOfProducts**: Number of products used by the customer
- **HasCrCard**: Whether the customer has a credit card
- **IsActiveMember**: Whether the customer is an active member
- **EstimatedSalary**: Estimated salary of the customer
- **Exited**: Whether the customer churned (1) or not (0)

## Conclusion

The model built and evaluated in this project successfully predicts customer churn based on the available features. The model's performance was assessed using accuracy, confusion matrix, classification report, and ROC AUC curve, showing good prediction capability. The project demonstrates the potential of using Artificial Neural Networks (ANNs) for customer churn prediction, which can help businesses take proactive actions to retain valuable customers.
