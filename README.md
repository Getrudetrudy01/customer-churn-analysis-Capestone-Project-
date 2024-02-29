# Customer Churn Prediction Capstone Project

This repository contains the code and dataset for the Customer Churn Prediction Capstone Project. The goal of this project is to develop a machine learning model that can predict customer churn in a telecom company.

## Project Overview

Customer churn refers to the phenomenon where customers discontinue using a company's products or services. It is a significant challenge for businesses, as losing customers can have a negative impact on revenue and growth. The Customer Churn Prediction Capstone Project aims to address this issue by building a predictive model that can identify customers who are likely to churn.

The project involves the following steps:

1. **Data Collection**: The `telecom_churn.csv` dataset is used for training and evaluating the churn prediction model. This dataset contains various features related to customer behavior, usage patterns, and demographics.

2. **Data Preprocessing**: The data is cleaned and preprocessed to handle missing values, encode categorical variables, and normalize numerical features. This ensures that the data is in a suitable format for training the machine learning model.

3. **Feature Selection/Engineering**: Relevant features are selected or engineered to improve the model's performance. This step involves analyzing the dataset, identifying informative features, and creating new features if necessary.

4. **Model Training**: Machine learning algorithms such as logistic regression, decision trees, or random forests are used to train a churn prediction model on the labeled dataset. The model learns patterns and relationships between the features and customer churn.

5. **Model Evaluation**: The performance of the trained model is assessed using appropriate evaluation metrics such as accuracy, precision, recall, and F1 score. This step helps determine how well the model predicts customer churn.

6. **Prediction**: The trained model can be utilized to make predictions on new, unseen data. This allows businesses to identify customers who are at a higher risk of churn and take proactive measures to retain them.

## Repository Files

The repository includes the following files:

1. `customer_churn_prediction.ipynb`: Jupyter Notebook containing the implementation of the customer churn prediction model. This notebook provides step-by-step instructions and explanations of the data preprocessing, feature selection/engineering, model training, model evaluation, and churn prediction.

2. `telecom_churn.csv`: CSV file containing the customer churn dataset used for training and evaluating the churn prediction model. The file includes information about customer behavior, usage patterns, demographics, and churn status.

## How to Use

To use this project and train a customer churn prediction model, follow these steps:

1. Clone the repository to your local machine using the following command:
   ```
   git clone https://github.com/getrudetrudy01/customer-churn-prediction.git
   ```

2. Open the `customer_churn_prediction.ipynb` Jupyter Notebook using Jupyter Notebook or JupyterLab.

3. Follow the instructions in the notebook to execute the code cells and train the churn prediction model.

4. If you want to use the trained model for prediction on new data, make sure to have the necessary input data in a similar format as the `telecom_churn.csv` file. You can load the trained model from the notebook and apply it to the new data to predict customer churn.

## Conclusion

The Customer Churn Prediction Capstone Project demonstrates the application of machine learning techniques to predict customer churn in a telecom company. By training a model on a labeled dataset, the project aims to develop a churn prediction model that can help businesses identify customers who are at a higher risk of churn. The Jupyter Notebook provided allows for an interactive and flexible environment to explore, develop, and enhance the churn prediction process.

## Contributing

Pull requests and contributions to enhance the project are welcome. Feel free to submit suggestions, bug reports, or improvements.

