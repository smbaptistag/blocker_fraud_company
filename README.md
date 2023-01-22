# BLOCKER FRAUD COMPANY

## 1. BUSINESS PROBLEM

**The Blocker Fraud Company**

The Blocker Fraud Company is a specialized company in fraud detection on financial transactions. It has the Blocker Fraud service, which ensures the block of fraudulent transactions.

The goal of the project is create a model with high precision and sensitivity with respect to transactions' fraud detection.

*This project is part of a study environment.*


## 2. THE DATASET

The dataset used on this project is a synthetic financial dataset generated in a simulator called PaySim and available on [kaggle](https://www.kaggle.com/ntnu-testimon/paysim1) [[1]](#references). The PaySim simulator uses aggregated data from private dataset to generate a synthetic dataset that resembles the normal operation of transactions and adds malicious behaviour to later evaluate the performance of fraud detection methods.


## 3. SOLUTION STRATEGY

The strategy adopted was the following:

**Step 01. Data Description:** Renaming columns, checking data types, searching for missing values and preliminary statistical description.

**Step 02. Feature Engineering:** Creating new features to improve the dataset for classification.

**Step 03. Data Filtering:** There was no consideration in this step.

**Step 04. Exploratory Data Analysis:** Analyzing the created features and checking the possibility of making a cleaner dataframe for classification. Understand existing data by creating a map of hypotheses. Study of the correlation between variables.

**Step 05. Data Preparation:** Rescaling the data for inputing to machine learning model. The encoding was used in the categorical features.

**Step 06. Feature Selection:** The Random Forest Classifier was used as feature selector to extract the most important features.

**Step 07. Machine Learning Modelling:** Training and predicting results with ML algorithms. Cross-validation were used to assess the best performance model. The best model was selected to be improved via hyperparameter fine tuning.

**Step 08. Hyperparameter Fine Tunning:** Identifying the best parameters taking into account ML model selected. Randomized Search CV was used to obtain the best parameters. Taking into account the time needed to run the model, only 10 iterations were considered with a cross-validation equal to 3. The metric considered was the F1 Score.


### 3.1. Key points

- The concentration of fraudulent amount transactions is under 1 million, with the highest percentage under 250 thousands.
- There is not fraud transaction above transaction amount of 10 million
- The amount of transfer and cash out fraudulent transactions are almost the same
- Only in 0.55% of the fraudulent transactions the origin balance difference before and after the transaction differs from the transaction amount
- In 98.05% of the fraudulent transactions the origin balance after the transaction is equal to zero
- Only in 65.15% of the fraudulent transactions the recipient balance after the transaction is equal to zero
- In almost all of the fraudulent transactions (99,81%) the recipient balance difference before and after the transaction differs from the transaction amount
- No fraudulent transactions that have merchant customer as recipient
- All the transactions flagged as fraud are indeed fraud
- The type of transactions are divided as follows:

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/transaction_type.jpg" width="600px">
</div>
</br>

## 4. MACHINE LEARNING MODELS
The following 5 machine learning algorithms were used to classify the transactions:

- Logistic Regression;
- Random Forest Classifier;
- Isolation Forest;
- XGBoost Classifier.
- K Nearest Neighbors (KNN)

Every model were cross-validated and its performance were compared.


## 05 MACHINE MODEL PERFORMANCE

The performance cross-validated for all 5 algorithms are displayed below:

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/models_performance.png" width="600px">
</div>
</br>

As it is an extremely unbalanced dataset, the accuracy metric can lead to a misinterpretation, since even a model that classify transactions randomly presents an accuracy close to 100% as ze can see in the baseline model.

The Random Forest Classifier model was chosen for the hyperparameters tuning, because it presents the best results, although the XGBoost Classifier presents similar results.

After tuning  hyperparameters using Randomized Search CV the model performance has improved (results for the same set of **training/validation data**):

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/rfc_performane_train.png" width="600px">
</div>
</br>

Confusion Matrix results:

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/cm_rfc_train.png" width="600px">
</div>
</br>

We can observe that the model continues to present good results for the test data and for the complete dataset (100% of the data):

**Test Dataset**

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/rfc_performane_test.png" width="600px">
</div>
</br>

Confusion Matrix results:

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/cm_rfc_test.png" width="600px">
</div>
</br>


**100 % Dataset**

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/rfc_performane_dataset.png" width="600px">
</div>
</br>

Confusion Matrix results:

<div align="center">
<img src="https://github.com/smbaptistag/blocker_fraud_company/blob/main/images/cm_rfc_dataset.png" width="600px">
</div>
</br>


## 6. Conclusion
This project was developed to meet the Blocker Fraud Company's goal of detecting fraudulent transactions. The solution was built on a synthetic financial dataset, which is an inherently unbalanced dataset. This characteristic required deep statistical and exploratory data analysis to understand fraudulent behavior and thereby extract insights and new features to inform model training and fine-tuning.

The project delivers a model with very good performance, as we can see in the results above.


## 7. NEXT STEPS

- Develop an app that intakes a portfolio of transactions and assigns for each transaction its respective probability of being fraudulent.
- Build a model retraining pipeline.


## References

[1] E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016
