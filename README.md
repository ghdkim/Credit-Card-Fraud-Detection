<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![LinkedIn][linkedin-shield]][linkedin-url]


<h3 align="center">Xccelerate Bootcamp</h3>
<h6 align="center">Unit 3 Machine Learning Project</h6>

<!-- ABOUT THE PROJECT -->
## About The Project
This project focuses on the detection of fraudulent transactions in credit card usage using machine learning techniques. The dataset contains transactions made by credit cards in September 2013 by European cardholders. The aim is to build a classification model that can accurately identify fraudulent transactions while minimizing false positives.


The dataset exhibits significant class imbalance; specifically, class = 1, representing fraud transactions, is notably rare. In this context, two metrics are crucial: model sensitivity (True Positives/Actual Positives) and precision (True Positives/Predicted Positives). We plan to implement the "Naive Bayes" method, a straightforward yet effective approach for predictive modeling and classification. A Naive Bayes model maintains a set of probabilities, including: 1. The probability of each class in the training dataset (class probability), and 2. The conditional probability of each input value given each class (conditional probability). An advantage of this model is its quick training time, as it only requires storing the aforementioned probabilities without the need for fitting coefficients through optimization processes.

Regarding Gaussian Naive Bayes, this variant is applied to real-valued attributes by assuming a Gaussian distribution. It is the simplest form to utilize, requiring only the calculation of the mean and standard deviation from the training data. For each class, we calculate the mean and standard deviation of the input values (X) to summarize the distribution.


### Built With
[![Python][Python-org]][Python-url]


<!-- Key Features -->
## Key Features
- **Outlier Detection and Handling:** Implementing methods to detect and handle outliers in the dataset.

- **Data Imputation:** Techniques to identify and impute missing data.

- **Feature Engineering:** Creating new columns that could enhance the model's predictive power.

- **Feature Selection:** Employing various feature selection techniques such as pandas profiling, feature importance analysis, correlation coefficients, and Variance Inflation Factor (VIF) model to identify the most relevant features for the classification task.

- **Model Development and Comparison:** Trying out different algorithms (like logistic regression, decision trees, random forests, etc.) and comparing their performances. This includes both single models and stacking approaches to see which combination yields the best results.

## Data Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Methodology

### Detecting and Handling Outliers
* Understanding the dataset such as the data shape, stat description etc.
* Understanding the imbalance dataset by visualising through a pie chart
* Understanding the relationship between the principal components (V1-V28) obtained with PCA (Principal Component Analysis)
  * Time (hrs) vs. Number of Transactions
  * Amount ($) vs. Number of Transactions
  * Understanding the correlation and shape of those principal components

### Feature Engineering
* Adding new columns to the DataFrame
  * Introducing a 'Time (Hr)' column will simplify comprehension and analysis, as understanding time in hours is much easier than dealing with time measured in seconds.
  * Introducing a 'Scaled Amount' as in machine learning algorithms, the scale of varaibles matters. Variables that are on a larger scale can unduly influence the model's output, leading to inaccurate results. In this dataset, 'Amount' has a wider range and larger values than other features. So, Standardization/Scaling sets mean to zero and standard deviation to one and makes the 'Amount' values comparable to other features. This can be particularly important if we are using a model that assumes normality.

### Model Development and Comparison
* Splitting the data into training and testing sets is crucial in the process of machine learning model building.
This process is needed so we can evaluate performance, avoid overfitting, model tuning and for a fair evaluation. In this case, we split the data into training set (70%) and test set (30%). 
* "predict()" method used to check whether a record should belong to "Genuine" or "Fraud" class. "predict_proba()" method used to present the probabilities for each class. These two methods are used to help us contorl precision and recall scores. 
* Creating a function to print the classifier's scores
  * confusion_matrix()
  * recall_score()
  * precision_score()
  * accuracy_score()
  * roc_auc_score()
* Training the Logistic-Regression Classification task in a different way (i.e. from under-sampled data). So, take only that percent of genuine-class cases which is equal to all fraud-classes (i.e. consider 50/50 ratio of both classes) 
* Comparing GaussianNB and Logistic-Regression

## Results (GaussianNB vs. Logistic Regression)
* Printed output:
  * Feature set
  * Dataset sizes:
    * Train set (70%) = 199364
    * Test set (30%) = 85443
    * Fraud cases in test set
  * Confusion Matrices: 
    * Train-set Confusion Matrix
      * Example array: `[a, b; c, d]`
        * This matrix shows how the model performed on the training set. Of the non-fraud cases (Class 0), `a`were correctly identified (True Negatives), and `b` were wrongly identified as fraud (False Positives). Of the fraud cases (Class 1), `c` were missed (False Negatives), and `d` were correctly identified (True Positives).
    * Test-set Confusion Matrix
      * Example array: `[a, b; c, d]`
        * Similar with the 'Train-set Confusion Matrix', this matrix shows how the model performed on the training set. Of the non-fraud cases (Class 0), `a`were correctly identified (True Negatives), and `b` were wrongly identified as fraud (False Positives). Of the fraud cases (Class 1), `c` were missed (False Negatives), and `d` were correctly identified (True Positives).
  * Recall score:  Indicates that the model correctly identifies `x%` of the fraud cases in the test set. If the recall score is high, it indicates that the model is very good at identifying positive instances. A high recall means that the model correctly classifies a large proportion of the actual positives in the dataset.
  * Precision score: Identifies `x%` of the transactions that the model predicts as fraud are actually fraud. If `x%` is low, it suggests that there is a high number of false positives.
  * F1 score:
    * Low F1 score:
      * Indicates an imbalance between precision and recall.
      * A low F1 score often arises in situations where either precision or recall is significantly lower than the other. For instance, a model might have high precision but very low recall (many false negatives), or high recall but low precision (many false positives).
      * In practical terms, a low F1 score suggests that the model is not effectively balancing the act of correctly identifying positive cases with the avoidance of misclassifying negative cases.
      * It's a sign that the model may not be performing adequately overall, especially in contexts where both false positives and false negatives are costly.
    * High F1 score: 
      * Indicates a good balance between precision and recall.
      * A high F1 score is achieved when both precision and recall are high, meaning the model is effectively identifying true positives while minimizing both false positives and false negatives.
      * In practical applications, a high F1 score suggests that the model is robust and reliable, making it suitable for tasks where it is important to maintain a balance between catching as many positive cases as possible without raising too many false alarms.
      * It's particularly valuable in scenarios where neither false positives nor false negatives can be heavily tolerated, such as in medical diagnostics or fraud detection.
  * Accuracy score: The model correctly predicts both fraud and non-fraud transactions `x%` of the time.
  * ROC AUC: The higher the `x%`, it presents that the model has a strong ability to differentiate between fraud and non-fraud transactions.

## Conclusion and Future Work
Comparing the NB and LR recall scores: NB = 0.83 (83%), LR = 0.91 (91%). 

This means that the model sensitivity is better through logistic-regression classification compared to GaussianNB. However, the precision score of NB (8%) is almost double from that of LR (4%) which indicates that NB classification model has a higher percentage of correct positive predictions. In other words, out of all the transactions that are predicted as Fraud, NB classification model has double the percentage where the model are indeed Fraud. 

**Limitations:**

- **Trade-off between Recall and Precision:**

  - Both models exhibit a trade-off between recall and precision. While LR has a higher recall, indicating better model sensitivity, NB has a higher precision, suggesting fewer false positives but more false negatives. Balancing these two metrics is a common challenge in classification tasks.


- **Model Complexity and Interpretability:**

    - Logistic Regression, generally providing more interpretable results than Naive Bayes, might still not offer sufficient insight into complex relationships within the data.


- **Assumptions and Data Characteristics:**

  - Naive Bayes assumes feature independence, which might not hold true in many real-world scenarios, potentially affecting its performance.
  Logistic Regression assumes a linear relationship between the independent variables and the log odds of the dependent variable, which may not always be appropriate.


- **Handling Imbalanced Data:**

  - Both models might struggle with highly imbalanced datasets, common in fraud detection, leading to biased predictions towards the majority class.


- **Generalization Capability:**

  - The models' ability to generalize to new, unseen data can be a concern, especially if the training data is not representative of the real-world scenario. Due to the confidentiality of the data source, the data could not provide the original features and more background information about the data. This can lead to assumptions of the data. 

**Future Work:**

- **Feature Engineering:**

  -    Improving the feature set through more sophisticated feature engineering could enhance model performance. This might include feature selection, transformation, or creation of interaction terms.


- **Hyperparameter Tuning:**

  -   Experimenting with different hyperparameters for each model could lead to better performance. For Naive Bayes, this might involve adjusting the default threshold for classification.


- **Exploring Other Models:**

  -   Testing other classification algorithms like Decision Trees, Random Forests, or Support Vector Machines could provide a better balance between recall and precision.


- **Cross-Validation and Robustness Checks:**

  -  Implementing cross-validation techniques can ensure the models are robust and generalize well to new data.


- **Handling Imbalanced Data:**

  -  Techniques like oversampling the minority class, undersampling the majority class, or using anomaly detection methods could help in dealing with imbalanced datasets.

## Dependencies
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

<!-- Installation -->
## Installation

To use the Job Scrapping Website, follow these steps:

   ```sh
   git clone [https://github.com/ghdkim/xccelerate_p3.git]
cd [xccelerate_p3]
python3 -m pip install -r requirements.txt
python3 main.py
   ```

<!-- CONTACT -->
## Contact

Gihyun Derek Kim -  ghdkimuni@gmail.com

Project Link: https://github.com/ghdkim/xccelerate_p3

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/gihyun-derek-kim/
[Python-org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/