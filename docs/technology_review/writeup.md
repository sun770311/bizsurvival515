# Technology Review: NYC Business Survival

## 1. Background and Use Case

### Our Application
Our application is a business survival prediction model designed to estimate whether a business is at risk of becoming inactive using structured signals derived from licensing records and 311 service request data. The primary goal is proactively detect risks, where higher-risk businesses can be identified in advanced based on observable behavioral and environmental patterns.

### Use Case
One important use of the model is helping useres classify businesses as either likely to remain or likely to become inactive. Combining category information, engineered complaint-risk signals, and historical inactivity trends, the system determines whether a business exhibits characteristics that are statistically associated with inactive outcomes. 

The model also supports comparative analysis across business groups. By learning which categories and 311 complaint types are historically associated with inactivity, the system can highlight broader structural patterns. Users can understand how survival outcomes vary across sectors and environments.

The technology allows scenario-based classification of hypothetical businesses. Given a proposed business category and location context, the model assigns a predicted status label based on similar historical cases. 

### Why a Python Library is Needed
The raw licensing and 311 datasets contain signals that are individually suggestive. From exploratory data analysis, some business categories and specific 311 service types appear more associated with inactive outcomes, but those patterns are not reliable enough to use directly. To make the system usable, we need a method that can combine many features at once, learn their relative importance from data, and produce a consistent binary classification of Active vs. Inactive. 

A Python machine learning library/package is needed to implement this end-to-end modeling workflow in a way that is reproducible and statistically grounded. Specifically, the library needs to support:

- **Preprocessing at scale**: transforming mixed data types into model-ready inputs such as one-hot encoding categorical variables like business category.

- **Model training for binary classification**: fitting a classifier that can jointly use category information, engineered features, and numeric predictors such as latitude and longitude to learn patterns associated with inactivity.

- **Reliable evaluation**: computing precision, recall, F1-score, and running repeated train/test splits so results are not dependent on one random partition.

- **A consistent pipeline**: packaging preprocessing and modeling into a single workflow that can be rerun and extended.

## 2. Python Package Choices

### 1. scikit-learn (Logistic Regression)

**Name**: [scikit-learn](https://github.com/scikit-learn/scikit-learn) (sklearn)

**Author**: Originally developed by David Cournapeau in 2007 as a Google Summer of Code project; now maintained by a team of volunteers.

**Purpose**: scikit-learn is a Python machine learning library that offers tools for data preprocessing, supervised and unsupervised learning (incluing classification and regression), model selection, and performance evaluation. It is built for efficiency and ease of use, and integrates well with the core scientific Python stack such as NumPy and pandas.

### 2. XGBoost 

**Name**: [XGBoost](https://github.com/dmlc/xgboost) (xgboost)

**Author**: Developed by Tianqi Chen and Carlos Guestrin working on a research project at University of Washington; currently maintained by the XGBoost open-source community with sponsors and contributors from companies such as Intel, NVIDIA, and others.

**Purpose**: XGBoost is a high-performance gradient boosting framework optimized by speed and scalability. A gradient boosting framework is a machine learning approach that build a strong predictive model by combining many weak models (usually decision trees) in sequence. Each new model is trained to correct the errors made by the previous ones, using gradient-based optimization to minimize a loss function, which allows the overall system to improve accuracy and handle complex nonlinear patterns. XGBoost is effective for predictive modeling on structured or tabular datasets.

## 3. Package Comparison

### `from sklearn.linear_model import LogisticRegression`

Logistic Regression serves as a strong and interpretable baseline model for this project's binary classification task (Active vs. Inactive). It supports probability estimation, regularization, and standard evaluation metrics, making it suitable for structured tabular prediction where relationships are relatively simple. Computationally, Logistic Regression is efficient for small-to-medium tabular datasets such as the ones in our project. The implementation in scikit-learn is highly stable and widely used in both academic and production settings, with extensive documentation and a long maintenance history.

**Open GitHub Issues**: Recent development activity around `LogisticRegression` primarily involves minor bug fixes (e.g., cross-validation folds, warnings, and edge cases), along with ongoing API modernization, documentation updates, and general maintenance. The Logistic Regression module is mature, stable, and actively maintained. Across the broader scikit-learn project, there are approximately 1,603 open issues and 10,364 closed issues, reflecting an active development community, and Logistic Regression remains one of the library’s longest-standing components.

### `from xgboost import XGBClassifier`

While its underlying learning algorithm and implementation come from the XGBoost library itself, XGBoost does provide the scikit-learn compatible `XGBClassifier` that is easily integrated into an sklearn workflow. XGBoost has more hyperparameters and requires tuning, but reasonable defaults often perform well. XGBoost is also mature and widely adopted in industry and academia. 

**Open GitHub Issues:** The repository currently shows 407 open and 5,214 closed issues, indicating active ongoing maintenance. Most open issues do not impact typical users performing standard CPU-based training through the scikit-learn API on tabular data; instead, they mainly relate to infrastructure, GPU support, or distributed computing. Some of these concerns are only relevant for users building XGBoost from source or contributing to the project.

### Comparison Table

| Aspect                       | Logistic Regression                                      | XGBoost                                                   |
| ---------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| **Model Type**               | Linear probabilistic classifier                          | Gradient boosted decision tree ensemble                   |
| **Project Role**             | Strong, interpretable baseline for binary classification | Captures nonlinear relationships and feature interactions |
| **Complexity Handling**      | Limited to mostly linear relationships                   | Models complex nonlinear patterns effectively             |
| **Ease of Use**              | Very easy to train, minimal tuning required              | Moderate; requires hyperparameter tuning                  |
| **Computational Efficiency** | Fast for small–medium tabular data                       | Highly optimized; efficient and scalable                  |
| **Stability & Maintenance**  | Mature, long-standing sklearn component                  | Mature, widely used, actively maintained                  |

## 4. Our Choice
To select the most suitable modeling library, we evaluated Logistic Regression and XGBoost using five different random seeds. The dataset consists of 311 complaints from 2025 that were spatially linked to businesses based on geographic coordinates. For each run, we measured accuracy, F1 score, precision, recall, ROC-AUC, and PR-AUC. Precision measures the fraction of predicted active/inactive businesses that are truly active/inactive. Recall measures the fraction of actual active/inactive businesses correctly identified. The F1 score is the harmonic mean of precision and recall. ROC-AUC measures how well the model can separate inactive from active businesses overall, across all possible decision thresholds. PR-AUC focuses on how accurately the model identifies inactive businesses while avoiding false alarms, which is especially useful when evaluating classification performance. Across all 5 runs, Logistic Regression consistently outperformed XGBoost. 

### Model Parameters

| Logistic Regression (scikit-learn)  | XGBoost                                       |
| ----------------------------------- | --------------------------------------------- |
| **Regularization:** L2 with C = 0.1 | **Number of trees:** 600                      |
| **Feature scaling:** StandardScaler | **Learning rate:** 0.05                       |
| **Optimizer:** lbfgs                | **Tree depth:** Moderate depth                |
| **Max iterations:** 2000            | **Subsampling:** Row and column subsampling   |
| —                                   | **Regularization:** reg_lambda = 1.0          |
| —                                   | **Objective:** Binary logistic classification |
| —                                   | **Tree method:** Histogram-based training     |

---

### Individual Run Results

| Seed | Model   | Accuracy | F1     | Precision | Recall | ROC-AUC | PR-AUC |
| ---- | ------- | -------- | ------ | --------- | ------ | ------- | ------ |
| 0    | LogReg  | 0.6569   | 0.6957 | 0.6248    | 0.7848 | 0.7309  | 0.7212 |
| 0    | XGBoost | 0.6344   | 0.6631 | 0.6147    | 0.7198 | 0.7057  | 0.6917 |
| 1    | LogReg  | 0.6432   | 0.6648 | 0.6270    | 0.7075 | 0.7135  | 0.7130 |
| 1    | XGBoost | 0.6290   | 0.6585 | 0.6102    | 0.7150 | 0.7022  | 0.6965 |
| 2    | LogReg  | 0.6444   | 0.6768 | 0.6201    | 0.7448 | 0.7139  | 0.7021 |
| 2    | XGBoost | 0.6361   | 0.6667 | 0.6148    | 0.7281 | 0.7035  | 0.6885 |
| 3    | LogReg  | 0.6369   | 0.6677 | 0.6153    | 0.7298 | 0.7084  | 0.6930 |
| 3    | XGBoost | 0.6173   | 0.6436 | 0.6020    | 0.6914 | 0.6909  | 0.6822 |
| 4    | LogReg  | 0.6411   | 0.6659 | 0.6226    | 0.7156 | 0.7094  | 0.6950 |
| 4    | XGBoost | 0.6282   | 0.6616 | 0.6068    | 0.7273 | 0.7014  | 0.6861 |

---

### Average Across 5 Runs

| Model   | Accuracy        | F1              | Precision       | Recall          | ROC-AUC         | PR-AUC          |
| ------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| LogReg  | 0.6445 ± 0.0075 | 0.6742 ± 0.0129 | 0.6220 ± 0.0045 | 0.7365 ± 0.0305 | 0.7152 ± 0.0091 | 0.7049 ± 0.0120 |
| XGBoost | 0.6290 ± 0.0074 | 0.6587 ± 0.0089 | 0.6097 ± 0.0054 | 0.7163 ± 0.0149 | 0.7007 ± 0.0057 | 0.6890 ± 0.0054 |

### Final Model Choice: Logistic Regression
Based on empirical performance and practical considerations, we select Logistic Regression (scikit-learn) as the final model. While XGBoost is capable of capturing more complex nonlinear interactions, Logistic Regression performs slightly better on this dataset while remaining simpler, faster to train, and easier to interpret. The predictive signals uncovered during exploratory analysis, particularly category risk and 311-type risk, exhibit relatively smooth relationships, meaning a linear decision boundary with well-engineered features is sufficient for reliable classification.

## 5. Drawbacks and Remaining Concerns
Despite its advantages, Logistic Regression has several limitations. Because it is a linear model, it cannot naturally capture complex non-linear interactions between features, which may limit predictive performance compared to more flexible models such as XGBoost. 

The model also depends heavily on the quality of feature engineering; if important signals are not encoded properly, Logistic Regression cannot discover them automatically. 

Future work may involve parameter tuning, exploring non-linear extensions, improving feature representations, and validating model calibration to ensure predicted probabilities remain reliable in practical deployment.