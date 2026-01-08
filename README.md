# AutoJudge-Predicting-Programming-Problem-Difficulty-Using-Textual_Features

# AutoJudge – Automated Programming Problem Difficulty Prediction

## 1. Project Overview
AutoJudge is an end-to-end Machine Learning and Natural Language Processing (NLP) based system designed to automatically estimate the difficulty of programming problems using only their textual descriptions. Competitive programming platforms such as Codeforces and CodeChef typically assign difficulty levels manually based on expert judgment and user feedback, which can be subjective, time-consuming, and inconsistent.

This project aims to automate that process by learning patterns directly from problem statements. The system performs two complementary tasks:
- **Difficulty Classification:** Categorizing problems into Easy, Medium, or Hard
- **Difficulty Score Prediction:** Estimating a continuous numerical difficulty score

The complete pipeline includes data preprocessing, feature extraction, model training, evaluation, and deployment via a Streamlit-based web interface.

---

## 2. Dataset Used
The dataset consists of programming problems collected from competitive programming sources and provided in JSON Lines (JSONL) format. Each record represents a single programming problem and contains both textual and label information.

### Dataset Fields
- **Title:** Short name or heading of the problem
- **Description:** Detailed problem statement
- **Input Description:** Input format and constraints
- **Output Description:** Output format and requirements
- **Problem Class:** Categorical difficulty label (Easy / Medium / Hard)
- **Problem Score:** Numerical difficulty rating

The dataset is pre-labeled and does not require manual annotation. During preprocessing, non-informative metadata such as URLs and source identifiers are removed.

---

## 3. Methodology

### 3.1 Data Preprocessing
The raw dataset undergoes multiple preprocessing steps to ensure consistency and suitability for machine learning:
- Removal of irrelevant columns
- Handling missing values in textual fields
- Standardization of difficulty class labels
- Text normalization (lowercasing, whitespace normalization, removal of unnecessary special characters)
- Combination of all textual fields into a single consolidated text input

This step ensures that each problem is represented by a clean and unified textual description.

---

### 3.2 Feature Extraction
To convert unstructured text into numerical features, a hybrid feature extraction strategy is employed:

#### TF-IDF Vectorization
- Unigrams, bigrams, and trigrams are extracted
- Stop words are removed
- Extremely rare and overly frequent terms are filtered
- Sublinear term frequency scaling is applied

TF-IDF captures the importance of terms that are indicative of problem complexity.

#### Handcrafted Numerical Features
To explicitly encode complexity-related information, additional features are added:
- Length of the problem text
- Count of mathematical and logical symbols
- Frequency of algorithm-related keywords (e.g., `dp`, `graph`, `dfs`, `greedy`, `recursion`)

The final feature matrix is formed by concatenating TF-IDF vectors with these numerical features.

---

## 4. Models Used

### 4.1 Classification Models
The following models were evaluated for difficulty class prediction:
- Logistic Regression
- Linear Support Vector Machine (SVM)
- **Random Forest Classifier (Best Performing)**

Random Forest achieved the highest classification accuracy due to its ability to model non-linear relationships in high-dimensional feature spaces.

---

### 4.2 Regression Models
To predict numerical difficulty scores, multiple regression models were trained:
- Linear Regression
- Random Forest Regressor
- **Gradient Boosting Regressor (Best Performing)**

Gradient Boosting demonstrated superior performance by effectively capturing complex interactions between textual features.

---

## 5. Evaluation Metrics

### Classification Evaluation
- Accuracy
- Confusion Matrix

### Regression Evaluation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- r2_score

Random Forest achieved the best classification accuracy, while Gradient Boosting Regressor obtained the lowest MAE and RMSE for difficulty score prediction.

---

## 6. Web Interface
A lightweight web application was developed using **Streamlit** to demonstrate real-time inference.

### Interface Functionality
- Text input fields for:
  - Problem Description
  - Input Description
  - Output Description
- Button to trigger prediction
- Display of:
  - Predicted difficulty class
  - Predicted numerical difficulty score

The web interface uses the same preprocessing and feature extraction pipeline as model training, ensuring consistency between training and deployment.

---

## 7. Running the Project Locally

### Step 1: Clone the Repository
```bash
git clone <repository-link>
cd AutoJudge
pip install streamlit scikit-learn scipy numpy joblib
AutoJudge/
│
├── app.py
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── difficulty_classifier.pkl
│   └── difficulty_score_regressor.pkl

python -m streamlit run app.py
http://localhost:8501

```

## 8. Demo Video

A short demo video demonstrating the working of the AutoJudge web application is available at the link below:

 **Demo Video:**  
<PASTE YOUR GOOGLE DRIVE / YOUTUBE LINK HERE>

The video demonstrates:
- Launching the Streamlit application
- Providing sample problem inputs
- Viewing the predicted difficulty class and difficulty score

## 9. Author Details

- **Name:** Disha Agarwal  
- **Institute:** Indian Institute of Technology Roorkee  
- **Department:** Chemical Engineering  
- **Enrollment:** 23112034 
-
