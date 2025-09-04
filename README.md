# 🧠 Parkinson’s Disease Prediction using Machine Learning
Project Overview

This project predicts whether a person has **Parkinson’s Disease** based on biomedical voice measurements.  
The model is built using **Support Vector Machine (SVM)** and achieves high accuracy in classification.  


## 📊 Dataset
- **Filename:** `parkinsons_data.csv`  
- **Target variable:**  
  - `status = 1` → Parkinson’s Positive  
  - `status = 0` → Healthy  
- The dataset contains biomedical voice features such as fundamental frequency, jitter, shimmer, NHR, HNR, etc.    


## ⚙️ Tech Stack & Libraries
- **Language:** Python  
- **Libraries:**  
  - NumPy → Numerical computations  
  - Pandas → Data handling & preprocessing  
  - Scikit-learn → Model training, evaluation, scaling  
  - StandardScaler → Feature standardization  
  - SVM Classifier → Parkinson’s prediction  

---

## 🚀 Project Workflow
1. **Data Collection & Exploration**
   - Load dataset, check shape, info, missing values  
   - Statistical summary & distribution of target variable  
   - Group dataset by `status` for analysis  

2. **Data Preprocessing**
   - Drop irrelevant columns (`name`)  
   - Separate features (`X`) and target (`y`)  
   - Split dataset into **Training (80%)** and **Testing (20%)**  
   - Standardize features using `StandardScaler`  

3. **Model Training**
   - Train an **SVM Classifier with linear kernel**  

4. **Model Evaluation**
   - Accuracy on training set  
   - Accuracy on test set  

5. **Prediction System**
   - Take patient’s biomedical features as input  
   - Standardize input & predict disease status  
   - Example output:
     ```
     The Person has Parkinson’s
     ```

---

## 📈 Results
- Example accuracy:
  - **Training Accuracy:** ~89%  
  - **Test Accuracy:** ~87%  
- The model effectively distinguishes between healthy and Parkinson’s patients.  
