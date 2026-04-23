# Multiple Disease Prediction System

An end-to-end machine learning system for predicting the likelihood of:
- Diabetes
- Heart Disease
- Chronic Kidney Disease

This project integrates data preprocessing, model training, evaluation, and deployment through an interactive Streamlit dashboard.

---

## 📌 Project Overview

This system applies supervised machine learning techniques to three healthcare datasets to predict disease risk. Multiple models were evaluated, including Logistic Regression, Random Forest, Support Vector Machine (SVM), and Gradient Boosting.

Model selection was based on a combined evaluation metric that prioritises both **F1-score and Recall**, ensuring that false negatives (missed diagnoses) are minimised — a critical requirement in medical screening applications.

Gradient Boosting (default configuration) was selected as the most consistent and reliable model across all datasets.

---

## ⚙️ Features

- Multi-disease prediction (Diabetes, Heart Disease, Kidney Disease)
- Data preprocessing (missing value handling, encoding, scaling)
- Model comparison across multiple algorithms
- Performance evaluation (Accuracy, Precision, Recall, F1-score, AUC)
- Interactive Streamlit dashboard
- Real-time prediction with probability output
- Visualisations:
  - ROC curves
  - Confusion matrices
  - Feature importance

---

## 📁 Project Structure

.
├── app.py  
├── Final_Master_Project.ipynb  
├── requirements.txt  

├── Dataset/  
│   ├── diabetes.csv  
│   ├── heart_disease_uci.csv  
│   ├── kidney_disease.csv  

├── models/  
│   ├── diabetes_model.pkl  
│   ├── heart_model.pkl  
│   ├── kidney_model.pkl  

---

## 🚀 How to Run the Project

### 1. Clone the repository

git clone https://github.com/anispaudel/multiple-disease-prediction-system.git  
cd multiple-disease-prediction-system  

#2. Install dependencies

pip install -r requirements.txt  

 3. Run the Streamlit application

streamlit run app.py  

---

 🧠 Model Selection Strategy

Model selection was based on a combined metric:

F1_Recall_Score = (F1 + Recall) / 2

This approach ensures a balance between precision and recall, while prioritising recall to reduce the risk of false negatives in medical diagnosis scenarios.

---

## ⚙️ Deployment (Streamlit Dashboard)

A web-based interactive dashboard was developed using Streamlit to demonstrate real-time prediction capabilities.

- Pre-trained Gradient Boosting models are loaded from `.pkl` files  
- No retraining occurs during runtime  
- Input data is processed using the same preprocessing pipeline as training  
- Users can input patient data and receive:
  - Predicted class  
  - Probability score  
  - Risk indication  

The dashboard includes:
- Overview (performance summary)  
- Data Explorer (dataset insights)  
- Model Performance (metrics and visualisations)  
- Live Prediction (real-time inference)  




## ⚠️ Disclaimer

This project is developed for **research and educational purposes only**.  
It is **not intended for clinical diagnosis or real-world medical decision-making**.

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Joblib  

---

## 🧠 Key Insight

Model selection prioritised Recall alongside F1-score to minimise false negatives, which is critical in healthcare applications where missed diagnoses can have severe consequences.

---

## 🚀 Future Improvements

- Integration with real-time clinical datasets  
- Deployment to cloud platforms (Streamlit Cloud / AWS)  
- Advanced models (XGBoost, Deep Learning)  
- Improved UI/UX for clinical usability  

---

 👤 Author

Anis Paudel  

