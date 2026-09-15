# Real-Time Fraud Detection & Risk Analytics System

A machine learning-powered system designed to detect potentially fraudulent financial transactions, calculate transaction risk, and provide real-time fraud analysis through an interactive dashboard.

## 📌 About the Project

The **Real-Time Fraud Detection & Risk Analytics System** analyzes financial transaction data to identify suspicious transaction patterns and classify transactions according to their risk level.

The system considers multiple transaction characteristics, including transaction amount, transaction time, device type, location changes, failed attempts, transaction frequency, account age, previous transaction amount, and distance from the previous transaction.

A **Random Forest Classifier** is used to predict whether a transaction is potentially fraudulent. The trained model is integrated with a **Streamlit dashboard** that allows users to enter transaction details and receive an immediate fraud probability, risk score, and risk level.

## 🎯 Objectives

- Detect potentially fraudulent financial transactions
- Analyze transaction behavior and suspicious patterns
- Engineer meaningful fraud-related features
- Train and evaluate a machine learning classification model
- Generate real-time fraud predictions
- Calculate transaction risk scores
- Classify transactions into Low, Medium, and High risk
- Provide an interactive analytics dashboard

## 🚀 Key Features

- 📊 Transaction dataset generation
- 🧹 Data preprocessing and cleaning
- ⚙️ Automated feature engineering
- 🤖 Random Forest fraud classification
- 📈 Model evaluation
- 🔢 Fraud probability calculation
- 🎯 Dynamic risk scoring
- 🚦 Low, Medium, and High risk classification
- 🖥️ Interactive Streamlit dashboard
- 🌐 FastAPI prediction API
- 💾 Saved machine learning model
- 📓 Jupyter Notebook for transaction analysis

## 🔄 System Workflow

```text
Transaction Data
       ↓
Dataset Generation
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Random Forest Classifier
       ↓
Model Evaluation
       ↓
Trained ML Model
       ↓
Real-Time Prediction
       ↓
Fraud Probability
       ↓
Risk Score
       ↓
Risk Level
       ↓
Streamlit Dashboard
