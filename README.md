# Student Stress Monitoring using Multimodal ML/DL Framework

## Overview
This project implements a multimodal stress classification system using physiological, behavioral, and digital indicators from a 50-student dataset.

The system classifies stress levels into:
- Low
- Medium
- High

## Models Implemented
- Random Forest (Achieved 90.2% Accuracy)
- Support Vector Machine (SVM)
- Logistic Regression
- KNN
- CNN
- LSTM

## Project Structure
- preprocessing.py – Data cleaning and normalization
- model_training.py – ML and DL model implementation
- requirements.txt – Required dependencies

## Technologies Used
Python, NumPy, Pandas, Scikit-learn, TensorFlow/Keras

## Results
Random Forest achieved 90.2% accuracy with F1-score of 0.89.
Key predictors: HRV, Sleep Hours, Screen Time.
