# Formual_1_pit_stop_calculation
I am creating a project based in old data to create a optimal timing for the pit stop in different scenario.
My objective is to create a program that will predict and analyse the optimal number and timing of pit stops in an F1 race, based on the race data, tire usage, track type and conditions
This is a personal project which i decided to work on during my summer break Currently the project has no deadline and no time limitations.
In this program i will be ignoring the relation between tyre wear and lap time to keep it easier in start
✅ Experiment Tracking

    import mlflow
    import mlflow.sklearn
  
  Used for tracking experiments, logging models, parameters, and metrics.

✅ Data Handling & Visualization

      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import seaborn as sns
  
  pandas, numpy: For data loading, manipulation, and numerical operations.
  matplotlib, seaborn: For visualization and plotting.

✅ Model Training & Preprocessing

      from sklearn.model_selection import train_test_split, GridSearchCV
      from sklearn.preprocessing import StandardScaler
      train_test_split: Split data into training and testing.
  
  GridSearchCV: For hyperparameter tuning.
  StandardScaler: Standardize features by removing the mean and scaling to unit variance.

✅ ML Models

      from sklearn.linear_model import LinearRegression
      from sklearn.tree import DecisionTreeRegressor
      from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
      from xgboost import XGBRegressor
      from catboost import CatBoostRegressor
  
  A wide variety of regression models:
  Linear, Tree-based (DecisionTree, RandomForest)
  Boosting (GradientBoosting, XGBoost, CatBoost)

✅ Model Evaluation
  
      from sklearn.metrics import mean_squared_error
  Used for evaluating regression models using MSE.

✅ Deep Learning

      import tensorflow as tf
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Dense, Dropout
      from tensorflow.keras.optimizers import Adam
  
  For building and training deep learning models using Keras API within TensorFlow.

✅ Web Deployment
  
      import streamlit as st
  
  Allows deployment of ML models as interactive web apps.

    💡 Suggestion
    We have to importing some libraries multiple times (e.g., train_test_split, StandardScaler). You can clean it up:
    
    
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
