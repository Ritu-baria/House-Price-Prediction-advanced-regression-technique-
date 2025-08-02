# 🏡 House Price Prediction – Data Science Project

Predicting real estate prices based on various features such as location, area, floor, furnishing, and more using machine learning.

## 📌 Problem Statement

The objective of this project is to build a machine learning model that can accurately predict the price of residential properties using structured data that includes area measurements, property status, furnishing type, facing direction, transaction type, and more.


## 📊 Dataset Overview

The dataset contains the following columns:

- **Location** – Geographic location of the property  
- **Carpet Area / Super Area / Plot Area** – Different measurements of property area  
- **Furnishing** – Furnishing status (Fully, Semi, Unfurnished)  
- **Transaction** – New property or resale  
- **Floor / Status / Facing / Overlooking** – Physical and directional attributes  
- **Bathroom / Balcony / Car Parking** – Amenities  
- **Ownership** – Freehold, Leasehold, etc.  
- **Price (in rupees)** – Target variable  


## ⚙️ Tech Stack

- Python 🐍
- Pandas & NumPy for data manipulation  
- Seaborn & Matplotlib for data visualization  
- Scikit-learn for ML modeling  
- Random Forest Regressor as baseline model


## 🚀 Project Workflow

1. **Data Cleaning**  
   - Remove non-predictive columns (e.g., title, description)  
   - Handle missing values  
   - Convert area and numerical fields properly

2. **Feature Engineering**  
   - Label encode categorical variables  
   - Extract area numbers from strings (e.g., “1,200 sqft” → 1200)

3. **Modeling**  
   - Split into training and testing sets  
   - Trained Random Forest Regressor  
   - Evaluated using RMSE and R² score  

4. **Evaluation**  
   - R² Score: `~0.85+` (can vary depending on dataset)
   - RMSE: Reasonable error range for real-estate

5. **Results**  
   - Predicted prices closely matched actual prices  
   - Feature importance highlighted which variables most influenced price

## 📂 Project Structure

# House-Price-Prediction-advanced-regression-technique-
