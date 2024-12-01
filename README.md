# Housing Data Analysis: Unsupervised and Supervised Techniques

This project analyzes housing data using both unsupervised and supervised learning techniques to uncover patterns and make accurate predictions of house prices.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methods Used](#methods-used)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Supervised Learning](#supervised-learning)
- [Key Findings](#key-findings)

---

## **Project Overview**
The main objectives of this project were:
1. To explore housing patterns using **clustering** techniques.
2. To predict house prices using **machine learning models**.

By combining both approaches, we derive actionable insights for better decision-making in real estate analysis.

---

## **Dataset**
The dataset contains various features related to California housing, including:
- **Geographic Information**: Longitude, Latitude  
- **Demographics**: Population, Median Income, etc.  
- **Housing Features**: Total Rooms, Median House Value  

### **Preprocessing Steps**:
- Handling missing values in the `total_bedrooms` column.
- Normalizing numerical features.
- One-hot encoding categorical features.

---

## **Methods Used**

### **Unsupervised Learning**
- **Principal Component Analysis (PCA)**: Reduced dimensionality while preserving 83.6% of the variance.
- **KMeans Clustering**: Determined 3 distinct clusters of housing data using silhouette scores.

### **Supervised Learning**
- **Linear Regression**: Baseline model for interpretability.  
- **Random Forest**: Explored feature importance and achieved strong predictive performance.  
- **Neural Networks (TensorFlow)**: Tuned model to capture non-linear relationships.  
- **XGBoost**: Best-performing model with high R² and low error metrics.  

---

## **Key Findings**
- **Unsupervised Analysis**:
  - PCA identified regional and economic housing patterns.
  - Clustering highlighted distinct groups based on demographics and geography.
  
- **Supervised Analysis**:
  - **XGBoost** was the top model with R² = 0.839 and MAE = 30,661.
  - Features like `median_income` and `ocean_proximity` were consistently impactful.
