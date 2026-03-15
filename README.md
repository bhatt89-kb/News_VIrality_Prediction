# 📰 News Virality Prediction

This project predicts whether a **news article will go viral** using Machine Learning.  
The model analyzes different features of an article such as title length, number of images, videos, links, and sentiment to estimate its popularity.

## 🚀 Project Overview
With the growth of online media, predicting which articles will gain high engagement is valuable for publishers and content creators.  
This project uses the **Online News Popularity dataset** to train a machine learning model that classifies articles as **viral or non-viral**.

A simple **Streamlit web application** allows users to input article features and instantly get predictions.

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Streamlit

## 📂 Project Structure

```
news-virality-prediction
│
├── app.py                     # Streamlit web application
├── model.pkl                  # Trained ML model
├── feature_cols.pkl           # Feature columns used by the model
├── notebook.ipynb             # Model training and analysis
├── OnlineNewsPopularity.csv   # Dataset
│
├── confusion_matrix.png
├── roc_curve.png
├── correlation_heatmap.png
├── shap_importance.png
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/news-virality-prediction.git
```

Move into the project folder:

```bash
cd news-virality-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the **local server link** shown in the terminal.

## 📊 Model Evaluation
The model performance is evaluated using:

- Confusion Matrix
- ROC Curve
- Feature Importance
- SHAP analysis

## 📌 Applications
- Media analytics
- Content strategy optimization
- Digital marketing insights
- News publishing platforms

## 👨‍💻 Author
Kapil Bhatt
