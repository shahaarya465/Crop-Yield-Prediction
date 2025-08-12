# 🌾 Crop Yield Prediction Web App

## 📖 Project Overview

This project is a **full-stack machine learning solution** for predicting crop yield based on **agricultural, environmental, and historical data**.

It combines **data science best practices** with a **user-friendly Flask web interface**, allowing users to input relevant features and receive **real-time yield predictions**.

---

## 🛠️ Features

- 🌐 **Interactive Web Form** – Input rainfall, pesticides, temperature, area, item, and year.
- 🤖 **Real-Time Predictions** – Powered by a trained **Random Forest** model.
- 🎨 **Modern UI** – Clean HTML/CSS design.
- 📊 **Complete ML Workflow** – EDA, preprocessing, feature engineering, and model selection.
- 📦 **Handles Large Models** – Git LFS support for deployment.
- 🔄 **Easily Extensible** – Add new features or retrain with updated data.

---

## 📂 Project Structure

```
├── crop_yield.ipynb        # Jupyter notebook: EDA, preprocessing, model training
├── app.py                  # Flask web app
├── templates/
│   └── index.html          # Web form UI
├── yield_df.csv            # Main dataset
├── rf_model.pkl            # Trained Random Forest model
├── model_columns.pkl       # Model feature columns
├── requirements.txt        # Dependencies
├── Procfile                # For Render deployment
└── README.md               # Project documentation
```

---

## ⚡ Quickstart

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shahaarya465/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train and Export the Model
- Open `crop_yield.ipynb` in Jupyter Notebook or VS Code and run all cells to:
  - Train multiple models (Random Forest, XGBoost, Linear Regression)
  - Export the best model as `rf_model.pkl`
  - Save model feature columns in `model_columns.pkl`

### 4️⃣ Run the Web App Locally
```bash
python app.py
```

### 5️⃣ Make Predictions
Enter the required values and click Predict to get crop yield in **kg/ha**.

---

## 🌐 Deployment on Render

This app is live at: [Crop Yield Prediction - Render](https://crop-yield-prediction-ji38.onrender.com)

To deploy your own version:
- Fork this repository
- Connect it to Render
- Ensure `Procfile` and `requirements.txt` are included
- Enable Git LFS for large model files

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
flask
scipy
gunicorn
jinja2
```

---

## 📝 Notes
- Predictions are given in **kg/ha** (kilograms per hectare)
- For best results, retrain the model with updated datasets
- Set `debug=False` in `app.py` for production
- The notebook is modular – each step (EDA, preprocessing, model training) is in a separate cell for clarity

---

## 🙏 Acknowledgements
- Dataset: [Kaggle - Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/mrigaankjaswal/crop-yield-prediction-dataset)
- Hosting: Render for free deployment
- Libraries: scikit-learn, XGBoost, Flask, and the open-source community

---

## 📌 Author
**Aarya Shah**  
🔗 [GitHub](https://github.com/shahaarya465)
