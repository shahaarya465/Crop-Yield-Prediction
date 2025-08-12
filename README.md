
# Crop Yield Prediction

## 🚀 Live Demo
[Try the app on Render](https://crop-yield-prediction-jj38.onrender.com)

This project predicts crop yield using machine learning models trained on agricultural, environmental, and historical data. The workflow includes data exploration, preprocessing, feature engineering, model training, evaluation, and deployment as a Flask web app.

## Project Structure
- `crop_yield.ipynb`: Jupyter notebook with the full ML workflow (EDA, preprocessing, feature engineering, model training, export)
- `app.py`: Flask web app for making predictions using the trained model
- `templates/index.html`: HTML form for user input and prediction display
- `yield_df.csv`: Main dataset
- `rf_model.pkl`, `model_columns.pkl`: Exported model and feature columns for deployment
- `requirements.txt`: List of dependencies
- `README.md`: Project overview and instructions

## How to Use
### 1. Train and Export the Model
- Open `crop_yield.ipynb` in Jupyter or VS Code.
- Run all cells to train models and export the best model and feature columns (`rf_model.pkl`, `model_columns.pkl`).

### 2. Run the Web App
- Ensure all dependencies are installed (see below).
- In your terminal, run:
   ```
   python app.py
   ```
- Open your browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to use the web interface.

### 3. Make Predictions
- Enter the required features in the form (rainfall, pesticides, temperature, area, item, year) and click Predict.
- The predicted yield will be shown in kg/ha (kilograms per hectare).

## Requirements
Install dependencies using pip:
```
pip install -r requirements.txt
```

## Dependencies
The main dependencies are:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- flask
- scipy

## Notes
- The model predicts yield in kg/ha (kilograms per hectare).
- For best results, retrain the model with updated data as needed.
- For deployment, set `debug=False` in `app.py`.

## Notes
- The notebook is modular: each EDA, preprocessing, and model training step is in a separate cell for clarity.
- The Flask app uses the exported model and columns for prediction.
- You can further customize the workflow or add new features as needed.