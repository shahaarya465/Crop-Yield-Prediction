
from flask import Flask, render_template, request

import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Define the main numeric and categorical features as per the form
main_features = [
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp'
]
categorical_features_names = ['Area', 'Item', 'Year']


# Load unique options for dropdowns from the CSV
df = pd.read_csv('yield_df.csv')
categorical_features = {
    'Area': sorted(df['Area'].dropna().unique()),
    'Item': sorted(df['Item'].dropna().unique()),
    'Year': sorted(df['Year'].dropna().unique())
}

# Load model and columns
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    # Default values for form fields
    main_inputs = {feat: '' for feat in main_features}
    selected = {cat: '' for cat in categorical_features_names}


    if request.method == 'POST':
        # Get numeric inputs
        for feat in main_features:
            main_inputs[feat] = request.form.get(feat, '')
        # Get categorical selections
        for cat in categorical_features_names:
            selected[cat] = request.form.get(cat, '')

        # Prepare input for model
        input_dict = {**main_inputs, **selected}
        # Convert numeric fields
        for feat in main_features:
            try:
                input_dict[feat] = float(input_dict[feat])
            except:
                input_dict[feat] = 0.0

        # Create input DataFrame with all model columns
        input_df = pd.DataFrame([input_dict])
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        # Predict
        try:
            pred = model.predict(input_df)[0]
            prediction = f"{pred:.2f} kg/ha (kilograms per hectare)"
        except Exception as e:
            prediction = f"Prediction error: {e}"

    return render_template(
        'index.html',
        main_features=main_features,
        categorical_features=categorical_features,
        main_inputs=main_inputs,
        selected=selected,
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)
