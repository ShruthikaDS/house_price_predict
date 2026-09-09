import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='../templates')

# Load the model directly from the linear.pkl file located in the root directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'linear.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input features from the form based on dataset training features
            square_footage = float(request.form.get('Square_Footage', 0))
            num_bedrooms = float(request.form.get('Num_Bedrooms', 0))
            num_bathrooms = float(request.form.get('Num_Bathrooms', 0))
            year_built = float(request.form.get('Year_Built', 0))
            lot_size = float(request.form.get('Lot_Size', 0))
            garage_size = float(request.form.get('Garage_Size', 0))
            neighborhood_quality = float(request.form.get('Neighborhood_Quality', 0))

            # Match feature names used during training: 
            # ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
            input_df = pd.DataFrame([{
                'Square_Footage': square_footage,
                'Num_Bedrooms': num_bedrooms,
                'Num_Bathrooms': num_bathrooms,
                'Year_Built': year_built,
                'Lot_Size': lot_size,
                'Garage_Size': garage_size,
                'Neighborhood_Quality': neighborhood_quality
            }])

            # Make prediction
            pred = model.predict(input_df)
            
            # Format output value
            if isinstance(pred, np.ndarray):
                prediction = f"${float(pred.flatten()[0]):,.2f}"
            else:
                prediction = f"${float(pred):,.2f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
