from flask import Flask, request, render_template
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
model = pickle.load(open('YieldPrice.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

def get_rainfall(district_name):
    # The URL from which to scrape the data
    url = "https://www.tsdps.telangana.gov.in/districtdata.jsp"
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all <tr> tags to get each row of the table
        rows = soup.find_all('tr')
        
        # Iterate over each row
        for row in rows:
            # Find all <td> tags for each row - each td is a cell in the row
            cells = row.find_all('td')
            if len(cells) > 1:  # If there are cells in the row
                # Check if the district name matches the one we're looking for
                if cells[1].text.strip().lower() == district_name.lower():
                    # Assuming the sixth cell is the actual cumulative rainfall
                    actual_cumulative_rainfall = cells[5].text.strip()
                    return actual_cumulative_rainfall
    
    # Return a default value if the district is not found or if there's an error
    return 0


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    crop = form_data['crop'].lower()  # Ensure the crop name is in lowercase for comparison
    district = form_data['district']
    area = float(form_data['area'])  # Assuming 'area' is passed from the form in acres
    rainfall = get_rainfall(district)
    max_temp = float(form_data['matemp'])
    min_temp = float(form_data['mitemp'])
    
    input_data = pd.DataFrame({
        'Crop': [crop],
        'Total Rainfall': [rainfall],
        'Max. Temp': [max_temp],
        'Min Temp': [min_temp],
        'District': [district]
    })
    
    # Prediction is assumed to be yield in kg per acre
    yield_per_acre = model.predict(input_data)[0]
    yield1 = round(yield_per_acre*area,2)
    output = round(yield_per_acre, 2)
    
    # Prices per kg and cost of cultivation per acre
    prices = {'paddy': 11.45, 'chilli': 155.6, 'maize': 22.56, 'bengal gram': 47.5, 'groundnut': 63.5}
    cultivation_costs = {'paddy': 72338, 'maize': 34463, 'bengal gram': 19996, 'groundnut': 51898, 'chilli': 62672}
    
    # Calculate income and profit
    income_per_acre = output * prices.get(crop, 0)
    total_income = income_per_acre * area
    profit_per_acre = income_per_acre - cultivation_costs.get(crop, 0)
    total_profit = profit_per_acre * area

    return render_template('output.html', 
                           crop=crop.capitalize(), 
                           district=district, 
                           area=area,
                           rainfall=rainfall, 
                           max_temp=max_temp, 
                           min_temp=min_temp, 
                           yield1=yield1,
                           total_income=round(total_income, 2), 
                           total_profit=round(total_profit, 2),
                           prediction_text=f'Predicted Total Yield per Acre: {output} kg')


if __name__ == "__main__":
    app.run(debug=True)
