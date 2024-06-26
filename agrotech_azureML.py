# REAL TIME REGRESSION INFERENCE APP WITH AZURE

import streamlit as st
import urllib.request
import json
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

# # read JSON file with countries list
with open('outputs/countries_final.json', 'r') as f:
    countries = json.load(f)
        
API_KEY = 'xxx' #Insert key generated by AzureML

URL = 'xxx' #Insert endpoint REST generated by AzureML

st.markdown("<p style='color: darkgreen; font-size: 36px; text-align: center;'>CropWise 🌱</p>", unsafe_allow_html=True)
st.markdown("<p style='color: darkgreen; font-size: 24px; text-align: center;'>A crop recommendation platform using <strong>AzureML</strong></p>", unsafe_allow_html=True)
  
# define ranges
area_min = 1.0
area_max = 3000000.0  
prec_min = 0.05
prec_max = 4000.0
temp_min = - 5.0
temp_max = 30.0

with st.form("best_crop_prediction_form"):
    country = st.selectbox('Country:', countries)
    # create sliders to select ranges
    area_harvested = st.slider('Harvested area (ha):', area_min, area_max)
    prec = st.slider('Rainfall per year (mm):', prec_min, prec_max)
    temp = st.slider('Temperature (ºC):', temp_min, temp_max)
    submit_button = st.form_submit_button(label='Predict best crop')
    
if submit_button:
    # Data structure for the POST request
    data = {
        "input_data": {
    "columns": [
      "Area",
      "area_harvested_ha",
      "avg_rainfall_mm_year",
      "avg_temp_ºC",
    ],
            "index": [0],
            "data": [[country,area_harvested,prec,temp]]
        }
    }

    # Convert to JSON
    body = str.encode(json.dumps(data))
 
    headers = {
        'Content-Type': 'application/json',
        'Authorization': ('Bearer ' + API_KEY)
    }
 
    # Create and send the POST request
    try:
        req = urllib.request.Request(URL, body, headers)
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            predicted_price = result[0]
            st.markdown(f"### Best predicted crop: **{predicted_price}**")
           
    except urllib.error.HTTPError as error:
        st.error(f"Error en la solicitud: {error.code}")
        st.write(error.info())
        st.write(error.read().decode("utf8", 'ignore'))