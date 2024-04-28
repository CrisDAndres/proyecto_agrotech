# ---------------------LIBRARIES----------------------#
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import base64
import gdown
# Visualization
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.graph_objects as go
# modelling and forecasting
import json
from joblib import load
from pycaret.regression import load_model, predict_model

# ---------------------SITE CONFIG----------------------#
st.set_page_config(
    page_title="Agrotech",
    page_icon="🌾",
    layout="wide", 
    initial_sidebar_state="collapsed", # or expanded  
)

# # ---------------------TITLE & LOGO----------------------#

logo = 'img/logo.png'
# Add logo next to the title
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{base64.b64encode(open(logo, 'rb').read()).decode()}" style="width: 100px; height: auto; margin-right: 20px;">
        <h1 style='font-family: Lato; font-size: 30px;'>Analysis of the distribution of primary crops around the world</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# # ---------------------MENU----------------------# 

page = option_menu(None, ["Intro", "Crops", "Pesticides", "Fertilizers", "Predictions"], 
    icons=["house", "tree", "bug", "clipboard-plus","bullseye"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": '#9bc27e'},
    }
)

# ---------------------LOAD DATA----------------------#

# read files from google drive
df_url = 'https://drive.google.com/uc?id=1R1fslBGYK-Jqnb4sjARC5qCof85_MxYy'

# read data
df = pd.read_csv(df_url)   

# define color palette
agro = ['#b2cb91','#9bc27e','#7fa465','#668f4f','#4e6f43','#59533e','#bf9000','#ffd966','#ffe599']
agro_r = ['#ffe599','#ffd966','#bf9000','#59533e','#4e6f43','#668f4f','#7fa465','#9bc27e','#b2cb91']

# ---------------------BACKGROUND IMAGE----------------------#

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
     <style>
        .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: contain 
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("img/wheat_mod.jpg")  


# ---------------------BODY----------------------#

# PAGE 1-------------------------------------
if page == "Intro":
    pass
# PAGE 2-------------------------------------
elif page == "Crops":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Let's explore crops:")
        # Create a country selector on the map
        selected_area = st.selectbox('🌍 Select country or type it in:', ['All Countries'] + list(df['Area'].unique()))
        # If "All Countries" is selected, display all crops without filtering by country
        if selected_area == 'All Countries':
            selected_crops = df['Crop'].unique()
            # Create a second selector to sort the crops
            order_by = st.selectbox('🌾Order crops by:', ['Area Harvested', 'Production', 'Yield'])
            # Create the checkbox to select the order
            ascending_order = st.checkbox("Ascending order", False)
            # Sort crops based on the selected option
            if order_by == 'Area Harvested':
                if ascending_order:
                    sorted_crops = df.groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=True).round(2)
                else:
                    sorted_crops = df.groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=False).round(2)
            elif order_by == 'Production':
                if ascending_order:
                    sorted_crops = df.groupby('Crop')['production_tonnes'].mean().sort_values(ascending=True).round(2)
                else:
                    sorted_crops = df.groupby('Crop')['production_tonnes'].mean().sort_values(ascending=False).round(2)
            else:  # 'Yield'
                if ascending_order:
                    sorted_crops = df.groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=True).round(2)
                else:
                    sorted_crops = df.groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=False).round(2)
            
        else:
        # Filter crop data by selected area
            selected_crops = df[df['Area'] == selected_area]['Crop'].unique()
            # Create a second selector to sort the crops
            order_by = st.selectbox('🌾Order crops by:', ['Area Harvested', 'Production', 'Yield'])
            # Create a second selector to sort the crops
            ascending_order = st.checkbox("Ascending order", False)
            # Sort crops based on the selected option
            if order_by == 'Area Harvested':
                if ascending_order:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=True).round(2)
                else:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=False).round(2)

            elif order_by == 'Production':
                if ascending_order:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['production_tonnes'].mean().sort_values(ascending=True).round(2)
                else:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['production_tonnes'].mean().sort_values(ascending=False).round(2)
                
            else:  # 'Yield'
                if ascending_order:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=True).round(2)
                else:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=False).round(2)
            
        # Show crops in order
        st.write('Crops in', selected_area, 'ordered by average', order_by, 'between 2010 and 2022:')
        st.write(sorted_crops)

    with col2:
        # ---Interactive maps
        st.markdown('### Interactive maps')
        
        interactive_maps = st.checkbox("Show interactive maps", False)
        if interactive_maps:
                
            st.markdown("""
                        Here you can explore between 3 different layers:
                        - **Average area harvested**
                        - **Average yield**
                        - **Average annual rainfall** around the globe:
                        """)
            # Open html file with the information of the maps generated with folium in read mode.
            HtmlFile = open("html/agromap.html", 'r', encoding='utf-8')
            # Read and load into source_code variable
            source_code = HtmlFile.read() 
            print(source_code)
            # view the content on streamlit
            components.html(source_code, height = 600)
        
        # ---Correlation
        st.markdown('### Correlation')
        
        correlation = st.checkbox("Show correlation", False)
        if correlation:
            # Display an image from a local file
            st.image("img/corr.png") 
            st.markdown('Correlation between variables:')

        # ---Data over time
        st.markdown('### Data over time')
        
        time = st.checkbox("Show data VS time", False)
        if time:
            # define variables
            areaharvestedbyyear = df.groupby('Year')['area_harvested_ha'].mean() 
            productionbyyear = df.groupby('Year')['production_tonnes'].mean() 
            yieldbyyear = df.groupby('Year')['yield_hg/ha'].mean() 
            
            # Create the selectbox to choose from the options
            selected_option_time = st.selectbox('Select data:', ['Area harvested VS time', 'Production VS time', 'Yield VS time'])
            
            if selected_option_time =='Area harvested VS time':
                # Plotly Graph
                fig = go.Figure(data=go.Scatter(x=areaharvestedbyyear.index, y=areaharvestedbyyear.values, line=dict(color='#9bc27e')))
                fig.update_layout(xaxis_title='Year', yaxis_title='Average area harvested (ha)', title='Average area harvested per Year',width=600, height=400,xaxis=dict(tickmode='linear', dtick=1),title_x=0.3) 
                st.plotly_chart(fig)
            elif selected_option_time == 'Production VS time':
                # Plotly Graph
                fig = go.Figure(data=go.Scatter(x=productionbyyear.index, y=productionbyyear.values, line=dict(color='#9bc27e')))
                fig.update_layout(xaxis_title='Year', yaxis_title='Average production (tonnes)', title='Average production per Year',width=600, height=400,xaxis=dict(tickmode='linear', dtick=1),title_x=0.3) 
                st.plotly_chart(fig)
            else:
                # Plotly Graph
                fig = go.Figure(data=go.Scatter(x=yieldbyyear.index, y=yieldbyyear.values, line=dict(color='#9bc27e')))
                fig.update_layout(xaxis_title='Year', yaxis_title='Average Yield (hg/ha)', title='Average Yield per Year',width=600, height=400,xaxis=dict(tickmode='linear', dtick=1),title_x=0.35) 
                st.plotly_chart(fig)

# PAGE 3-------------------------------------
elif page == "Pesticides": 
    col1, col2 = st.columns(2)
    with col1:
   
        # question 1
        st.markdown("<div style='font-size: 24px;'><strong>Has the use of pesticides increased in the last 15 years?</strong></div>", unsafe_allow_html=True)
        # Open html file with the information of the maps generated with folium in read mode.
        HtmlFile = open("html/pesticides1.html", 'r', encoding='utf-8')
        # Read and load into source_code variable
        source_code = HtmlFile.read() 
        # view the content on streamlit
        components.html(source_code, height = 600)

        # question 2
        st.markdown("<div style='font-size: 24px;'><strong>Does the use of pesticides affect crop yield?</strong></div>", unsafe_allow_html=True)
        st.image("img/corr_pest.png",width=800) 
        st.markdown('**There is a moderate positive correlation between pesticide use and crop yields**') 

        # question 3 
        st.markdown("<div style='font-size: 24px;'><strong>What is the most commonly used type of pesticide?</strong></div>", unsafe_allow_html=True)
        HtmlFile = open("html/pesticides2.html", 'r', encoding='utf-8')
        # Read and load into source_code variable
        source_code = HtmlFile.read() 
        # view the content on streamlit
        components.html(source_code, height = 600)

        # question 4
        st.markdown("<div style='font-size: 24px;'><strong>Which are the top 50 countries with the highest levels of pesticide use in tonnes?</strong></div>", unsafe_allow_html=True)
        HtmlFile = open("html/pesticides3.html", 'r', encoding='utf-8')
        # Read and load into source_code variable
        source_code = HtmlFile.read() 
        # view the content on streamlit
        components.html(source_code, height = 600)
    
# PAGE 4-------------------------------------
elif page == "Fertilizers": 
    st.markdown("<h1 style='text-align: center; color: black;'> <strong>Interactive PowerBI panel</strong> </h1>", unsafe_allow_html=True)
    # Power BI panel HTML code centered
    html_code = """
    <div style="display: flex; justify-content: center;">
        <iframe title="proyecto_agrotech" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=d5e9d64d-7a09-4db0-8444-5725a56b2cd4&autoAuth=true&ctid=8aebddb6-3418-43a1-a255-b964186ecc64" frameborder="0" allowFullScreen="true"></iframe>
    </div>
    """  
    # Insert HTML code 
    components.html(html_code, height = 1000)

# PAGE 5-------------------------------------
elif page == "Predictions":
    st.markdown("<p style='color: darkgreen; font-size: 36px; text-align: center;'>CropWise 🌱</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: darkgreen; font-size: 24px; text-align: center;'>A crop recommendation platform using machine learning</p>", unsafe_allow_html=True)
    # download models files from google drive
    url1 = "https://drive.google.com/uc?id=1TfydDkRqT2zJINmXc6RvrG7HOEuxkZgG"
    url2 = "https://drive.google.com/uc?id=1jRCQOX5_n6-Z-KtTZaCaDg-HNIneG6wN"
    
    model_classif = "crop_RF.pkl"
    model_regr = "yield_RF.pkl"
    
    gdown.download(url1, model_classif, quiet=True) 
    gdown.download(url2, model_regr, quiet=True) 
    
    # upload models
    model_classif = load(model_classif) # classification Random Forest model
    model_regr = load(model_regr) # regression Random Forest model
    
    # upload scalers
    scaler_classif = load('scaler_classif.pkl') # classification model scaler
    scaler_regr = load('scaler_regr.pkl') # regression model scaler

    # read JSON files with encoder and decoder
    with open('json/encoder_area.json', 'r') as f:
        encoder_area = json.load(f)
    
    with open('json/encoder_crop.json', 'r') as f:
        encoder_crop = json.load(f)

    with open('json/decoder_area.json', 'r') as f:
        decoder_area = json.load(f)

    with open('json/decoder_crop.json', 'r') as f:
        decoder_crop = json.load(f)
        
# read JSON file with countries list
    with open('json/countries_final.json', 'r') as f:
        countries = json.load(f)
# read JSON file with crop list
    with open('json/crops.json', 'r') as f:
        crops = json.load(f)
        
     # ---------------------TABS----------------------#
    tab1, tab2 = st.tabs(
        ['Best crop predictor','Yield predictor']) 
    
    # PREDICTOR 1 ---------------------------------------------------------   

    with tab1:
        
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
                input_data = pd.DataFrame([[country,area_harvested, prec, temp]],
                                        columns=['Area','area_harvested_ha', 'avg_rainfall_mm_year', 'avg_temp_ºC']) # Same order as training

            # 1- First I encode what the user enters into numbers using the mapping json.
                input_data['Area'] = input_data['Area'].replace(encoder_area)

            # 2 - Normalise the input data
                input_data_scaled = scaler_classif.transform(input_data)
    
            # 3 - Make the prediction with the model
        
                prediction = model_classif.predict(input_data_scaled)
                
            # 4 - Decode countries using the reverse mapping dictionary.
                input_data['Area'] = input_data['Area'].replace(decoder_area)

                
                predicted_crop = prediction[-1]  # Generally, the prediction is in the last column
                st.write(f"<p style='font-size: 24px; font-weight: bold;'>The best crop based on the selected variables is: {predicted_crop}</p>", unsafe_allow_html=True)

    # PREDICTOR 2 ---------------------------------------------------------   
        
    with tab2:
        # define ranges
        area_min = 1.0
        area_max = 3000000.0 
        prod_min = 0.06
        prod_max = 15000000.0   
        temp_min = - 5.0
        temp_max = 30.0
        
        with st.form("yield_prediction_form"): 
            
            crop = st.selectbox('Crop:', crops)
            # create sliders to select ranges
            area_harvested = st.slider('Harvested area (ha):', area_min, area_max,area_min)
            production = st.slider('Production (tonnes):', prod_min, prod_max,prod_min)
            temp = st.slider('Temperature (ºC):', temp_min, temp_max,temp_min)
            submit_button = st.form_submit_button(label='Predict yield')

            if submit_button:
                input_data = pd.DataFrame([[crop,area_harvested, production, temp]],
                                        columns=['Crop','area_harvested_ha', 'production_tonnes', 'avg_temp_ºC']) # Same order as training

            # 1- First I encode what the user enters into numbers using the mapping json.
                input_data['Crop'] = input_data['Crop'].replace(encoder_crop)

            # 2 - Normalise the input data
                input_data_scaled = scaler_regr.transform(input_data)


            # 3 - Make the prediction with the model
        
                prediction = model_regr.predict(input_data_scaled)
                
            # 4 - Decode countries using the reverse mapping dictionary.
                input_data['Crop'] = input_data['Crop'].replace(decoder_crop)

                
                predicted_yield = prediction[-1]  
                st.write(f"<p style='font-size: 24px; font-weight: bold;'>The prediction of the crop yield based on the selected variables is: {predicted_yield:.2f} hg/ha</p>", unsafe_allow_html=True)
