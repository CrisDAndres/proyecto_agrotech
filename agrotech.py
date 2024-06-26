# ---------------------LIBRARIES----------------------#
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import base64
# Visualization
import plotly.graph_objects as go
# modelling and forecasting
import json
from joblib import load
import zipfile


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

page = option_menu(None, ["Home", "Crops", "Pesticides", "Fertilizers", "Predictions"], 
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
if page == "Home":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### Introduction')
        st.markdown("""
            <p style='font-size:20px; text-align: justify;'>
            The agricultural sector faces multiple challenges that hinder its efficiency and profitability.
            <strong>Agricultural technology</strong> aims to harness the power of machine learning algorithms to predict crop requirements and parameters,
            providing farmers with personalized crop recommendations based on various variables determined by their specific circumstances,
            such as <strong>climatic conditions</strong> or <strong>soil characteristics</strong>.
            </p>
            """,
            unsafe_allow_html=True)
        
        st.image("img/31816.jpg", width=400, caption="Image generated by AI", use_column_width=True)
        
        st.markdown("""
            <p style='font-size:20px; text-align: justify;'>
                Advances in the collection of data from multiple sources, such as satellite imagery, meteorological records, soil analysis and crop growth models, have made it possible to train machine learning models to provide valuable information for improved crop selection and management. This optimises agricultural production and increases profitability.
                In addition, this approach makes a significant contribution to reducing the carbon footprint by promoting sustainable farming practices that not only reduce waste but also improve long-term soil viability.
            </p>
            """,
            unsafe_allow_html=True)
        
        st.image("img/2952.jpg",width=400,caption="Image generated by AI",use_column_width=True) 

    with col2:
        st.markdown('### About the project')
        st.markdown("""
            <p style='font-size:20px; text-align: justify;'>
                For the data analysis of this project, several data sources were used, and data analysis techniques such as 
                <b>data extraction, pre-processing, exploratory data analysis and the implementation of machine learning models</b> 
                were applied to create a crop prediction platform.
            </p>
            """,
            unsafe_allow_html=True)
        st.write('-------------------')
        st.markdown('### Data source:')
        
        st.markdown("""
                    - [*FAOSTAT: Food and Agriculture Organization of the United Nations*](https://www.fao.org/faostat/en/#release_calendar)
                    - [*The world bank*](https://databank.worldbank.org/)
                    - *https://tradingeconomics.com/country-list/temperature*
                    """
                    )
        st.write('---------------------')
        st.write('<img src="https://emojicdn.elk.sh/👩🏻‍💻" width="50" height="50">', unsafe_allow_html=True)
        st.markdown('<a style="font-size: 20px; color: blue;" href="www.linkedin.com/in/cristinadeandres"></i> LinkedIn profile</a>', unsafe_allow_html=True)

    

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
                        - **Average annual rainfall**:
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
            st.markdown('Correlation between variables:')
            st.image("img/corr.png") 
            st.markdown("""
                        Some of the conclusions are:
                        - **Harvested area** has a high positive correlation with **crop production**. The larger the harvested area, the higher the yield.
                        - There is a moderate negative correlation between **yield** and **harvested area**, suggesting that the higher the harvested area, the lower the yield.
                        - The **average temperature** of the country follows a moderate negative correlation with **yield**. The higher the temperature, the lower the yield of some crops.""")
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
   
    # question 1
    st.markdown("<div style='font-size: 24px;'><strong>Has the use of pesticides increased in the last 11 years?</strong></div>", unsafe_allow_html=True)
    # Open html file with the information of the maps generated with folium in read mode.
    HtmlFile = open("html/pesticides1.html", 'r', encoding='utf-8')
    # Read and load into source_code variable
    source_code = HtmlFile.read() 
    # view the content on streamlit
    components.html(source_code, height = 400)
    st.markdown('**The use of pesticides has increased in recent decades due to the need for higher crop yields and increased food production.**')
    
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
    components.html(source_code, height = 530)
    st.markdown('**Herbicides**, used to control weeds and other unwanted plants, are the most commonly used type of pesticide, followed by **insecticides**, which act against insects, and **fungicides and bactericides**, which control fungi and bacteria that harm crops.')
    
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
    <iframe title="proyecto_agrotech" width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiZThkOTUxOTktYjNmNC00MmZlLWFlMGQtYWVkYmE2ZTRiODAzIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9" frameborder="0" allowFullScreen="true"></iframe>
    </div>
    """  
    # Insert HTML code 
    components.html(html_code, height = 1000)

# PAGE 5-------------------------------------
elif page == "Predictions":
    st.markdown("<p style='color: darkgreen; font-size: 36px; text-align: center;'>CropWise 🌱</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: darkgreen; font-size: 24px; text-align: center;'>A crop recommendation platform using machine learning</p>", unsafe_allow_html=True)

   
    # Function to load models from ZIP files
    def load_model_from_zip(zip_file_path, file_name):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            with zip_ref.open(file_name) as file:
                model = load(file)
        return model

    # Function to load files from JSON
    def load_data_from_json_files():
        with open('outputs/encoder_area.json', 'r') as f:
            encoder_area = json.load(f)
        with open('outputs/encoder_crop.json', 'r') as f:
            encoder_crop = json.load(f)
        with open('outputs/decoder_area.json', 'r') as f:
            decoder_area = json.load(f)
        with open('outputs/decoder_crop.json', 'r') as f:
            decoder_crop = json.load(f)
        with open('outputs/countries_final.json', 'r') as f:
            countries = json.load(f)
        with open('outputs/crops.json', 'r') as f:
            crops = json.load(f)
        return encoder_area, encoder_crop, decoder_area, decoder_crop, countries, crops

    # Load models and data only once when starting the application with st.cache_data
    @st.cache_resource()
    def load_models_and_data():
        # load model from ZIP
        model_classif = load_model_from_zip("models/crop_RF.zip", "crop_RF.pkl")
        # load scaler
        scaler_classif = load('outputs/scaler_classif.pkl') # classification model scaler
        # load files from JSON
        encoder_area, encoder_crop, decoder_area, decoder_crop, countries, crops = load_data_from_json_files()
        
        return model_classif, scaler_classif, encoder_area, encoder_crop, decoder_area, decoder_crop, countries, crops

    # Start loading of models and data at start of the application
    model_classif, scaler_classif, encoder_area, encoder_crop, decoder_area, decoder_crop, countries, crops = load_models_and_data()
        
     # ---------------------TABS----------------------#
    tab1, tab2 = st.tabs(
        ['Best crop predictor','Yield predictor']) 
    
    # PREDICTOR 1: CLASSIFICATION MODEL ---------------------------------------------------------   

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
                input_data['Area'] = input_data['Area'].replace(decoder_area).infer_objects(copy=False) # .infer_objects(copy=False) to keep the last replace behaviour, because in future versions it will be deleted

                
                predicted_crop = prediction[-1]  # Generally, the prediction is in the last column
                st.write(f"<p style='font-size: 24px; font-weight: bold;'>The best crop based on the selected variables is: {predicted_crop}</p>", unsafe_allow_html=True)

    # PREDICTOR 2: REGRESSION MODEL ---------------------------------------------------------   
        
    with tab2:
        st.markdown('## 🚧 IN PROGRESS 🚧')
    #     # define ranges
    #     area_min = 1.0
    #     area_max = 3000000.0 
    #     prod_min = 0.06
    #     prod_max = 15000000.0   
    #     temp_min = - 5.0
    #     temp_max = 30.0
        
    #     with st.form("yield_prediction_form"): 
            
    #         crop = st.selectbox('Crop:', crops)
    #         # create sliders to select ranges
    #         area_harvested = st.slider('Harvested area (ha):', area_min, area_max,area_min)
    #         production = st.slider('Production (tonnes):', prod_min, prod_max,prod_min)
    #         temp = st.slider('Temperature (ºC):', temp_min, temp_max,temp_min)
    #         submit_button = st.form_submit_button(label='Predict yield')

    #         if submit_button:
    #             input_data = pd.DataFrame([[crop,area_harvested, production, temp]],
    #                                     columns=['Crop','area_harvested_ha', 'production_tonnes', 'avg_temp_ºC']) # Same order as training

    #         # 1- First I encode what the user enters into numbers using the mapping json.
    #             input_data['Crop'] = input_data['Crop'].replace(encoder_crop)

    #         # 2 - Normalise the input data
    #             input_data_scaled = scaler_regr.transform(input_data)


    #         # 3 - Make the prediction with the model
        
    #             prediction = model_regr.predict(input_data_scaled)
                
    #         # 4 - Decode countries using the reverse mapping dictionary.
    #             input_data['Crop'] = input_data['Crop'].replace(decoder_crop).infer_objects(copy=False)

                
    #             predicted_yield = prediction[-1]  
    #             st.write(f"<p style='font-size: 24px; font-weight: bold;'>The prediction of the crop yield based on the selected variables is: {predicted_yield:.2f} hg/ha</p>", unsafe_allow_html=True)
