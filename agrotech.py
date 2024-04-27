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
# Agregar el logo de la empresa al lado del título
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{base64.b64encode(open(logo, 'rb').read()).decode()}" style="width: 100px; height: auto; margin-right: 20px;">
        <h1 style='font-family: Lato; font-size: 30px;'>Analysis of the distribution of primary crops around the world</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# # ---------------------MENU----------------------# clipboard-plus

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

# Descargar archivos desde Google Drive
url = 'https://drive.google.com/file/d/1R1fslBGYK-Jqnb4sjARC5qCof85_MxYy/view?usp=sharing'
df = 'df_preprocessed.csv'
gdown.download(url, df, quiet=False)
url = 'https://drive.google.com/file/d/1kvYkeQ3aDCkT2DX8OwlcPrWlOZRNY8S8/view?usp=drive_link'
pest_df = 'pest_preprocessed.csv'
gdown.download(url, pest_df, quiet=False)
url = 'https://drive.google.com/file/d/1Xz4M1IUWEUkJdoOU70rgPgFCROIqTPKe/view?usp=drive_link'
pest = 'pest_crops.csv'
gdown.download(url, pest, quiet=False)

# read data
df = pd.read_csv("Data/df_preprocessed.csv")
pest_df = pd.read_csv("Data/pest_preprocessed.csv") # original pest dataframe
pest = pd.read_csv("Data/pest_crops.csv")    

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

# st.markdown(
#     f"""
#     <style>
#     [data-testid="stHeader"] {{
#     background-color: rgba(0, 0, 0, 0);
#     }}
#     [data-testid="stSidebar"]{{                 
#     background-color: rgba(0, 0, 0, 0);
#     border: 0.5px solid #59533e;
#         }}
#     </style>
#     """
# , unsafe_allow_html=True)


# ---------------------BODY----------------------#

# PAGE 1-------------------------------------
if page == "Intro":
    pass
# PAGE 2-------------------------------------
elif page == "Crops":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Let's explore crops:")
        # Crear un selector de área en el mapa
        selected_area = st.selectbox('🌍 Select country or type it in:', ['All Countries'] + list(df['Area'].unique()))
        # Si se selecciona "All Countries", mostrar todos los cultivos sin filtrar por país
        if selected_area == 'All Countries':
            selected_crops = df['Crop'].unique()
            # Crear un segundo selector para ordenar los cultivos
            order_by = st.selectbox('🌾Order crops by:', ['Area Harvested', 'Production', 'Yield'])
            # Crea la checkbox para seleccionar el orden
            ascending_order = st.checkbox("Ascending order", False)
            # Ordenar los cultivos en base a la opción seleccionada
            if order_by == 'Area Harvested':
                if ascending_order:
                    sorted_crops = df.groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=True)
                else:
                    sorted_crops = df.groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=False)
            elif order_by == 'Production':
                if ascending_order:
                    sorted_crops = df.groupby('Crop')['production_tonnes'].mean().sort_values(ascending=True)
                else:
                    sorted_crops = df.groupby('Crop')['production_tonnes'].mean().sort_values(ascending=False)
            else:  # 'Yield'
                if ascending_order:
                    sorted_crops = df.groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=True)
                else:
                    sorted_crops = df.groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=False)
            
        else:
        # Filtrar los datos de los cultivos por el área seleccionada
            selected_crops = df[df['Area'] == selected_area]['Crop'].unique()
            # Crear un segundo selector para ordenar los cultivos
            order_by = st.selectbox('🌾Order crops by:', ['Area Harvested', 'Production', 'Yield'])
            # Crea la checkbox para seleccionar el orden
            ascending_order = st.checkbox("Ascending order", False)
            # Ordenar los cultivos en base a la opción seleccionada
            if order_by == 'Area Harvested':
                if ascending_order:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=True)
                else:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['area_harvested_ha'].mean().sort_values(ascending=False)

            elif order_by == 'Production':
                if ascending_order:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['production_tonnes'].mean().sort_values(ascending=True)
                else:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['production_tonnes'].mean().sort_values(ascending=False)
                
            else:  # 'Yield'
                if ascending_order:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=True)
                else:
                    sorted_crops = df[df['Area'] == selected_area].groupby('Crop')['yield_hg/ha'].mean().sort_values(ascending=False)
            
        # Mostrar los cultivos ordenados
        st.write('Crops in', selected_area, 'ordered by average', order_by, 'between 2010 and 2022:')
        st.write(sorted_crops)

    with col2:
        # Interactive maps
        st.markdown('### Interactive maps')
        # Selector de radio para elegir entre las opciones
        interactive_maps = st.checkbox("Show interactive maps", False)
        if interactive_maps:
                
            st.markdown("""
                        Here you can explore between 3 different layers:
                        - **Average area harvested**
                        - **Average yield**
                        - **Average annual rainfall** around the globe:
                        """)
            # Abrir archivo html con la información de los mapas generados con folium en modo lectura
            HtmlFile = open("html/agromap.html", 'r', encoding='utf-8')
            # Leer y cargar en la variable source_code
            source_code = HtmlFile.read() 
            print(source_code)
            # visualizar el contenido en streamlit
            components.html(source_code, height = 600)
        # Correlation
        st.markdown('### Correlation')
        # Selector de radio para elegir entre las opciones
        correlation = st.checkbox("Show correlation", False)
        if correlation:
            # Muestra una imagen desde un archivo local
            st.image("img/corr.png") 
            st.markdown('Correlation between variables:')

        # Data over time
        st.markdown('### Data over time')
        # Selector de radio para elegir entre las opciones
        time = st.checkbox("Show data VS time", False)
        if time:
            # define variables
            areaharvestedbyyear = df.groupby('Year')['area_harvested_ha'].mean() 
            productionbyyear = df.groupby('Year')['production_tonnes'].mean() 
            yieldbyyear = df.groupby('Year')['yield_hg/ha'].mean() 
            
            # Crear la selectbox para elegir entre las opciones
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
        pestbyyear = pest.groupby('Year')['agricultural_use_tonnes'].mean()
        fig = go.Figure(data=go.Scatter(x=pestbyyear.index, y=pestbyyear.values, line=dict(color=agro[3])))
        fig.update_layout(xaxis_title='Year', yaxis_title='Pesticides use (tonnes)', title='Average pesticides use per Year',width=800, height=400,xaxis=dict(tickmode='linear', dtick=1),title_x=0.35) 
        st.plotly_chart(fig) 

        # question 2
        st.markdown("<div style='font-size: 24px;'><strong>Does the use of pesticides affect crop yield?</strong></div>", unsafe_allow_html=True)
        st.image("img/corr_pest.png",width=800) 
        st.markdown('**There is a moderate positive correlation between pesticide use and crop yields**') 

        # question 3 
        st.markdown("<div style='font-size: 24px;'><strong>What is the most commonly used type of pesticide?</strong></div>", unsafe_allow_html=True)
        pest_total_area = pest_df.groupby(['Area','pesticides_type'])['agricultural_use_tonnes'].mean().reset_index()     
        new_palette = ['#BDF08D', '#1D761D','#C0A45E', '#764E1D', '#F5F78E', '#F7CC8E']
        fig = px.pie(pest_total_area, values='agricultural_use_tonnes', names='pesticides_type',color_discrete_sequence=new_palette,labels={'agricultural_use_tonnes': 'Use of pesticides (tonnes)', 'pesticides_type': 'Pesticide type'})
        fig.update_layout(
            template="plotly_white",
            title="Average pesticides use by type",
            title_x=0.35,
            width=800, height=500,
            legend={
                'x': 1,  
                'y': 0.5})
        fig.update_traces(
            textinfo='label+percent',
            textfont_size=12,
            text=pest_total_area['pesticides_type'],
            marker=dict(line=dict(color='black', width=0.5))
        )
        st.plotly_chart(fig)  

        # question 4
        st.markdown("<div style='font-size: 24px;'><strong>Which are the top 50 countries with the highest levels of pesticide use in tonnes?</strong></div>", unsafe_allow_html=True)
        pest_total_area_sorted = pest_total_area.sort_values(by='agricultural_use_tonnes', ascending=False)
        top50 = pest_total_area_sorted.head(50)
        top50['Area'].replace('TÃ¼rkiye','Turkey',inplace=True)
        fig = px.bar(
            top50,
            y='Area',
            x='agricultural_use_tonnes',
            color='Area',  
            color_discrete_sequence=agro,
            labels={'agricultural_use_tonnes': 'Use of pesticides (tonnes)', 'pesticides_type': 'Pesticide type'}
        )
        # Especificar el orden de las categorías en el eje y
        fig.update_layout(title='top50 countries using pesticides',title_x=0.35,yaxis={'categoryorder':'total ascending'},width=800, height=600,showlegend=False)
        st.plotly_chart(fig)
    
# PAGE 4-------------------------------------
elif page == "Fertilizers": 
    st.markdown("<h1 style='text-align: center; color: black;'> <strong>Interactive PowerBI panel</strong> </h1>", unsafe_allow_html=True)
    # Power BI panel HTML code
    html_code = """<iframe title="proyecto_agrotech" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=d5e9d64d-7a09-4db0-8444-5725a56b2cd4&autoAuth=true&ctid=8aebddb6-3418-43a1-a255-b964186ecc64" frameborder="0" allowFullScreen="true"></iframe>"""
    # Insert HTML code 
    components.html(html_code, height = 1000)
# PAGE 5-------------------------------------
elif page == "Predictions":
    st.markdown("<p style='color: darkgreen; font-size: 36px; text-align: center;'>CropWise 🌱</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: darkgreen; font-size: 24px; text-align: center;'>A crop recommendation platform using machine learning</p>", unsafe_allow_html=True)

    ## -- Carga de archivos 
    scaler_regr = load('scaler_regr.pkl') # scaler del modelo de regresion
    scaler_classif = load('scaler_classif.pkl') # scaler del modelo de clasificacion
    model_regr = load_model('models/yield_RF') #Le cargamos el modelo (de regresion) que he entrenado con rf
    model_classif = load_model('models/crop_RF') #Le cargamos el modelo (de clasificacion) que he entrenado con rf

    # # read JSON file with countries list
    with open('json/encoder_area.json', 'r') as f:
        encoder_area = json.load(f)
    # read JSON file with crop list
    with open('json/encoder_crop.json', 'r') as f:
        encoder_crop = json.load(f)
    # # read JSON file with countries list
    with open('json/decoder_area.json', 'r') as f:
        decoder_area = json.load(f)
    # read JSON file with crop list
    with open('json/decoder_crop.json', 'r') as f:
        decoder_crop = json.load(f)
# # read JSON file with countries list
    with open('json/countries_final.json', 'r') as f:
        countries = json.load(f)
# read JSON file with crop list
    with open('json/crops.json', 'r') as f:
        crops = json.load(f)
     # ---------------------TABS (pestañas)----------------------#
    tab1, tab2 = st.tabs(
        ['Best crop predictor','Yield predictor']) 
    
    # PREDICTOR 1 ---------------------------------------------------------   

    with tab1:
        

        # definir rangos
        area_min = 1.0
        area_max = 3000000.0  
        prec_min = 0.05
        prec_max = 4000.0
        temp_min = - 5.0
        temp_max = 30.0
        
        with st.form("best_crop_prediction_form"): #Metemos todas las variables que hemos usado en el entrenamiento, en el mismo orden
            country = st.selectbox('Country:', countries)
            # creamos sliders para seleccionar rangos
            area_harvested = st.slider('Harvested area (ha):', area_min, area_max)
            prec = st.slider('Rainfall per year (mm):', prec_min, prec_max)
            temp = st.slider('Temperature (ºC):', temp_min, temp_max)
            submit_button = st.form_submit_button(label='Predict best crop')

            if submit_button:
                input_data = pd.DataFrame([[country,area_harvested, prec, temp]],
                                        columns=['Area','area_harvested_ha', 'avg_rainfall_mm_year', 'avg_temp_ºC']) # mismo orden que entrenamiento
            # Mismo orden que en el notebook 

            # 1- Primero codifico a números lo que ingresa el usuario utilizando el json de mapeo
                input_data['Area'] = input_data['Area'].replace(encoder_area)

            # 2 - Después normalizo los datos de entrada
                input_data_scaled = scaler_classif.transform(input_data)
    

            # 3 - Realiza la predicción con el modelo
        
                prediction = model_classif.predict(input_data_scaled)
                
              
            # 4 - Por último decodifico los barrios utilizando el diccionario de mapeo inverso
                input_data['Area'] = input_data['Area'].replace(decoder_area)

                # Asegurémonos de acceder al nombre correcto de la columna de predicciones
                predicted_crop = prediction[-1]  # Generalmente, la predicción está en la última columna
                st.write(f"<p style='font-size: 24px; font-weight: bold;'>The best crop based on the selected variables is: {predicted_crop}</p>", unsafe_allow_html=True)

    # PREDICTOR 2 ---------------------------------------------------------   
        
    with tab2:
        # definir rangos
        area_min = 1.0
        area_max = 3000000.0 
        prod_min = 0.06
        prod_max = 15000000.0   
        temp_min = - 5.0
        temp_max = 30.0
        
        with st.form("yield_prediction_form"): #Metemos todas las variables que hemos usado en el entrenamiento, en el mismo orden
            # country = st.selectbox('Country:', countries)
            crop = st.selectbox('Crop:', crops)
            # creamos sliders para seleccionar rangos
            area_harvested = st.slider('Harvested area (ha):', area_min, area_max,area_min)
            production = st.slider('Production (tonnes):', prod_min, prod_max,prod_min)
            temp = st.slider('Temperature (ºC):', temp_min, temp_max,temp_min)
            submit_button = st.form_submit_button(label='Predict yield')

            if submit_button:
                input_data = pd.DataFrame([[crop,area_harvested, production, temp]],
                                        columns=['Crop','area_harvested_ha', 'production_tonnes', 'avg_temp_ºC']) # mismo orden que entrenamiento
            # Mismo orden que en el notebook 

            # 1- Primero codifico a números lo que ingresa el usuario utilizando el json de mapeo
                input_data['Crop'] = input_data['Crop'].replace(encoder_crop)

            # 2 - Después normalizo los datos de entrada
                input_data_scaled = scaler_regr.transform(input_data)


            # 3 - Realiza la predicción con el modelo
        
                prediction = model_regr.predict(input_data_scaled)
                
              
            # 4 - Por último decodifico los barrios utilizando el diccionario de mapeo inverso
                input_data['Crop'] = input_data['Crop'].replace(decoder_crop)

                # Asegurémonos de acceder al nombre correcto de la columna de predicciones
                predicted_yield = prediction[-1]  # Generalmente, la predicción está en la última columna
                st.write(f"<p style='font-size: 24px; font-weight: bold;'>The prediction of the crop yield based on the selected variables is: {predicted_yield:.2f} hg/ha</p>", unsafe_allow_html=True)
