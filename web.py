import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and label encoder
model = joblib.load("./model/model.pkl")
label_encoder = joblib.load("./model/label_encoder.pkl")
scalers = joblib.load("./model/scalers.pkl")

# Load the dataset
df = pd.read_csv("updated_pollution_dataset.csv")
columns = df.columns
target = "Air Quality"
attribute = columns.drop(target)


st.set_page_config(
    page_icon="✈️",
    layout="wide",
)

def data_visualization():
    st.title("Air Quality and Pollution Assessment Data Visualization")
    
    st.markdown(
    '''
    ## Dataset
    The file `updated_pollution_dataset.csv` is a dataset sourced from [Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment/). This dataset focuses on assessing air quality in various regions. It contains 5000 samples and includes key environmental and demographic factors affecting pollution levels.
    
    Key Features:

    1. **Temperature (°C)**: Average temperature of the region.
    2. **Humidity (%)**: Relative humidity recorded in the region.
    3. **PM2.5 Concentration (µg/m³)**: Fine particulate matter levels.
    4. **PM10 Concentration (µg/m³)**: Coarse particulate matter levels.
    5. **NO2 Concentration (ppb)**: Nitrogen dioxide levels.
    6. **SO2 Concentration (ppb)**: Sulfur dioxide levels.
    7. **CO Concentration (ppm)**: Carbon monoxide levels.
    8. **Proximity to Industrial Areas (km)**: Distance to the nearest industrial zone.
    9. **Population Density (people/km²)**: Number of people per square kilometer in the region.

    Target Variable: **Air Quality**

    - **Good**: Clean air with low pollution levels.
    - **Moderate**: Acceptable air quality but with some pollutants present.
    - **Poor**: Noticeable pollution that may cause health issues for sensitive groups.
    - **Hazardous**: Highly polluted air posing serious health risks to the population.

    This dataset will be used to build a model capable of predicting air quality in a region based on the input *Key Features*. The model implemented will be a `RandomForestClassifier`, an ensemble learning method for classification tasks.
    ''')


    # Define custom colors for classes
    custom_colors = {
        "Good": "#2ecc71",     
        "Moderate": "#f1c40f", 
        "Poor": "#e67e22",      
        "Hazardous": "#e74c3c" 
    }

    # Visualize the target class distribution
    st.markdown("### Target Class Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Countplot
    sns.countplot(data=df, x="Air Quality", ax=axes[0], palette=custom_colors)
    axes[0].set_title("Distribution of Air Quality Classes - Countplot")
    axes[0].set_xlabel("Air Quality")
    axes[0].set_ylabel("Count")

    # Pie chart
    air_quality_counts = df["Air Quality"].value_counts()
    axes[1].pie(
        air_quality_counts, 
        labels=air_quality_counts.index, 
        autopct='%1.1f%%', 
        colors=[custom_colors[label] for label in air_quality_counts.index]
    )
    axes[1].set_title("Distribution of Air Quality Classes - Pie Chart")

    st.pyplot(fig)


    st.markdown("### Key Feature Distribution")
    distribution_data_selection = st.multiselect("Select columns to visualize", attribute, default=["Temperature", "Humidity"])

    # Plot the distribution for each selected column
    if distribution_data_selection:
        fig, axes = plt.subplots(len(distribution_data_selection), 2, figsize=(20, 6 * len(distribution_data_selection)))
        if len(distribution_data_selection) == 1:
            axes = [axes] 
        for ax_pair, column in zip(axes, distribution_data_selection):
            hist_ax, box_ax = ax_pair
            sns.histplot(data=df, x=column, kde=True, ax=hist_ax)
            hist_ax.set_title(f"Distribution of {column} - Histogram")
            sns.boxplot(data=df, x=column, ax=box_ax)
            box_ax.set_title(f"Distribution of {column} - Boxplot")
        st.pyplot(fig)
    
    # Multiselect for target-column relationship with boxplots
    st.markdown("### Target Column Relationships")
    target_column_selected_column = st.multiselect("Select columns to visualize with respect to Air Quality", attribute, default=["Temperature", "Humidity"])

    if target_column_selected_column:
        fig, axes = plt.subplots(len(target_column_selected_column), 1, figsize=(20, 6 * len(target_column_selected_column)))
        if len(target_column_selected_column) == 1:
            axes = [axes]  # Ensure axes is iterable for a single plot
        for ax, column in zip(axes, target_column_selected_column):
            sns.boxplot(data=df, x="Air Quality", y=column, ax=ax, palette=custom_colors)
            ax.set_title(f"Relationship of {column} with Air Quality")
            ax.set_xlabel("Air Quality")
            ax.set_ylabel(column)
        st.pyplot(fig)

    # Humidity vs. Temperature Scatter Plot
    st.markdown("### Humidity vs. Temperature")
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.scatterplot(x='Temperature', y='Humidity', data=df, ax=ax)
    ax.set_title('Humidity vs. Temperature')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Humidity')
    st.pyplot(fig)


def model_prediction():
    st.title("Air Quality Prediction")
    st.write("Provide input values to predict air quality class.")

    # Input fields for the user
    temperature = st.number_input("Temperature (°C)", value=29.8)
    humidity = st.number_input("Humidity (%)", value=59.1)
    pm25 = st.number_input("PM2.5 Concentration (µg/m³)", value=5.2)
    pm10 = st.number_input("PM10 Concentration (µg/m³)", value=17.9)
    no2 = st.number_input("NO2 Concentration (ppb)", value=18.9)
    so2 = st.number_input("SO2 Concentration (ppb)", value=9.2)
    co = st.number_input("CO Concentration (ppm)", value=1.72)
    proximity_to_industrial_areas = st.number_input("Proximity to Industrial Areas (km)", value=6.3)
    population_density = st.number_input("Population Density (people/km²)", value=319)

    # Collect input into a feature dictionary
    input_data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'SO2': so2,
        'CO': co,
        'Proximity_to_Industrial_Areas': proximity_to_industrial_areas,
        'Population_Density': population_density
    }

    # Apply scalers to the input data
    scaled_features = []
    for col, value in input_data.items():
        scaler = scalers.get(col)
        if scaler:
            scaled_value = scaler.transform([[value]])[0][0]
            scaled_features.append(scaled_value)
        else:
            scaled_features.append(value)

    features_array = np.array(scaled_features).reshape(1, -1)

    if st.button("Predict"):
        # Perform prediction
        prediction = model.predict(features_array)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        st.success(f"The predicted air quality class is: {predicted_class}")

# Main logic
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Visualization", "Model Prediction"])

if page == "Data Visualization":
    data_visualization()
elif page == "Model Prediction":
    model_prediction()