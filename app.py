import streamlit as st
import joblib
import pandas as pd
from streamlit_option_menu import option_menu
import sqlite3
from PIL import Image
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import geocoder
import time
import mysql.connector
import os
from dotenv import load_dotenv


# Set the webpage title and layout 
st.set_page_config(page_title="INVESTIGATION AND PREDICTION OF TERRORISM IN NIGERIA", page_icon='imag3.webp', layout="wide")   

# Add meta tag for viewport
#url("https://your-image-url.jpg");
st.markdown(
    """
    <style>
    .stApp {
        background-image: background.JPG 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    };
    
    .reportview-container {
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap='small')

with col1:
    logo = Image.open('img9.JPG')
    
    # Set the desired width and height
    width = 700
    height = 350
    
    # Resize the image
    logo = logo.resize((width, height))
    
    # Display the resized image
    st.image(logo) 

with col2:
    st.title(':red[INVESTIGATION AND PREDICTION OF TERRORISM IN NIGERIA]')
    st.subheader(":blue[Terrorism will spill over if you don’t speak up]")
    

    #st.write("Malala Yousafzai")
    
with st.sidebar:
    selected = option_menu(menu_title='Main menu', options=['Home', 'Visualizations', 'Prediction', 'Make Report', 'About'], 
                           icons=['house-fill', 'bar-chart-fill', 'globe', 'x-diamond-fill', 'person-fill'],
                           menu_icon="cast", default_index=0)

# Load the trained Random Forest model
model = joblib.load('final_model1.pkl')

# Load the encoders used during training
#ordinal_encoders = joblib.load('re_ordinal_encoders.pkl')  # For year, month, and day
state_label_encoder = joblib.load('state_label_encoder1.pkl')      # For State

# Load the training dataset
training_data = pd.read_csv('dataset_before_encoding.csv')  
full_data = pd.read_csv('Clean_terrorism_db.csv')

# Extract state-specific data
state_data = training_data.groupby('State').agg({
    'mean_pct_read_seng15': 'mean',  
 #   'avg_houshold_size': 'mean'
}).reset_index()

# Home Tab
if selected == "Home":
    # Subtitle
    st.subheader(":blue[INTRODUCTION]")
    st.write("One of the biggest challenges to modern society is still terrorism. In recent years, terrorism has had a detrimental impact on a variety of industries, groups, countries, and the world at large. The different forms of terrorist attacks in Nigeria in the recent times are Boko-Haram attack, Fulani/Herdsmen attack, Inter/Intra-group conflicts, robbery, and lack of intentionality. To curb or reduce these activities in Nigeria, there is a need to develop models that can be used to understand these terrorists’ activities and prevent or reduce future occurrences (Olufemi. et., al. 2022). Terrorism or terrorist activities are unpredictable in themselves since they are likely to be conducted by unknown persons in an unknown place and at unpredicted times. Recent advances in technology have compelled authorities to reconsider how they carry out terrorist operations (McKendrick 2019). Security organizations can take preventative measures and more precisely allocate resources by using machine learning algorithms to help make accurate forecasts.  Hence, an effective method to counter terrorism depends possibly on accurate prediction.")

    st.subheader(":blue[Statement of the Problem]")
    st.write("For a thorough grasp of the variables driving terrorist activity, it is imperative to integrate a variety of data sources, including socioeconomic, cultural, and geopolitical elements. Through creating and improving Machine learning models that address current shortcomings, this research seeks to make predictions of different factors that lead to terrorist activities in Nigeria. The model's interpretability is enhanced by incorporating varied data sources, resolving ethical problems, and applying advanced approaches to adapt to dynamic threats. Concurrently, the development of a distributive terrorist activity database is pivotal for ensuring the availability of high-quality, diverse data, thereby contributing to the enhancement of model performance. These findings offer significant insights for researchers, policymakers, and security organizations involved in effective counterterrorism efforts.")

    st.subheader(":blue[Research Objectives]")
    st.write("The main objective of this study is to develop and evaluate a machine learning models for the classification and forecasting of terrorist activities, with a focus on violent attack. While the specific objectives are to: I.	Create predictive models for terrorist classification based on suicide, successful attack, type of weapon, region, type of attack and targeted location. II.	Extract information from google maps in accurately determining the locations of terrorist attacks using various forms of location data (e.g., latitude and longitude, addresses) III.	Develop a distributive terrorist activity database for easy retrieval of information of terrorist data. ")

    st.subheader(":blue[Scope of the Study]")
    st.write("The scope of this study covers the classification and forecasting of terrorist activities and development of distributive database of terrorism in Nigeria. The project intends to enable real-time adaptation of prediction models.")

    st.subheader(":blue[Significance of the Study]")
    st.write("The relevance of this research rests in its ability to function as a proactive measure that is critical for public safety and national security. The outcomes support global counterterrorism efforts and international cooperation. Policymakers can use these findings to develop effective policies that support regional stability and mitigate the potential humanitarian consequences of attacks. Furthermore, the research raises public awareness, makes terrorist datasets available for future analysis, fortifies intelligence services, and advances academic knowledge, all of which contribute to the development of a comprehensive plan to address the complex issues arising from terrorism in Nigeria.")


# Visualization Tab
elif selected == "Visualizations": 
    st.title(":blue[Visualization]")
    st.write("Visualize the historical data and the model's performance.")
    # Add visualization code here, such as charts or graphs.
    st.write("Visualization content goes here.")

# Prediction Tab
elif selected == "Prediction":
    st.title(":blue[Prediction]")

    
    st.header(":blue[Input Features]")
    st.write("Provide the information below and click the prdict button to make prediction")

    # Input fields for features
    state = st.selectbox("State", options=state_label_encoder.classes_)  # Use label encoder classes for selection

    # Automatically set mean_pct_read_seng15 and avg_unemploy_state based on selected State 
    mean_pct_read_seng15 = state_data[state_data['State'] == state]['mean_pct_read_seng15'].values[0]
   # avg_houshold_size = state_data[state_data['State'] == state]['avg_houshold_size'].values[0]

    
    # Input fields for year, month, and day
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)

    # Create a DataFrame for the inputs
    input_df = pd.DataFrame({
        'mean_pct_read_seng15': [mean_pct_read_seng15],
     #   'avg_houshold_size': [avg_houshold_size],
        'year': [year],
        'month': [month],
        'day': [day],
        'State': [state]
    })

    

    
    input_df['State'] = state_label_encoder.transform(input_df['State'])



    # Add a button for making predictions
    if st.button('Predict'):
        # Make prediction
        probability = model.predict_proba(input_df)[0][1]  # Probability of terrorist attack

        # Display the result
        st.subheader(":blue[Prediction]")
        
        #st.write(f"The probability of a terrorist attack in {state} state is: {probability:.2%}")


            # Check if the probability exceeds 60%
        if probability > 0.6:
            # Display in red
            st.markdown(f"<span style='color:red;'>The probability of a terrorist attack in {state} is: {probability:.2%}. It is strongly recommeded that the security agency take appropraite precautionary measure to ensure maximum security and mitigation of possible attack.</span>", unsafe_allow_html=True)
        else:
            # Display in default color
            st.write(f"The probability of a terrorist attack in {state} is: {probability:.2%}")
            

# Report Tab
elif selected == "Make Report":
    st.title(":blue[Make Report]")
    st.write("Report any act of terrorism. Terrorism will spill over if you don’t speak up.")


    # Get the user's IP-based location
    g = geocoder.ip('me')

    # Extract latitude and longitude
    latitude = g.latlng[0]
    longitude = g.latlng[1]

    st.write(f"Latitude: {latitude}, Longitude: {longitude}")


    # Reverse geocode to get the location name
    g = geocoder.osm([latitude, longitude], method='reverse')

    # Print the address
    st.write(g.address)


    # Function to connect and create MySQL table
    def create_reports_table():

        # Establish connection to the MySQL database
        connection = mysql.connector.connect(
            host='sql5.freesqldatabase.com',
            user='sql5729527',
            password='T2RVEewsi7',
            database='sql5729527'
        )

        # Create a cursor object using the connection
        cursor = connection.cursor()

        # Create the table with MySQL syntax
        cursor.execute('''CREATE TABLE IF NOT EXISTS reports (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    State VARCHAR(255),
                    city VARCHAR(255),
                    latitude FLOAT,
                    longitude FLOAT,
                    suicide VARCHAR(255),
                    attacktype1_txt VARCHAR(255),
                    target_type VARCHAR(255),
                    target_subtype VARCHAR(255),
                    target VARCHAR(255),
                    group_name VARCHAR(255),
                    weapon_type VARCHAR(255),
                    weapon_subtype VARCHAR(255),
                    no_killed INT,
                    no_wounded INT,
                    full_name VARCHAR(255),
                    mobile_contact VARCHAR(255),
                    email_contact VARCHAR(255),
                    address TEXT
                    )''')

        # Commit the transaction
        connection.commit()

        # Close the connection
        connection.close()


    # Function to add a report to the database
    def add_report(date, state, city, latitude, longitude, suicide,
                    attacktype1_txt, target_type, target_subtype, target, group_name,
                    weapon_type, weapon_subtype, no_killed, no_wounded, full_name,
                    mobile_contact, email_contact, address):

        
        
        # Establish connection to the MySQL database
        conn = mysql.connector.connect(
            host='sql5.freesqldatabase.com',
            user='sql5729527',
            password='T2RVEewsi7',
            database='sql5729527'
        )
        
        # Create a cursor object using the connection
        c = conn.cursor()
        
        # Execute the SQL command to insert data
        c.execute('''INSERT INTO reports (date, State, city, latitude, longitude, suicide,
                                        attacktype1_txt, target_type, target_subtype, target, group_name,
                                        weapon_type, weapon_subtype, no_killed, no_wounded, full_name,
                                        mobile_contact, email_contact, address)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                (date, state, city, latitude, longitude, suicide,
                attacktype1_txt, target_type, target_subtype, target, group_name,
                weapon_type, weapon_subtype, no_killed, no_wounded, full_name,
                mobile_contact, email_contact, address))
        
        # Commit the transaction
        conn.commit()
        
        # Close the connection
        conn.close()


    # Create the table when the app starts
    create_reports_table()

    # Initialize session state to manage form submission
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False

    # Form for making a report
    if not st.session_state.form_submitted:
        with st.form("report_form"):
            st.write("Please fill out the form below to submit a report:")

            date = st.date_input("Date")
            state = st.selectbox("State", options=state_label_encoder.classes_)
            city = st.text_input("City")
            #latitude = st.number_input("Latitude")
            #longitude = st.number_input("Longitude")
            suicide = st.selectbox("Suicide", ["Yes", "No"])
            attacktype1_txt = st.selectbox('Select the type of attack', full_data['attacktype1_txt'].unique())
            target_type = st.selectbox('Select the target type of the attack', full_data['target_type'].unique())
            target_subtype = st.selectbox('Select the target subtype', full_data['target_subtype'].unique())
            target = st.text_input("Target (Enter the name of person or group attacked)")
            group_name = st.selectbox('Select the terrorist group name', full_data['group_name'].unique())
            weapon_type = st.selectbox('Select the type of weopon used for the attack', full_data['weapon_type'].unique())
            weapon_subtype = st.selectbox('Select the weopon subtype', full_data['weapon_subtype'].unique())
            no_killed = st.number_input("Number Killed", min_value=0)
            no_wounded = st.number_input("Number Wounded", min_value=0)
            full_name = st.text_input("Full Name (of user)")
            mobile_contact = st.text_input("Mobile Contact (of user)")
            email_contact = st.text_input("Email Contact (of user)")
            address = st.text_area("Address (of user)")

            # Submit button
            submitted = st.form_submit_button("Submit Report")

            if submitted:
                add_report(date, state, city, latitude, longitude, suicide,
                           attacktype1_txt, target_type, target_subtype, target, group_name,
                           weapon_type, weapon_subtype, no_killed, no_wounded, full_name,
                           mobile_contact, email_contact, address)

                with st.spinner('Processing...'):
                        time.sleep(3)  # Simulate a delay


                # Display the progress bar
                progress_bar = st.progress(0)

                # Simulate a delay with progress updates
                for i in range(100):
                    time.sleep(0.02)  # Adjust the delay to control the speed of the progress bar
                    progress_bar.progress(i + 1)


                st.success(f"Report submitted successfully from the location with the Geographic Coordinates; {latitude}, {longitude}!")
                st.session_state.form_submitted = True

    # Option to submit another report
    if st.session_state.form_submitted:
        st.write("Thank you for your report.")
        if st.button("Submit Another Report"):
            st.session_state.form_submitted = False
            st.experimental_rerun()

# About Tab
elif selected == "About":
    st.subheader(":blue[About]")
    st.write("This study focus on addressing terrorism using machine learning and Geographic Information Systems (GIS). It highlights the significant impact of terrorism on various sectors, particularly in Nigeria, and the need for predictive models to understand and mitigate these threats. Machine learning's capability to analyze historical data and forecast aspects of terrorist attacks—such as weapon types, attack success, and location—supports more effective resource allocation and prevention strategies. The research integrates GIS to capture and analyze spatial data, enhancing the prediction of terrorist hotspots. By combining GIS with machine learning, the study aims to improve counterterrorism efforts, helping security agencies to prevent attacks and devise strategies that promote safety and national peace.")
