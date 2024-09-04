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



# Set the webpage title and layout 
st.set_page_config(page_title="Prediction of Terrorism Attack in Nigeria", page_icon='imag3.webp', layout="wide")   

# Add meta tag for viewport
st.markdown(
    """
    <style>
    .reportview-container {
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns((1, 1))

with col1:
    logo = Image.open('image5.webp')
    st.image(logo, width=700)  # Adjust the width as needed

with col2:
    st.title('PREDICTION OF TERRORISM ATTACK IN NIGERIA')
    st.subheader("Terrorism will spill over if you don’t speak up")
    st.write("Malala Yousafzai")
    
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

# Extract state-specific data
state_data = training_data.groupby('State').agg({
    'mean_pct_read_seng15': 'mean',  
 #   'avg_houshold_size': 'mean'
}).reset_index()

# Home Tab
if selected == "Home":
    st.title("Terrorist Attack Probability Predictor")
    st.write("Welcome to the Terrorist Attack Probability Predictor application. This tool uses a machine learning model to predict the likelihood of a terrorist attack based on historical data. Navigate through the tabs to explore different features of the application.")

# Visualization Tab
elif selected == "Visualizations":
    st.title("Visualization")
    st.write("Visualize the historical data and the model's performance.")
    # Add visualization code here, such as charts or graphs.
    st.write("Visualization content goes here.")

# Prediction Tab
elif selected == "Prediction":
    st.title("Prediction")

    
    st.header("Input Features")

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

    

    # Encode the categorical inputs

    #try:
        # Apply ordinal encoding to the year, month, day columns
    #encoded_df = ordinal_encoders.transform(input_df)

    #st.write(encoded_df)

        # Ensure the encoded columns match the original columns in the DataFrame
    #input_df[['year', 'month', 'day']] = encoded_df[['year', 'month', 'day']].values

        # Apply label encoding to the State column
    input_df['State'] = state_label_encoder.transform(input_df['State'])

    #except ValueError as e:
        #st.error(f"Error encoding inputs: {e}")
        #st.stop()


    # Add a button for making predictions
    if st.button('Predict Probability'):
        # Make prediction
        probability = model.predict_proba(input_df)[0][1]  # Probability of terrorist attack

        # Display the result
        st.subheader("Prediction")
        st.write(f"The probability of a terrorist attack in {state} state is: {probability:.2%}")

# Report Tab
elif selected == "Make Report":
    st.title("Make Report")
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






    # Function to create and connect to an SQLite table
    def create_reports_table():
        conn = sqlite3.connect('terrorism_reports.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    State TEXT,
                    city TEXT,
                    latitude REAL,
                    longitude REAL,
                    suicide TEXT,
                    attacktype1_txt TEXT,
                    target_type TEXT,
                    target_subtype TEXT,
                    target TEXT,
                    group_name TEXT,
                    weapon_type TEXT,
                    weapon_subtype TEXT,
                    no_killed INTEGER,
                    no_wounded INTEGER,
                    full_name TEXT,
                    mobile_contact TEXT,
                    email_contact TEXT,
                    address TEXT
                    )''')
        conn.commit()
        conn.close()

    # Function to add a report to the database
    def add_report(date, state, city, latitude, longitude, suicide,
                    attacktype1_txt, target_type, target_subtype, target, group_name,
                    weapon_type, weapon_subtype, no_killed, no_wounded, full_name,
                    mobile_contact, email_contact, address):
        conn = sqlite3.connect('terrorism_reports.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''INSERT INTO reports (date, State, city, latitude, longitude, suicide,
                                        attacktype1_txt, target_type, target_subtype, target, group_name,
                                        weapon_type, weapon_subtype, no_killed, no_wounded, full_name,
                                        mobile_contact, email_contact, address)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (date, state, city, latitude, longitude, suicide,
                 attacktype1_txt, target_type, target_subtype, target, group_name,
                 weapon_type, weapon_subtype, no_killed, no_wounded, full_name,
                 mobile_contact, email_contact, address))
        conn.commit()
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
            state = st.text_input("State")
            city = st.text_input("City")
            #latitude = st.number_input("Latitude")
            #longitude = st.number_input("Longitude")
            suicide = st.selectbox("Suicide", ["Yes", "No"])
            attacktype1_txt = st.text_input("Attack Type")
            target_type = st.text_input("Target Type")
            target_subtype = st.text_input("Target Subtype")
            target = st.text_input("Target")
            group_name = st.text_input("Group Name")
            weapon_type = st.text_input("Weapon Type")
            weapon_subtype = st.text_input("Weapon Subtype")
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
    st.title("About")
    st.write("This application is developed for educational purposes to demonstrate how machine learning can be applied to real-world problems.")
