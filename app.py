import streamlit as st
import pandas as pd
import pickle

# Load the trained model and label encoder
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)
    return model, le

rf_model, le = load_model()

# Define the skill mapping
skill_map = {
    'Not Interested': 0, 'Poor': 1, 'Beginner': 2,
    'Average': 3, 'Intermediate': 4, 'Professional': 5, 'Excellent': 5
}

# Skill categories used in the model (based on X_train columns)
# You might want to retrieve these dynamically or hardcode if they are fixed.
model_columns = [
    'Database Fundamentals', 'Computer Architecture', 'Distributed Computing Systems',
    'Cyber Security', 'Networking', 'Software Development', 'Programming Skills',
    'Project Management', 'Computer Forensics Fundamentals', 'AI ML', 'Software Engineering',
    'Business Analysis', 'Data Science', 'Troubleshooting skills', 'Graphics Designing'
]

# Streamlit app title
st.title('Career Path Predictor')
st.write('Enter your skill levels to get a career recommendation.')

user_skill_inputs = {}

# Create input widgets for each skill
for column in model_columns:
    # Create a selectbox for each skill with predefined options
    selected_level = st.selectbox(
        f'Your level for {column}:',
        list(skill_map.keys()), # Options from skill_map keys
        index=2 # Default to 'Beginner'
    )
    user_skill_inputs[column] = skill_map[selected_level]

if st.button('Predict Career'):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_skill_inputs])

    # Ensure the order of columns matches the training data
    input_df = input_df[model_columns]

    # Make prediction
    predicted_numeric_id = rf_model.predict(input_df)[0]

    # Decode the prediction
    predicted_role = le.inverse_transform([predicted_numeric_id])[0]

    st.success(f'### Recommended Career Path: {predicted_role}')
