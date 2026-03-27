import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Configuration ---
st.set_page_config(page_title="Food Bill Predictor", layout="centered")

# --- Custom CSS for Background ---
def add_bg_and_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                        url("https://images.unsplash.com/photo-1504674900247-0877df9cc836?q=80&w=2070");
            background-size: cover;
            background-attachment: fixed;
        }
        .main .block-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
        }
        h1, h2, h3, p, label { color: #ffffff !important; }
        .stButton>button {
            background-color: #e67e22 !important;
            color: white !important;
            border: none;
            width: 100%;
            font-weight: bold;
            height: 3em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_and_style()

# --- Load Model Artifacts ---
@st.cache_resource
def load_assets():
    file_path = 'model_data.pkl'
    if not os.path.exists(file_path):
        st.error(f"Error: {file_path} not found in repository root!")
        return None
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

assets = load_assets()

if assets:
    model = assets['model']
    train_cols = assets['columns']
    train_mean = assets['mean']
    train_std = assets['std']
    options = assets['categorical_options']

    # --- UI ---
    st.title("🍽️ Food Bill Predictor")
    st.write("Predict your order total with machine learning precision.")

    col1, col2 = st.columns(2)

    with col1:
        cuisine = st.selectbox("Cuisine Type", options['cuisine_type'])
        meal = st.selectbox("Meal Time", options['meal_time'])
        items = st.number_input("Number of Items", 1, 20, 3)
        price = st.number_input("Avg Item Price (₹)", value=200.0)

    with col2:
        gender = st.selectbox("Customer Gender", options['customer_gender'])
        weekend = st.selectbox("Weekend Order?", options['weekend'])
        discount = st.slider("Discount %", 0, 30, 5)
        dist = st.slider("Distance (km)", 0.0, 20.0, 3.0)

    with st.expander("Advanced Customer Parameters"):
        rate = st.slider("Delivery Rating", 1.0, 5.0, 4.0)
        age = st.number_input("Customer Age", 18, 100, 30)
        prev = st.number_input("Previous Orders", 0, 100, 5)

    if st.button("Calculate Prediction"):
        # 1. Prepare Input
        input_df = pd.DataFrame({
            'num_items': [items], 'avg_item_price': [price], 'discount_percent': [discount],
            'delivery_distance_km': [dist], 'delivery_rating': [rate],
            'customer_age': [age], 'num_previous_orders': [prev],
            'cuisine_type': [cuisine], 'meal_time': [meal],
            'customer_gender': [gender], 'weekend': [weekend]
        })
        
        # 2. Manual Encoding
        input_encoded = pd.get_dummies(input_df)
        final_features = pd.DataFrame(0, index=[0], columns=train_cols)
        for col in input_encoded.columns:
            if col in final_features.columns:
                final_features[col] = input_encoded[col]
                
        # 3. Manual Scaling & Prediction
        scaled_features = (final_features - train_mean) / train_std
        prediction = model.predict(scaled_features)[0]
        
        st.markdown(f"""
            <div style="background-color: rgba(46, 204, 113, 0.2); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #2ecc71; margin-top: 20px;">
                <h2 style="margin:0; color: #2ecc71;">Estimated Total Bill</h2>
                <h1 style="margin:0; color: white;">₹{max(0, prediction):,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)