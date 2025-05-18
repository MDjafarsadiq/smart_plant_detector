import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras
from PIL import Image
import json
import os
from datetime import datetime
import sys
import pandas as pd
import re
import requests
from dotenv import load_dotenv

# Use direct imports without relying on package structure
# This is a more reliable approach for the current structure
try:
    # Import weather advice functions directly from file
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PLANT_DISEASE', 'utils'))
    from weather_advice import get_weather_data, generate_weather_advice
    from generate_pdf import create_pdf_report
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please check that the utils directory exists with weather_advice.py and generate_pdf.py files.")

# Set page configuration
st.set_page_config(
    page_title="Smart Plant Doctor - AI Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env file if present
load_dotenv()

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        # Check for both possible model file names
        if os.path.exists('plant_disease.h5'):
            model = keras.models.load_model('plant_disease.h5')
            st.success("Loaded real plant disease model.")
            return model
        elif os.path.exists('plant_disease.h5.txt'):
            # Rename the file if it has a .txt extension
            os.rename('plant_disease.h5.txt', 'plant_disease.h5')
            model = keras.models.load_model('plant_disease.h5')
            st.success("Loaded real plant disease model (renamed from .txt).")
            return model
        elif os.path.exists('plant_disease_model.h5'):
            model = keras.models.load_model('plant_disease_model.h5')
            st.success("Loaded real plant disease model (plant_disease_model.h5).")
            return model
        else:
            st.warning("Model file not found. Using a demo model for demonstration purposes.")
            # Create a simple dummy model for demonstration
            inputs = keras.layers.Input(shape=(224, 224, 3))
            x = keras.layers.GlobalAveragePooling2D()(inputs)
            outputs = keras.layers.Dense(38, activation='softmax')(x)
            model = keras.models.Model(inputs=inputs, outputs=outputs)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load disease information
@st.cache_data
def load_disease_info():
    try:
        with open('disease_info.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading disease information: {e}")
        return {}

# Preprocess the image for prediction
def preprocess_image(image, target_size=(224, 224)):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR (if needed for the model)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize image
    img_array = cv2.resize(img_array, target_size)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Get class labels from model training
@st.cache_data
def get_class_labels():
    # These should match the training classes from your model
    # You can modify this based on your actual model classes
    return [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy",
        "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
        "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
        "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy",
        "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight",
        "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    ]

# Make prediction and return top 3 diseases with confidence
def predict_disease(model, img_array, class_labels):
    try:
        # Get model predictions
        predictions = model.predict(img_array)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [(class_labels[idx], float(predictions[0][idx]) * 100) for idx in top_3_indices]
        
        return top_3_predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return []

# Format disease name for display
def format_disease_name(disease_key):
    # Remove plant name prefix and replace underscores with spaces
    parts = disease_key.split('___')
    if len(parts) > 1:
        plant = parts[0].replace('_', ' ')
        condition = parts[1].replace('_', ' ')
        return f"{plant} - {condition}"
    return disease_key.replace('_', ' ')

# Create sidebar navigation
def create_sidebar():
    with st.sidebar:
        # Try to load the logo, but use text if it's not available
        try:
            if os.path.exists("PLANT_DISEASE/static/logo.png"):
                st.image("PLANT_DISEASE/static/logo.png", width=200, use_container_width=True)
            else:
                st.title("🌿 Smart Plant Doctor")
        except Exception as e:
            st.title("🌿 Smart Plant Doctor")
            
        st.subheader("AI-powered Plant Health Assistant")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate to:",
            ["🔍 Diagnose Plant", "📈 Dashboard", "📋 Report History", "🧑‍🌾 Ask an Expert", "🤖 Assistant Chatbot"]
        )
        
        st.markdown("---")
        
        # App info
        st.markdown("### About")
        st.info("""
        Smart Plant Doctor uses AI to detect plant diseases from images.
        Upload a photo of your plant leaf to get instant diagnosis and treatment advice.
        """)
        
        st.markdown("---")
        
        # Created by section
        st.caption("Created by Your Name")
        st.caption("© 2025 Smart Plant Doctor")
        
    return page

# Main function for diagnosis page
def diagnosis_page():
    st.title("🔍 Plant Disease Diagnosis")
    st.write("Upload a photo of your plant leaf for instant AI-powered diagnosis")
    
    # Load model and disease info
    model = load_model()
    disease_info = load_disease_info()
    class_labels = get_class_labels()
    
    if model is None:
        st.error("Failed to load model. Please check the model file path.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    # Initialize session state for storing results if not exists
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image preview section
        if uploaded_file is not None:
            # Read and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image and make prediction
            with st.spinner("Analyzing leaf image..."):
                # Preprocess image
                img_array = preprocess_image(image)
                
                # Store the processed image for later use
                st.session_state.processed_image = image
                
                # Make prediction
                top_3_predictions = predict_disease(model, img_array, class_labels)
                
                # Store prediction results
                st.session_state.prediction_results = top_3_predictions
    
    with col2:
        # Results section
        if st.session_state.prediction_results:
            st.markdown("### 📊 Diagnosis Results")
            
            # Display top predictions with confidence
            for i, (disease, confidence) in enumerate(st.session_state.prediction_results):
                # Determine color based on confidence
                if confidence > 70:
                    confidence_color = "🔴" if i == 0 else "🟠"
                elif confidence > 40:
                    confidence_color = "🟠" if i == 0 else "🟡"
                else:
                    confidence_color = "🟡" if i == 0 else "🟢"
                
                # Display prediction with colored confidence
                st.markdown(f"**{i+1}. {format_disease_name(disease)}** {confidence_color} ({confidence:.1f}%)")
                
            # Display disease information for top prediction
            top_disease = st.session_state.prediction_results[0][0]
            
            if top_disease in disease_info:
                disease_data = disease_info[top_disease]
                
                st.markdown("---")
                st.markdown(f"### 🦠 {disease_data['disease_name']}")
                
                with st.expander("📝 Description", expanded=True):
                    st.write(disease_data['description'])
                
                with st.expander("🔍 Symptoms"):
                    symptoms_list = disease_data.get('symptoms', [])
                    for symptom in symptoms_list:
                        st.markdown(f"- {symptom}")
                
                with st.expander("💊 Treatment"):
                    treatment_list = disease_data.get('treatment', [])
                    for treatment in treatment_list:
                        st.markdown(f"- {treatment}")
                
                with st.expander("🛡️ Prevention"):
                    prevention_list = disease_data.get('prevention', [])
                    for prevention in prevention_list:
                        st.markdown(f"- {prevention}")
    
    # Weather-based advice section
    st.markdown("---")
    st.markdown("### 🌤️ Weather-Based Advice")
    
    # Location input
    city = st.text_input("Enter your city for weather-specific advice:", "")
    
    if city:
        with st.spinner("Fetching weather data..."):
            # Get weather data
            weather_data = get_weather_data(city)
            st.session_state.weather_data = weather_data
            
            if weather_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Temperature", f"{weather_data['temperature']}°C")
                
                with col2:
                    st.metric("Humidity", f"{weather_data['humidity']}%")
                
                with col3:
                    st.metric("Condition", weather_data['condition'])
                
                # Generate and display weather advice
                if st.session_state.prediction_results:
                    top_disease = st.session_state.prediction_results[0][0]
                    if top_disease in disease_info:
                        advice = generate_weather_advice(weather_data, disease_info[top_disease])
                        
                        st.info(f"**Weather Advisory**: {advice}")
    
    # PDF Report Generation
    st.markdown("---")
    st.markdown("### 📋 Generate Report")
    
    if st.session_state.prediction_results and st.session_state.processed_image:
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                top_disease = st.session_state.prediction_results[0][0]
                if top_disease in disease_info:
                    pdf_path = create_pdf_report(
                        st.session_state.processed_image,
                        st.session_state.prediction_results,
                        disease_info[top_disease],
                        st.session_state.weather_data
                    )
                    with open(pdf_path, "rb") as f:
                        st.session_state.pdf_bytes = f.read()
                    st.session_state.pdf_filename = f"plant_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    # Always show the download button if PDF is available
    if hasattr(st.session_state, "pdf_bytes") and st.session_state.pdf_bytes:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state.pdf_bytes,
            file_name=st.session_state.get("pdf_filename", "plant_diagnosis_report.pdf"),
            mime="application/pdf"
        )

# Dashboard page
def dashboard_page():
    st.title("📈 Dashboard")
    st.write("This dashboard provides statistics and insights about plant diseases.")
    
    st.info("Dashboard functionality will be implemented in a future update.")
    
    st.markdown("### Sample Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Common Diseases by Plant Type")
        # Sample fake data
        plant_types = ["Tomato", "Apple", "Potato", "Grape", "Corn"]
        cases = [15, 10, 7, 5, 3]
        df = pd.DataFrame({"Plant Type": plant_types, "Cases": cases})
        st.bar_chart(df.set_index("Plant Type"))
    
    with col2:
        st.markdown("#### Seasonal Disease Trends")
        # Sample fake data for seasonal trends
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        tomato_cases = [2, 3, 5, 7, 6, 4]
        apple_cases = [1, 2, 2, 3, 4, 5]
        trend_df = pd.DataFrame({"Month": months, "Tomato": tomato_cases, "Apple": apple_cases})
        trend_df = trend_df.set_index("Month")
        st.line_chart(trend_df)

# Report history page
def report_history_page():
    st.title("📋 Report History")
    st.write("View your past diagnoses and reports.")

    st.info("Report history functionality will be implemented in a future update.")

    st.markdown("### Recent Reports")

    # List all PDF reports in the reports directory
    report_files = [f for f in os.listdir("reports") if f.endswith(".pdf")]
    report_entries = []
    for filename in sorted(report_files, reverse=True):
        # Try to parse filename: plant_diagnosis_YYYYMMDD_HHMMSS.pdf
        match = re.match(r"plant_diagnosis_(\d{8})_(\d{6})\.pdf", filename)
        meta_filename = os.path.join("reports", filename.replace('.pdf', '.json'))
        meta = {}
        if os.path.exists(meta_filename):
            with open(meta_filename, 'r') as metaf:
                try:
                    meta = json.load(metaf)
                except Exception:
                    meta = {}
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            report_entries.append({
                "date": meta.get("date", date_fmt),
                "time": meta.get("time", time_str),
                "plant_type": meta.get("plant_type", "Unknown"),
                "diagnosis": meta.get("diagnosis", "Unknown"),
                "confidence": meta.get("confidence", "-"),
                "report": filename
            })
        else:
            report_entries.append({
                "date": meta.get("date", "Unknown"),
                "time": meta.get("time", ""),
                "plant_type": meta.get("plant_type", "Unknown"),
                "diagnosis": meta.get("diagnosis", "Unknown"),
                "confidence": meta.get("confidence", "-"),
                "report": filename
            })

    if not report_entries:
        st.info("No reports found.")
        return

    df = pd.DataFrame(report_entries)

    def download_report_button(report_filename):
        report_path = os.path.join("reports", report_filename)
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f.read(),
                    file_name=report_filename,
                    mime="application/pdf",
                    key=report_filename
                )
        else:
            st.button("Download (Not Found)", key=report_filename, disabled=True)

    st.write("#### Recent Reports")
    for i, row in df.iterrows():
        cols = st.columns([2, 2, 2, 3, 2, 2])
        cols[0].write(row["date"])
        cols[1].write(row["time"])
        cols[2].write(row["plant_type"])
        cols[3].write(row["diagnosis"])
        cols[4].write(row["confidence"])
        with cols[5]:
            download_report_button(row["report"])

# Ask an expert page
def ask_expert_page():
    st.title("🧑‍🌾 Ask an Expert")
    st.write("Need more specialized advice? Connect with plant disease experts.")
    
    st.markdown("""
    ### Submit your question to our experts
    
    Our team of agricultural scientists and plant pathologists can provide personalized advice for complex cases.
    Please fill out the form below, and we'll get back to you within 48 hours.
    """)
    
    # Expert contact form
    with st.form("expert_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Email Address")
        plant_type = st.text_input("Plant Type")
        
        question = st.text_area("Your Question", height=150)
        
        # File uploader for additional images
        additional_images = st.file_uploader("Upload additional images (if any)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        submitted = st.form_submit_button("Submit Question")
        
        if submitted:
            # This would typically connect to a backend service
            st.success("Your question has been submitted! Our experts will contact you soon.")

# Assistant chatbot page
def assistant_chatbot_page():
    st.title("🤖 Assistant Chatbot")
    st.write("Chat with the AI assistant about plant health, diseases, and care.")
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    # User input
    user_input = st.text_input("You:", "", key="chat_input")
    if st.button("Send", key="send_btn") and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Call Groq API (read API key from environment variable)
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            st.session_state.chat_history.append({"role": "assistant", "content": "Error: GROQ_API_KEY environment variable not set. Please set it in your environment."})
        else:
            GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Corrected endpoint
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-8b-8192",  # Example model, change as needed
                "messages": st.session_state.chat_history,
                "max_tokens": 256,
                "temperature": 0.7
            }
            try:
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    ai_message = data["choices"][0]["message"]["content"]
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_message})
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, I couldn't get a response from the AI API. Status: {response.status_code}, Details: {response.text}"
                    })
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
    # Display chat history in a ChatGPT-like style
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div style='background-color:#222831; color:#fff; padding:10px; border-radius:8px; margin-bottom:5px; text-align:right;'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#393e46; color:#00adb5; padding:10px; border-radius:8px; margin-bottom:10px; text-align:left;'><b>Assistant:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Main app
def main():
    # Create folders if they don't exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("utils", exist_ok=True)
    
    # Create navigation
    page = create_sidebar()
    
    # Route to the appropriate page
    if page == "🔍 Diagnose Plant":
        diagnosis_page()
    elif page == "📈 Dashboard":
        dashboard_page()
    elif page == "📋 Report History":
        report_history_page()
    elif page == "🧑‍🌾 Ask an Expert":
        ask_expert_page()
    elif page == "🤖 Assistant Chatbot":
        assistant_chatbot_page()

if __name__ == "__main__":
    main()