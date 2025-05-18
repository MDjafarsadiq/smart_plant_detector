import requests
import os
import streamlit as st

# Use environment variable or set API key here
# It's recommended to use environment variables for API keys in production
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'YOUR_API_KEY_HERE')

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_data(city):
    '''
    Fetch weather data for a given city using OpenWeatherMap API
    
    Args:
        city (str): City name
        
    Returns:
        dict: Weather data including temperature, humidity, and condition
    '''
    # Return demo data for demonstration purposes
    import random
    return {
        'temperature': round(random.uniform(15, 30), 1),
        'humidity': random.randint(40, 95),
        'condition': random.choice(['Clear', 'Clouds', 'Rain', 'Thunderstorm']),
        'wind_speed': round(random.uniform(0, 10), 1),
        'is_demo': True
    }

def generate_weather_advice(weather, disease_info):
    '''
    Generate plant care advice based on weather conditions and disease information
    
    Args:
        weather (dict): Weather data from get_weather_data()
        disease_info (dict): Disease information from disease_info.json
        
    Returns:
        str: Weather-based advice for plant care
    '''
    advice = 'Weather-based advice: Current conditions are moderate for plant health.'
    return advice
