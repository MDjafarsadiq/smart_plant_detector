# Smart Plant Doctor 🌿

An AI-powered Plant Disease Detection and Plant Health Assistant

## Project Overview

Smart Plant Doctor is a Streamlit web application that uses deep learning to detect plant diseases from leaf images. The application provides detailed information about detected diseases, treatment recommendations, and weather-based advice for plant care.

## Features

- **AI-powered Plant Disease Detection**: Upload a photo of your plant leaf and get instant diagnosis
- **Detailed Disease Information**: Get comprehensive information about detected diseases including symptoms, treatment, and prevention tips
- **Weather-Based Advice**: Receive customized plant care recommendations based on your local weather conditions
- **PDF Report Generation**: Generate and download detailed diagnosis reports
- **User-Friendly Interface**: Clean, modern, and intuitive UI design

## Project Structure

```
smart_plant_doctor/
├── app.py                 # Main application file
├── plant_disease_model.h5 # Trained CNN model for disease detection
├── disease_info.json      # Database of plant disease information
├── requirements.txt       # Python dependencies
├── utils/                 # Utility modules
│   ├── generate_pdf.py    # PDF report generation functionality
│   ├── weather_advice.py  # Weather data fetching and advice generation
├── static/                # Static files (images, logos, etc.)
│   ├── logo.png           # Application logo
├── reports/               # Generated PDF reports (created at runtime)
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/smart-plant-doctor.git
cd smart-plant-doctor
```

### Step 2: Create and activate a virtual environment (optional but recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up the model

Ensure that your trained model `plant_disease_model.h5` is in the project root directory.

### Step 5: Set up OpenWeatherMap API key (optional)

For weather-based advice functionality, you can set up an OpenWeatherMap API key:

1. Sign up at [OpenWeatherMap](https://openweathermap.org/) and get an API key
2. Set it as an environment variable:

```bash
# On Windows
set OPENWEATHER_API_KEY=your_api_key_here

# On macOS/Linux
export OPENWEATHER_API_KEY=your_api_key_here
```

Or add it directly in the `utils/weather_advice.py` file (not recommended for production).

## Running the Application

To start the Streamlit application, run:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Model Training Information

The plant disease detection model is a Convolutional Neural Network (CNN) trained on a dataset of plant leaf images with various diseases. The model is capable of identifying 38 different classes of plant diseases and healthy plants.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used for training is based on the [Plant Village Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Special thanks to all contributors who have helped improve this project

---

Created by [Your Name] - © 2025 Smart Plant Doctor