import os
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import numpy as np
import tempfile
import json

def create_pdf_report(image, predictions, disease_info, weather_data=None):
    """
    Create a PDF report with diagnosis results
    
    Args:
        image (PIL.Image): Uploaded plant image
        predictions (list): List of (disease_name, confidence) tuples
        disease_info (dict): Disease information for the top prediction
        weather_data (dict, optional): Weather data for the user's location
        
    Returns:
        str: Path to the generated PDF file
    """
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/plant_diagnosis_{timestamp}.pdf"
    # Save metadata for report history
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "plant_type": disease_info.get("plant", "Unknown"),
        "diagnosis": disease_info.get("disease_name", "Unknown"),
        "confidence": f"{predictions[0][1]:.1f}%",
        "report": filename
    }
    meta_filename = filename.replace('.pdf', '.json')
    with open(meta_filename, 'w') as metaf:
        json.dump(meta, metaf)
    
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter, 
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]
    
    # Create custom style for descriptions
    desc_style = ParagraphStyle(
        'Description',
        parent=normal_style,
        spaceAfter=10,
        leading=14,
    )
    
    # Create custom style for headers
    header_style = ParagraphStyle(
        'Header',
        parent=heading_style,
        textColor=colors.darkgreen,
        fontSize=14,
        spaceAfter=10,
        spaceBefore=20,
    )
    
    # Create content list
    elements = []
    
    # Add title
    elements.append(Paragraph("Smart Plant Doctor - Diagnosis Report", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add date and time
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add the image
    img_width = 4*inch
    img_temp = BytesIO()
    
    # Resize and save image
    img_copy = image.copy()
    img_copy.thumbnail((img_width, img_width * img_copy.height / img_copy.width))
    img_copy.save(img_temp, format='JPEG')
    img_temp.seek(0)
    
    # Add image to PDF
    img = RLImage(img_temp, width=img_width, height=img_width * image.height / image.width)
    elements.append(img)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add diagnosis results header
    elements.append(Paragraph("Diagnosis Results", header_style))
    
    # Format the top predictions into a table
    data = [["Disease", "Confidence"]]
    for disease_key, confidence in predictions:
        # Format disease name
        parts = disease_key.split('___')
        if len(parts) > 1:
            plant = parts[0].replace('_', ' ')
            condition = parts[1].replace('_', ' ')
            disease_name = f"{plant} - {condition}"
        else:
            disease_name = disease_key.replace('_', ' ')
            
        data.append([disease_name, f"{confidence:.1f}%"])
    
    # Create the table
    table = Table(data, colWidths=[4*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkgreen),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add disease information
    elements.append(Paragraph("Disease Information", header_style))
    
    # Add description
    elements.append(Paragraph("Description:", styles["Heading4"]))
    elements.append(Paragraph(disease_info.get("description", "No description available."), desc_style))
    
    # Add symptoms
    elements.append(Paragraph("Symptoms:", styles["Heading4"]))
    symptoms = disease_info.get("symptoms", ["No symptoms information available."])
    for symptom in symptoms:
        elements.append(Paragraph(f"• {symptom}", desc_style))
    
    # Add treatment
    elements.append(Paragraph("Treatment:", styles["Heading4"]))
    treatments = disease_info.get("treatment", ["No treatment information available."])
    for treatment in treatments:
        elements.append(Paragraph(f"• {treatment}", desc_style))
    
    # Add prevention
    elements.append(Paragraph("Prevention:", styles["Heading4"]))
    preventions = disease_info.get("prevention", ["No prevention information available."])
    for prevention in preventions:
        elements.append(Paragraph(f"• {prevention}", desc_style))
    
    # Add weather information if available
    if weather_data:
        elements.append(Paragraph("Weather Information & Advice", header_style))
        
        # Create weather data table
        weather_data_table = [
            ["Temperature", "Humidity", "Condition"],
            [f"{weather_data.get('temperature', 'N/A')}°C", 
             f"{weather_data.get('humidity', 'N/A')}%", 
             weather_data.get('condition', 'N/A')]
        ]
        
        weather_table = Table(weather_data_table, colWidths=[2*inch, 2*inch, 2*inch])
        weather_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(weather_table)
        elements.append(Spacer(1, 0.15*inch))
        
        # Add weather-based advice
        import weather_advice
        advice = weather_advice.generate_weather_advice(weather_data, disease_info)
        elements.append(Paragraph("Weather-Based Advice:", styles["Heading4"]))
        elements.append(Paragraph(advice, desc_style))
    
    # Add footer
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("© 2025 Smart Plant Doctor - AI-powered Plant Health Assistant", 
                             ParagraphStyle(
                                 'Footer',
                                 parent=normal_style,
                                 fontSize=8,
                                 textColor=colors.grey,
                             )))
    
    # Build the PDF
    doc.build(elements)
    
    return filename