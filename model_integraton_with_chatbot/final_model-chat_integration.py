import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session
from flask_sqlalchemy import SQLAlchemy
import logging
logging.basicConfig(level=logging.DEBUG)
import joblib
from datetime import datetime
import json

# Set up API key for Gemini
API_KEY = "rahul_gopalan"#replace with valid key
genai.configure(api_key=API_KEY)

# Initialize Flask and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agriculture_assistant.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'agriculture_ai_assistant_secret_key'
db = SQLAlchemy(app)

# Set up model paths
MODEL_DIR = os.path.join(os.getcwd(), 'models')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database models
class UserQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    model_used = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserQuery {self.id}>"

# Model loading utilities
def load_models():
    """Load all saved models"""
    models = {}
    
    # 1. Plant Disease Prediction Model
    try:
        models['plant_disease'] = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.h5'))
        print("Plant disease model loaded successfully")
    except Exception as e:
        print(f"Error loading plant disease model: {e}")
    
    # 2. Weed Detection Model
    try:
        models['weed_detection'] = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'weed_detection_model.h5'))
        print("Weed detection model loaded successfully")
    except Exception as e:
        print(f"Error loading weed detection model: {e}")
    
    # 3. Price Prediction Models
    try:
        with open(os.path.join(MODEL_DIR, 'price_prediction_model.pkl'), 'rb') as f:
            models['price_prediction'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'price_range_model.pkl'), 'rb') as f:
            models['price_range'] = pickle.load(f)
        print("Price prediction models loaded successfully")
    except Exception as e:
        print(f"Error loading price prediction models: {e}")
    
    # 4. Crop Recommendation Model
    try:
        with open(os.path.join(MODEL_DIR, 'crop_recommendation_model.pkl'), 'rb') as f:
            models['crop_recommendation'] = pickle.load(f)
        print("Crop recommendation model loaded successfully")
    except Exception as e:
        print(f"Error loading crop recommendation model: {e}")
    
    # 5. Smart Irrigation Model
    try:
        # Load Irrigation Prediction Models
        models['irrigation_prediction'] = joblib.load('models/smart_irrigation_model.pkl')
        
        logging.debug("Irrigation prediction models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading irrigation prediction models: {e}")
    
    return models
    
    # 6. Plant Health Models
    try:
        with open(os.path.join(MODEL_DIR, 'plant_health_model.pkl'), 'rb') as f:
            models['plant_health'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'plant_health_scaler.pkl'), 'rb') as f:
            models['plant_health_scaler'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'plant_health_label_encoder.pkl'), 'rb') as f:
            models['plant_health_label_encoder'] = pickle.load(f)
        print("Plant health models loaded successfully")
    except Exception as e:
        print(f"Error loading plant health models: {e}")
    
    return models

# Load all models at startup
models = load_models()


def predict_from_numpy(numpy_file_path):
    """Predict plant disease directly from a saved NumPy array file"""
    if 'plant_disease' not in models:
        return {"error": "Plant disease model not loaded"}
    
    try:
        # Load the preprocessed numpy array
        img_array = np.load(numpy_file_path)
        
        # Add batch dimension if needed
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions directly without any preprocessing
        predictions = models['plant_disease'].predict(img_array)
        
        # Add comprehensive debugging information
        print(f"NumPy file: {numpy_file_path}")
        print(f"Image shape: {img_array.shape}")
        print(f"Image value range: min={np.min(img_array)}, max={np.max(img_array)}")
        print(f"Plant disease raw prediction shape: {predictions.shape}")
        top_10_indices = np.argsort(predictions[0])[-10:][::-1]
        print(f"Top 10 prediction indices: {top_10_indices}")
        print(f"Top 10 prediction values: {[predictions[0][i] for i in top_10_indices]}")
        
        # The model's output maps directly to the classes in this order
        class_indices = {
            'Apple Apple scab': 0, 
            'Apple Black rot': 1, 
            'Apple Cedar apple rust': 2, 
            'Apple healthy': 3, 
            'Bacterial leaf blight in rice leaf': 4, 
            'Blight in corn Leaf': 5, 
            'Blueberry healthy': 6, 
            'Brown spot in rice leaf': 7, 
            'Cercospora leaf spot': 8, 
            'Cherry (including sour) Powdery mildew': 9, 
            'Cherry (including_sour) healthy': 10, 
            'Common Rust in corn Leaf': 11, 
            'Corn (maize) healthy': 12, 
            'Garlic': 13, 
            'Grape Black rot': 14, 
            'Grape Esca Black Measles': 15, 
            'Grape Leaf blight Isariopsis Leaf Spot': 16, 
            'Grape healthy': 17, 
            'Gray Leaf Spot in corn Leaf': 18, 
            'Leaf smut in rice leaf': 19, 
            'Nitrogen deficiency in plant': 20, 
            'Orange Haunglongbing Citrus greening': 21, 
            'Peach healthy': 22, 
            'Pepper bell Bacterial spot': 23, 
            'Pepper bell healthy': 24, 
            'Potato Early blight': 25, 
            'Potato Late blight': 26, 
            'Potato healthy': 27, 
            'Raspberry healthy': 28, 
            'Sogatella rice': 29, 
            'Soybean healthy': 30, 
            'Strawberry Leaf scorch': 31, 
            'Strawberry healthy': 32, 
            'Tomato Bacterial spot': 33, 
            'Tomato Early blight': 34, 
            'Tomato Late blight': 35, 
            'Tomato Leaf Mold': 36, 
            'Tomato Septoria leaf spot': 37, 
            'Tomato Spider mites Two spotted spider mite': 38, 
            'Tomato Target Spot': 39, 
            'Tomato Tomato mosaic virus': 40, 
            'Tomato healthy': 41, 
            'Waterlogging in plant': 42, 
            'algal leaf in tea': 43, 
            'anthracnose in tea': 44, 
            'bird eye spot in tea': 45, 
            'brown blight in tea': 46, 
            'cabbage looper': 47, 
            'corn crop': 48, 
            'ginger': 49, 
            'healthy tea leaf': 50, 
            'lemon canker': 51, 
            'onion': 52, 
            'potassium deficiency in plant': 53, 
            'potato crop': 54, 
            'potato hollow heart': 55, 
            'red leaf spot in tea': 56, 
            'tomato canker': 57
        }
        
        # Create a reverse mapping from index to class name
        index_to_class = {v: k for k, v in class_indices.items()}
        
        # Get the predicted class index and name
        predicted_class = np.argmax(predictions[0])
        if predicted_class not in index_to_class:
            return {"error": f"Prediction index {predicted_class} out of range"}
            
        predicted_class_name = index_to_class[predicted_class]
        confidence = float(predictions[0][predicted_class])
        
        # Get top 3 predictions for alternatives
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        alternatives = [index_to_class[i] for i in top_3_indices[1:]] if len(top_3_indices) > 1 else []
        
        return {
            "prediction": predicted_class_name,
            "confidence": confidence,
            "predicted_index": int(predicted_class),
            "alternatives": alternatives,
            "all_probabilities": {index_to_class[i]: float(predictions[0][i]) for i in range(len(predictions[0]))}
        }
    except Exception as e:
        print(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}
# Image processing utilities
def process_image(image_path, target_size=(512, 512)):
    """Process an image for model input"""
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Model prediction functions
def predict_plant_disease(image_path):
    """Predict plant disease from image"""
    if 'plant_disease' not in models:
        return {"error": "Plant disease model not loaded"}
    
    img_array = process_image(image_path)
    if img_array is None:
        return {"error": "Failed to process image"}
    
    # Placeholder for classes - replace with your actual classes
    classes = ["healthy", "bacterial_leaf_blight", "brown_spot", "leaf_smut"]
    
    predictions = models['plant_disease'].predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    return {
        "prediction": classes[predicted_class],
        "confidence": confidence,
        "all_probabilities": {cls: float(prob) for cls, prob in zip(classes, predictions[0])}
    }

def predict_weed_type(image_path):
    """Predict weed type from image"""
    if 'weed_detection' not in models:
        return {"error": "Weed detection model not loaded"}
    
    img_array = process_image(image_path)
    if img_array is None:
        return {"error": "Failed to process image"}
    
    # Update with the actual weed classes from your model
    weed_classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
                   'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
                   'ShepherdтАЩs Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    
    predictions = models['weed_detection'].predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Safety check to prevent index out of range
    if predicted_class >= len(weed_classes):
        return {"error": f"Prediction index {predicted_class} out of range for classes list"}
    
    confidence = float(predictions[0][predicted_class])
    
    return {
        "prediction": weed_classes[predicted_class],
        "confidence": confidence,
        "all_probabilities": {cls: float(prob) for cls, prob in zip(weed_classes, predictions[0])}
    }

def predict_price(state, apmc, commodity, commodity_arrivals, commodity_traded, year, month, day):
    """Predict commodity price and price range"""
    if 'price_prediction' not in models or 'price_range' not in models:
        return {"error": "Price prediction models not loaded"}
    
    input_data = np.array([[state, apmc, commodity, commodity_arrivals, commodity_traded, year, month, day]])
    
    predicted_price = float(models['price_prediction'].predict(input_data)[0])
    predicted_range = int(models['price_range'].predict(input_data)[0])
    
    # Map price range category to meaningful descriptions
    price_range_mapping = {
        0: "Low Price Range",
        1: "Medium-Low Price Range",
        2: "Medium Price Range",
        3: "Medium-High Price Range",
        4: "High Price Range"
    }
    
    range_description = price_range_mapping.get(predicted_range, "Unknown Price Range")
    
    return {
        "predicted_price": predicted_price,
        "price_range_category": predicted_range,
        "price_range_description": range_description
    }

def recommend_crop(soil_climate_data):
    """Recommend best crop based on soil and climate data"""
    if 'crop_recommendation' not in models:
        return {"error": "Crop recommendation model not loaded"}
    
    try:
        # Convert dictionary to DataFrame with expected features
        df = pd.DataFrame([soil_climate_data])
        
        # Check if model is correctly loaded (should be a scikit-learn model)
        if not hasattr(models['crop_recommendation'], 'predict'):
            return {"error": "Crop recommendation model is not properly loaded or not a valid model"}
        
        # Make prediction
        predicted_crop = models['crop_recommendation'].predict(df)[0]
        
        # Get prediction probabilities for all crops
        crop_probabilities = models['crop_recommendation'].predict_proba(df)[0]
        
        # Get class names (crops)
        crop_classes = models['crop_recommendation'].classes_
        
        # Create probability dictionary
        probability_dict = {crop: float(prob) for crop, prob in zip(crop_classes, crop_probabilities)}
        
        # Sort by probability to get top alternatives
        sorted_probs = sorted(probability_dict.items(), key=lambda x: x[1], reverse=True)
        alternatives = [crop for crop, _ in sorted_probs[1:4]] if len(sorted_probs) > 1 else []
        
        return {
            "recommended_crop": predicted_crop,
            "confidence": float(probability_dict[predicted_crop]),
            "alternative_crops": alternatives,
            "all_probabilities": probability_dict
        }
    except Exception as e:
        return {"error": f"Error in crop recommendation: {str(e)}"}

def predict_irrigation(irrigation_data):
    """Predict optimal irrigation based on soil and climate data"""
    if 'smart_irrigation' not in models:
        return {"error": "Smart irrigation model not loaded"}
    
    try:
        # Convert dictionary to DataFrame with expected features
        df = pd.DataFrame([irrigation_data])
        
        # Make predictions
        predictions = models['smart_irrigation'].predict(df)[0]
        
        # For demonstration, assuming the model outputs three values
        optimal_irrigation = float(predictions[0])
        water_deficiency_risk = float(predictions[1])
        water_efficiency_score = float(predictions[2])
        
        # Risk level categorization
        risk_level = "Low"
        if water_deficiency_risk > 0.6:
            risk_level = "High"
        elif water_deficiency_risk > 0.3:
            risk_level = "Medium"
        
        return {
            "optimal_irrigation_amount": optimal_irrigation,
            "water_deficiency_risk": water_deficiency_risk,
            "water_efficiency_score": water_efficiency_score,
            "risk_level": risk_level,
            "recommendations": generate_irrigation_recommendations(optimal_irrigation, water_deficiency_risk, water_efficiency_score, irrigation_data)
        }
    except Exception as e:
        return {"error": f"Error in irrigation prediction: {str(e)}"}

def generate_irrigation_recommendations(irrigation, risk, efficiency, data):
    """Generate irrigation recommendations based on predictions"""
    recommendations = []
    
    if risk > 0.5:
        recommendations.append("Consider increasing irrigation frequency to reduce water deficiency risk.")
    
    if efficiency < 0.7:
        recommendations.append("Review your irrigation system for potential water loss or inefficiencies.")
    
    if data.get('soil_type') == 'sandy':
        recommendations.append("Sandy soil requires more frequent but lighter irrigation.")
    elif data.get('soil_type') == 'clay':
        recommendations.append("Clay soil holds water longer - avoid overwatering.")
    
    if data.get('temperature', 0) > 30:
        recommendations.append("High temperatures increase evaporation - adjust irrigation timing to early morning or evening.")
    
    return recommendations

def predict_plant_health(health_data):
    """Predict plant health status based on input data"""
    if not all(model in models for model in ['plant_health', 'plant_health_scaler', 'plant_health_label_encoder']):
        return {"error": "Plant health models not loaded"}
    
    try:
        # Extract features from the input dictionary
        features = np.array([[
            health_data.get('soil_moisture', 0),
            health_data.get('ambient_temperature', 0),
            health_data.get('soil_temperature', 0),
            health_data.get('humidity', 0),
            health_data.get('light_intensity', 0),
            health_data.get('soil_ph', 0),
            health_data.get('nitrogen_level', 0),
            health_data.get('phosphorus_level', 0),
            health_data.get('potassium_level', 0),
            health_data.get('chlorophyll_content', 0),
            health_data.get('electrochemical_signal', 0)
        ]])
        
        # Scale the features
        scaled_features = models['plant_health_scaler'].transform(features)
        
        # Make prediction
        predicted_class = models['plant_health'].predict(scaled_features)
        
        # Get prediction probabilities
        class_probabilities = models['plant_health'].predict_proba(scaled_features)[0]
        
        # Decode the predicted class
        predicted_label = models['plant_health_label_encoder'].inverse_transform(predicted_class)[0]
        
        # Get all class names
        class_names = models['plant_health_label_encoder'].classes_
        
        # Map numeric classes to labels for probabilities
        probability_dict = {
            models['plant_health_label_encoder'].inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(class_probabilities)
        }
        
        # Generate recommendations based on health status
        recommendations = generate_health_recommendations(predicted_label, health_data)
        
        return {
            "health_status": predicted_label,
            "confidence": float(probability_dict[predicted_label]),
            "all_probabilities": probability_dict,
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": f"Error in plant health prediction: {str(e)}"}

def generate_health_recommendations(health_status, data):
    """Generate recommendations based on plant health status"""
    recommendations = []
    
    if health_status == "Healthy":
        recommendations.append("Continue current care practices.")
        recommendations.append("Monitor regularly to maintain optimal health.")
    
    elif health_status == "Moderate Stress":
        if data.get('soil_moisture', 0) < 40:
            recommendations.append("Increase watering frequency to improve soil moisture.")
        if data.get('light_intensity', 0) < 8000:
            recommendations.append("Consider relocating to an area with more sunlight.")
        if data.get('nitrogen_level', 0) < 25:
            recommendations.append("Apply nitrogen-rich fertilizer to boost plant nutrition.")
    
    elif health_status == "High Stress":
        recommendations.append("Immediate intervention required to prevent plant loss.")
        if data.get('soil_ph', 0) < 5.5 or data.get('soil_ph', 0) > 7.5:
            recommendations.append("Adjust soil pH to optimal range (6.0-7.0).")
        if data.get('humidity', 0) < 40:
            recommendations.append("Increase ambient humidity through misting or humidifier.")
        if data.get('potassium_level', 0) < 30:
            recommendations.append("Apply balanced fertilizer with focus on potassium.")
        recommendations.append("Check for pest infestation and disease symptoms.")
    
    return recommendations

def generate_response(message, user_data=None):
    """Generate a response from the Gemini API based on model predictions and user input"""
    try:
        # Create a comprehensive context for Gemini API
        context = (
            "You are an AI agricultural assistant with expertise in plant disease detection, weed identification, "
            "Always mention in brief that you can give better insights if user provided data so it could predict with its ML models and give more accurate decision."
            "Your goal is to help farmers and gardeners make informed decisions about their crops and agricultural practices. "
            "and provide practical advice based on agricultural science. "
            "Always maintain a helpful, informative tone and provide actionable recommendations. "

            "If the user asks about price trends, profitability based on total land, crop selection, or fertilizers used, "
            "perform an internet search (dont mention this ,mention it as look through multiple sources) to provide accurate and up-to-date insights on market prices, demand, and best practices. "
            "If the user inquires about government policies, subsidies, or schemes related to agriculture, retrieve the latest "
            "information from official sources. "

            "Ensure multilingual support, responding in Hindi or any other Indian language if requested. "
            "Always provide responses that include actionable insights based on real-time data, ensuring practical "
            "and implementable recommendations for farmers."
            "Always prioritize clear, structured responses that farmers can easily act upon."
        )

        
        # Add user data context if available
        if user_data:
            context += "Based on your recent predictions and data: " + json.dumps(user_data) + " "
        
        # Combine context with user message
        full_input = context + " User query: " + message
        
        # Generate response using Gemini API
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_input)
        
        return response.text
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question."

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-from-numpy', methods=['POST'])
def predict_numpy():
    """Endpoint to predict from a preprocessed numpy file"""
    try:
        data = request.json
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({"error": "File path is required"}), 400
            
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
            
        result = predict_from_numpy(file_path)
        
        # Store prediction in session for context in future chats
        if 'user_data' not in session:
            session['user_data'] = {}
        
        session['user_data']['plant_disease'] = result
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get previously stored user data if available
        user_data = session.get('user_data', {})
        
        # Generate response
        response = generate_response(message, user_data)
        
        # Log the interaction
        query_log = UserQuery(query=message, response=response, model_used="gemini")
        db.session.add(query_log)
        db.session.commit()
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Determine prediction type
        prediction_type = request.form.get('type', 'plant_disease')
        
        # Check if it's an NPY file (preprocessed numpy array)
        if file.filename.lower().endswith('.npy'):
            # Use the predict_from_numpy function for NPY files
            result = predict_from_numpy(filename)
        elif prediction_type == 'plant_disease':
            result = predict_plant_disease(filename)
        elif prediction_type == 'weed_detection':
            result = predict_weed_type(filename)
        else:
            return jsonify({"error": "Invalid prediction type"}), 400
        
        # Store prediction in session for context in future chats
        if 'user_data' not in session:
            session['user_data'] = {}
        
        session['user_data'][prediction_type] = result
        session['user_data']['last_image'] = filename
        
        return jsonify(result)

@app.route('/predict-price', methods=['POST'])
def price_prediction():
    try:
        data = request.json
        result = predict_price(
            data.get('state', 0),
            data.get('apmc', 0),
            data.get('commodity', 0),
            data.get('commodity_arrivals', 0),
            data.get('commodity_traded', 0),
            data.get('year', 2025),
            data.get('month', 1),
            data.get('day', 1)
        )
        
        # Store prediction in session for context in future chats
        if 'user_data' not in session:
            session['user_data'] = {}
        
        session['user_data']['price_prediction'] = result
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error predicting price: {str(e)}"}), 500

    
import logging

logging.basicConfig(level=logging.DEBUG)

from sklearn.preprocessing import StandardScaler
import joblib


@app.route('/plant_health', methods=['POST'])
def health_prediction():
    try:
        import joblib  # Explicitly import joblib in case of scoping issues

        data = request.json
        logging.debug(f"Received data: {data}")
        if not data:
            return jsonify({"error": "No data provided."}), 400

        # Validate the input data
        required_fields = [
            'soil_moisture', 'ambient_temperature', 'soil_temperature',
            'humidity', 'light_intensity', 'soil_ph', 'nitrogen_level',
            'phosphorus_level', 'potassium_level', 'chlorophyll_content', 'electrochemical_signal'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logging.error(f"Missing fields: {missing_fields}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Log the scaler type
        scaler = joblib.load('models/plant_health_scaler.pkl')  # Update the path if necessary
        print(logging.debug(f"Scaler type: {type(scaler)}"))

        # Validate the scaler type
        if not isinstance(scaler, StandardScaler):
            logging.error(f"Scaler is not a valid StandardScaler object. Type: {type(scaler)}")
            return jsonify({"error": "Scaler is not a valid StandardScaler object."}), 500

        # Prepare the input for the model
        input_data = [
            data['soil_moisture'], data['ambient_temperature'], data['soil_temperature'],
            data['humidity'], data['light_intensity'], data['soil_ph'],
            data['nitrogen_level'], data['phosphorus_level'], data['potassium_level'],
            data['chlorophyll_content'], data['electrochemical_signal']
        ]

        # Scale the input data
        scaled_input = scaler.transform([input_data])

        # Predict the plant health status
        prediction = models['plant_health'].predict(scaled_input)
        predicted_label = models['plant_health_label_encoder'].inverse_transform(prediction)

        logging.debug(f"Prediction result: {predicted_label[0]}")
        return jsonify({"plant_health_status": predicted_label[0]}), 200
    except Exception as e:
        logging.error(f"Error predicting plant health: {str(e)}")
        return jsonify({"error": f"Error predicting plant health: {str(e)}"}), 500
    
@app.route('/recommend-crop', methods=['POST'])
def crop_recommendation():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided."}), 400

        # Validate input and predict
        result = recommend_crop(data)  # Replace with your crop recommendation logic
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error recommending crop: {str(e)}"}), 500

@app.route('/irrigation_prediction', methods=['POST'])
def irrigation_prediction():
    try:
        data = request.json
        logging.debug(f"Received data for irrigation prediction: {data}")
        if not data:
            logging.error("No data provided for irrigation prediction")
            return jsonify({"error": "No data provided."}), 400

        # Validate the input data
        required_fields = [
            'soil_moisture', 'temperature', 'humidity', 'ph', 'rainfall'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logging.error(f"Missing fields for irrigation prediction: {missing_fields}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Prepare the input for the model
        input_data = [
            data['soil_moisture'], data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]
        logging.debug(f"Input data for irrigation prediction: {input_data}")

        # Predict the irrigation requirement
        model = models['irrigation_prediction']
        prediction = model.predict([input_data])
        logging.debug(f"Irrigation prediction result: {prediction[0]}")

        # Return the prediction result
        return jsonify({"irrigation_requirement": prediction[0]}), 200
    except Exception as e:
        logging.error(f"Error predicting irrigation requirement: {str(e)}")
        return jsonify({"error": f"Error predicting irrigation requirement: {str(e)}"}), 500


# Create database tables before first request
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
    

 # Path to your .npy file from the notebook
numpy_file = 'test_image_Tomato Septoria leaf spot.npy'  # Update with your actual file path
        
        # Ensure models are loaded
if not models:
     models = load_models()
        
# Direct prediction from numpy file
result = predict_from_numpy(numpy_file)
print("\nPrediction result:")
print(json.dumps(result, indent=2))

    
