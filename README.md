# Terrathon-Cisco  
First-place winner for the Cisco problem statement at the Terrathon hackathon, sponsored by Cisco.  

# AI-Powered Smart Agriculture System  

## Terrathon 4.0 Hackathon - First Place Winner  

This repository contains my individual contribution to the AI-powered smart agriculture system developed for the CISCO problem statement in the Terrathon 4.0 Hackathon, where our team secured first place. I was responsible for all AI, ML, and backend aspects of the project.  

This project aims to enhance agricultural sustainability, optimize resource usage, and improve farming efficiency using AI and deep learning.  

---  

## Repository Structure  

### 1. all_models/  
Contains code for all machine learning and deep learning models used in this project.  

- **other_models/**: Includes all models except for Plant Disease Prediction and Weed Detection.  
- **Plant Disease Prediction**: Deep learning model trained on approximately 60,000 images.  
- **Weed Detection**: CNN model trained on a 1.2GB dataset.  

### 2. model_integration_with_chatbot/  
Contains code for integrating the models with the Gemini API for a chatbot-based AI assistant.  

- **index.html**: A simple chatbot UI where users can:  
  - Upload images for Plant Disease Detection and Weed Detection.  
  - Ask the chatbot agriculture-related questions.  
  - Currently, the chatbot only receives context from Plant Disease and Weed Detection. Integration with other models is still being developed.  

### 3. presentation.pdf  
Contains the final presentation that was presented to the judges.  

---  

## AI & ML Models  

The system integrates multiple ML models to address various agricultural challenges:  

1. **Plant Disease Prediction** (Deep Learning - InceptionV3)  
   - Trained on approximately 60,000 plant images  
   - Accuracy: 97 percent  
   - Identifies 58 plant diseases and plant types  

2. **Weed Detection** (CNN)  
   - Trained on a 1.2GB dataset  
   - Accuracy: 89 percent  
   - Detects 12 maize weed types  
   - Lower accuracy due to dataset imbalance  

3. **Plant Health Prediction** (Decision Tree - Max Depth 6)  
   - Uses 11 soil, environmental, and nutrient factors  
   - Accuracy: 85.6 percent  

4. **Price Prediction** (XGBoost + Random Forest)  
   - Modal Price Prediction (Regression): 87.3 percent (R² Score)  
   - Price Range Prediction (Classification): 83.9 percent (F1 Score)  

5. **Crop Recommendation** (Random Forest - 22 features)  
   - Accuracy: 90.1 percent  

6. **Smart Irrigation** (XGBoost + Decision Tree)  
   - Predicts water efficiency and irrigation risk  
   - Accuracy: 88.5 percent  

---  

## Dataset Considerations  
For most models (except Plant Disease and Weed Detection), relatively small datasets were used to achieve decent accuracy. The goal was to prioritize feature diversity over dataset size, as real-time deployment would allow for continuous learning with larger datasets.  

---  

## LLM Integration with Gemini API (Work in Progress)  

The chatbot uses Google’s Gemini API for natural language understanding and integrates outputs from ML models.  

### How It Works  
1. Users interact via index.html  
2. ML models process image uploads for Plant Disease and Weed Detection  
3. The chatbot receives context from these models and responds accordingly  
4. Full integration with other models (crop recommendation, irrigation, price prediction, etc.) is still under development  

### Technical Implementation  
- Flask Backend  
- ML Model Outputs stored in Flask session for chatbot context  
- Gemini API processes queries based on:  
  - User questions  
  - Model predictions (currently limited to disease and weed detection)  

---  

## Conclusion  
This project represents an AI-powered all-in-one smart agriculture solution, combining deep learning, machine learning, and LLM-based chatbot interactions. While still a work in progress, the system lays the foundation for precision farming, proactive decision-making, and sustainability in agriculture.
