import streamlit as st 
import tensorflow as tf 
import numpy as np 

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image into batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Recommendation System
def get_recommendations(disease_name):
    recommendations = {
        'Apple___Apple_scab': {
            'pesticide': 'Use fungicides like Captan, Carbendazim, or Mancozeb.',
            'fertilizer': 'Cow manure or vermicompost.'
        },
        'Apple___Black_rot': {
            'pesticide': 'Use fungicides such as Mancozeb or Copper oxychloride',
            'fertilizer': 'Compost or farmyard manure (FYM).'
        },
        'Apple___Cedar_apple_rust': {
            'pesticide': 'Use fungicides containing Myclobutanil or Copper oxychloride.',
            'fertilizer': ' Neem cake or compost.'
        },
        'Apple___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost, vermicompost, and cow manure'
        },
        'Bean___angular_leaf_spot': {
            'pesticide': 'Copper-based fungicides like Copper oxychloride',
            'fertilizer': 'Neem cake or compost.'
        },
        'Bean___rust': {
            'pesticide': 'Use fungicides containing Mancozeb or Chlorothalonil',
            'fertilizer': 'Farmyard manure or vermicompost.'
        },
        'Beans___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Blueberry___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Use organic fertilizers like compost, pine needle mulch, and well-rotted manure.'
        },
        'Cherry_(including_sour)___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or well-rotted manure'
        },
        'Cherry_(including_sour)___Powdery_mildew': {
            'pesticide': 'Use sulfur-based fungicides or fungicides containing Myclobutanil',
            'fertilizer': 'Vermicompost or compost.'
        },
        'Corn___Common_Rust': {
            'pesticide': 'Use fungicides like Mancozeb or Copper oxychloride',
            'fertilizer': 'Vermicompost or green manure..'
        },
        'Corn___Gray_Leaf_Spot': {
            'pesticide': 'Use fungicides like Azoxystrobin or Propiconazole',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Corn___Healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Corn___Leaf_Blight': {
            'pesticide': 'Use fungicides like Azoxystrobin or Propiconazole',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Grape___Black_rot': {
            'pesticide': 'Use fungicides like Myclobutanil or Mancozeb.',
            'fertilizer': 'Neem cake or compost.'
        },
        'Grape___Esca_(Black_Measles)': {
            'pesticide': 'Use systemic fungicides such as Myclobutanil.',
            'fertilizer': 'Vermicompost or compost.'
        },
        'Grape___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'pesticide': 'Use fungicides such as Mancozeb or Captan.',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Invalid': {
            'pesticide': '',
            'fertilizer': ''
        },
        'Orange___Haunglongbing_(Citrus_greening)': {
            'pesticide': 'Manage with insecticides like Imidacloprid for the Asian citrus psyllid and nutritional sprays with micronutrients. ',
            'fertilizer': 'Neem cake or compost.'
        },
        'Peach___Bacterial_spot': {
            'pesticide': 'Use copper-based bactericides.',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Peach___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or well-rotted manure.'
        },
        'Pepper,_bell___Bacterial_spot': {
            'pesticide': 'Use copper-based bactericides.',
            'fertilizer': 'Neem cake or vermicompost.'
        },
        'Pepper,_bell___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Potato___Early_Blight': {
            'pesticide': 'Use fungicides such as Chlorothalonil or Mancozeb.',
            'fertilizer': 'Vermicompost or farmyard manure.'
        },
        'Potato___Healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Potato___Late_Blight': {
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Compost or green manure.'
        },
        'Raspberry___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or well-rotted manure.'
        },
        'Rice___Brown_Spot': {
            'pesticide': 'Use fungicides like Propiconazole. Organic fertilizer',
            'fertilizer': 'Neem cake or green manure..'
        },
        'Rice___Healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Rice___Hispa': {
            'pesticide': 'Use insecticides like Carbaryl.',
            'fertilizer': 'Farmyard manure or vermicompost.'
        },
        'Rice___Leaf_Blast': {
            'pesticide': 'Use fungicides containing Tricyclazole or Isoprothiolane.',
            'fertilizer': 'Green manure or compost.'
        },
        'Soybean___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Squash___Powdery_mildew': {
            'pesticide': ' Use sulfur-based fungicides or fungicides containing Myclobutanil.',
            'fertilizer': 'Compost or neem cake.'
        },
        'Strawberry___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Strawberry___Leaf_scorch': {
            'pesticide': 'Use fungicides like Myclobutanil or Captan.',
            'fertilizer': 'Vermicompost or farmyard manure.'
        },
        'Sugarcane___Healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost, farmyard manure, and green manure.'
        },
        'Sugarcane___Mosaic': {
            'pesticide': 'Use resistant varieties and ensure good plant nutrition with organic fertilizers.',
            'fertilizer': 'Neem cake or compost.'
        },
        'Sugarcane___RedRot': {
            'pesticide': ' Focus on resistant varieties and balanced fertilization.',
            'fertilizer': ' Vermicompost or farmyard manure.'
        },
        'Sugarcane___Rust': {
            'pesticide': 'Use fungicides like Propiconazole.',
            'fertilizer': 'Green manure or compost  .'
        },
        'Sugarcane___Yellow': {
            'pesticide': 'Ensure good plant nutrition and use balanced organic fertilizers',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Tomato___Bacterial_spot': {
            'pesticide': 'Use copper-based bactericides.',
            'fertilizer': 'Neem cake or vermicompost.'
        },
        'Tomato___Early_blight': {
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Tomato___healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Tomato___Late_blight': {
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': ' Vermicompost or green manure.'
        },
        'Tomato___Leaf_Mold': {
            'pesticide': 'Use fungicides like Chlorothalonil or Copper-based products',
            'fertilizer': 'Compost or neem cake.'
        },
        'Tomato___Septoria_leaf_spot': {
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'pesticide': 'Use miticides like Abamectin.',
            'fertilizer': 'Vermicompost or neem cake.'
        },
        'Tomato___Target_Spot': {
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Tomato___Tomato_mosaic_virus': {
            'pesticide': 'Ensure good plant nutrition with organic fertilizers and use resistant varieties.',
            'fertilizer': 'Vermicompost or compost.'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'pesticide': 'Control whiteflies with insecticides like Imidacloprid.',
            'fertilizer': 'Neem cake or compost.'
        },
        'Wheat___Brown_Rust': {
            'pesticide': 'Use fungicides like Propiconazole or Mancozeb.',
            'fertilizer': 'Farmyard manure or green manure.'
        },
        'Wheat___Healthy': {
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Wheat___Yellow_Rust': {
            'pesticide': 'Use fungicides like Propiconazole or Tebuconazole',
            'fertilizer': 'Green manure or compost.'
        },
        
        # Add more disease recommendations here...
    }
    return recommendations.get(disease_name, {'pesticide': 'No recommendation', 'fertilizer': 'No recommendation'})

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "homepage.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
               Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.  
                """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### About Dataset
India is an agricultural country where crops such as Rice, Corn/Maize, Potato, Wheat, etc. are some of the major crops.
This dataset has 54 classes focusing on the major crops of India. The images from the dataset were collected from the PlantVillage dataset, Rice Disease Dataset, and Wheat Disease Dataset.

**The 54 classes of the dataset:**

 Apple___Apple_scab,  
 Apple___Black_rot,  
 Apple___Cedar_apple_rust,  
 Apple___healthy,  
 Bean___angular_leaf_spot,  
 Bean___rust,  
 Beans___healthy,  
 Blueberry___healthy,  
 Cherry_(including_sour)___Powdery_mildew,  
 Cherry_(including_sour)___healthy,  
 Corn___Common_Rust,  
 Corn___Gray_Leaf_Spot,  
 Corn___Healthy,  
 Corn___Leaf_Blight,  
 Grape___Black_rot,  
 Grape___Esca_(Black_Measles),  
 Grape___Leaf_blight_(Isariopsis_Leaf_Spot),  
 Grape___healthy,   
 Orange___Haunglongbing_(Citrus_greening),  
 Peach___Bacterial_spot,  
 Peach___healthy,  
 Pepper,_bell___Bacterial_spot,  
 Pepper,_bell___healthy,  
 Potato___Early_Blight,  
 Potato___Healthy,  
 Potato___Late_Blight,  
 Raspberry___healthy,  
 Rice___Brown_Spot,  
 Rice___Healthy,  
 Rice___Hispa,  
 Rice___Leaf_Blast,  
 Soybean___healthy,  
 Squash___Powdery_mildew,  
 Strawberry___Leaf_scorch,  
 Strawberry___healthy,  
 Sugarcane___Healthy,  
 Sugarcane___Mosaic,  
 Sugarcane___RedRot,  
 Sugarcane___Rust,  
 Sugarcane___Yellow,  
 Tomato___Bacterial_spot,  
 Tomato___Early_blight,  
 Tomato___Late_blight,  
 Tomato___Leaf_Mold,  
 Tomato___Septoria_leaf_spot,  
 Tomato___Spider_mites Two-spotted_spider_mite,  
 Tomato___Target_Spot,  
 Tomato___Tomato_Yellow_Leaf_Curl_Virus,  
 Tomato___Tomato_mosaic_virus,  
 Tomato___healthy,  
 Wheat___Brown_Rust,  
 Wheat___Healthy,  
 Wheat___Yellow_Rust  

**Total 91885 files belonging to 54 classes.**
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition 🌿🔍")
    test_image = st.file_uploader("Choose an Image : ")
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        #dfgdfst.snow()
        st.write("Our Prediction : ")
        result_index = model_prediction(test_image)
        # Define class
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Bean___angular_leaf_spot',
 'Bean___rust',
 'Beans___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn___Common_Rust',
 'Corn___Gray_Leaf_Spot',
 'Corn___Healthy',
 'Corn___Leaf_Blight',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Invalid',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_Blight',
 'Potato___Healthy',
 'Potato___Late_Blight',
 'Raspberry___healthy',
 'Rice___Brown_Spot',
 'Rice___Healthy',
 'Rice___Hispa',
 'Rice___Leaf_Blast',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Sugarcane___Healthy',
 'Sugarcane___Mosaic',
 'Sugarcane___RedRot',
 'Sugarcane___Rust',
 'Sugarcane___Yellow',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy',
 'Wheat___Brown_Rust',
 'Wheat___Healthy',
 'Wheat___Yellow_Rust']
        disease_name = class_name[result_index]
        st.success(f"Model is predicting it's {disease_name}")
        
        # Get recommendations
        recommendations = get_recommendations(disease_name)
        st.write("Pesticide Recommendation: ", recommendations['pesticide'])
        st.write("Fertilizer Recommendation: ", recommendations['fertilizer'])
    