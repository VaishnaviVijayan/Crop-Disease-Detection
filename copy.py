import streamlit as st 
import tensorflow as tf 
import numpy as np 

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image into batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


#Home page
if (app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "homepage.jpg"
    st.image(image_path,use_column_width=True)
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
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    ### About Dataset
Bangladesh is an agricultural country were crops such as Rice, Corn/Maize, Potato, Wheat, etc. are some of the major crops.
This dataset has 15 classes focusing on the major crops of Bangladesh. The images from the dataset were collected from the PlantVillage dataset, Rice Disease Dataset and Wheat Disease Dataset.

**The 15 classes of the dataset:**

Corn___Common_Rust
Corn___Gray_Leaf_Spot
Corn___Healthy
Corn___Leaf_Blight
Potato___Early_Blight
Potato___Healthy
Potato___Late_Blight
Rice___Brown_Spot
Rice___Healthy
Rice___Hispa
Rice___Leaf_Blast
Wheat___Brown_Rust
Wheat___Healthy
Wheat___Yellow_Rust


**Total 31,053 files in 54 classes.**
                """)
    

#Prediction Page
elif (app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image : ")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction : ")
        result_index = model_prediction(test_image)
        #Define class
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
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))