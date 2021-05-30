 
import streamlit as st 
st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://github.com/hardikroutray/ECG/" target="_blank">Erdos ECG team</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from keras.preprocessing import image
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sn
#import warnings
#warnings.filterwarnings("ignore")
import base64

from PIL import Image
import requests
from io import BytesIO

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

#model = load_model("/Users/hexuser/ECG/app/models/bestCNN2D.h5") 


# Showing the original raw data
# if st.checkbox("Show Raw Data", False):
#     st.subheader('Raw data')
#     st.write(df)
# st.title('Quick Explore Models')
# st.sidebar.subheader(' Quick  Explore')
# st.markdown("Tick the box on the side panel to explore the trained models.")
# if st.sidebar.checkbox('2D CNN'):
#     if st.sidebar.checkbox('Dataset Quick Look'):
#         st.subheader('Dataset Quick Look:')
#         # st.write(df.head())
#     if st.sidebar.checkbox("Show Columns"):
#         st.subheader('Show Columns List')
#         # all_columns = df.columns.to_list()
#         # st.write(all_columns)
   
#     if st.sidebar.checkbox('Statistical Description'):
#         st.subheader('Statistical Data Descripition')
#         # st.write(df.describe())
#     if st.sidebar.checkbox('Missing Values?'):
#         st.subheader('Missing values')
#         # st.write(df.isnull().sum())


if st.sidebar.checkbox('Predict yourself (User Interactive)', True):
    # st.subheader('Predict using our trained model :') 

    st.markdown("# Predict using our trained 2D CNN model") 
    st.markdown("Input an ECG image to the trained model to see the real-time predicted output. The images are pre-labelled and randomly selected from the repository.") 
    st.markdown("Additional feature under development: Input your own ECG image")

    if st.button("Normal"):
        # img=Image.open("https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/Normal/Cropped_Images/Normal_1Cropped_lead4.png")
        num = np.random.randint(0,100)
        url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/Normal/Cropped_Images/Normal_{}Cropped_lead4.png".format(num)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        st.image(img, width=700, caption="Normal ECG")

        st.markdown("The model predicted this ECG to be of a person with a **normal** heart.")


    if st.button("Myocardial Infarction"):
        num = np.random.randint(0,100)
        url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/MI/Cropped_Images/MI_{}Cropped_lead4.png".format(num)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        st.image(img, width=700, caption="Myocardial Infarction ECG")

        st.markdown("The model predicted this ECG to be of a person having a **heart attack**.")

    if st.button("Abnormal Heartbeat"):
        num = np.random.randint(0,100)
        url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/HB/Cropped_Images/HB_{}Cropped_lead4.png".format(num)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        st.image(img, width=700, caption="Abnormal Heartbeat ECG")

        st.markdown("The model predicted this ECG to be of a person having a **abnormal heartbeat**.")


    if st.button("History of Myocardial Infarction"):
        num = np.random.randint(0,100)
        url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/PMI/Cropped_Images/PMI_{}Cropped_lead4.png".format(num)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        st.image(img, width=700, caption="History of MI ECG")

        st.markdown("The model predicted this ECG to be of a person having a **history of heart attack**.")

    st.markdown(
        "The data is publicly available **[here](https://doi.org/10.1016/j.dib.2021.106762)** under Creative Commons License. The 2D CNN notebook is hosted **[here](https://github.com/hardikroutray/ECG/blob/main/CNN2D_ECG.ipynb)**")

#    st.markdown('####')
#    st.markdown('####')
#    st.markdown('####')
#    st.markdown('####')
#    st.markdown('####')
#    st.markdown('####')
#    st.markdown('####')


#    st.markdown(" View the app **[source](https://github.com/hardikroutray/ECG_app)** ")


if st.sidebar.checkbox('Time Series (See Animation)', True):
    filename = "movie.gif"
#    if st.button("Play video",True):
#    video_file = open(filename, 'rb')
#    video_bytes = video_file.read()
#    st.video(video_bytes)

#    img = Image.open(filename)
#    st.image(img,  caption="Time Series ECG")

    file_ = open("movie.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
        )

    st.markdown("The 1D CNN notebook is hosted **[here](https://github.com/hardikroutray/ECG/blob/main/Multi_lead_1dCNN.ipynb)**")

if st.sidebar.checkbox('2D CNN (Images)', False):

    st.markdown("# Exploratory Visualization of 2D CNN Model")

    st.markdown("Representative single lead ECG images belonging to different classes of cardiological conditions")
    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/representative_ECG_images.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    st.image(img)
    st.markdown("The single lead images have been cropped from a more larger 12 lead image and preprocessed for classification purposes. Only lead 2 image is used for classification in this project. There is ongoing effort to use all the lead images as different viewpoints and finally use an ensemble method for classification. The images are input as grayscale to the model after resizing them to a standard dimension for all the classes.")

    st.markdown("The images presented are of 4 different classes of cardiological conditions - **Normal,** **Myocardial Infarction (MI),** **Abnormal Heartbeat** and **Previous History of MI**.")

    st.markdown("# Model Summary")
    #model.summary()

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/CNN2D_summary.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=500)
 
    st.markdown("# Accuracy - 90%")

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/Accuracy_2DCNN.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=600)

    st.markdown("The model has an average overall accuracy of **90%** on the test set.")

    st.markdown("# Confusion Matrix")

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/Confusion_matrix_2DCNN.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=600)

# st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)
    st.markdown("The model predicts normal and MI ECG images with a whopping **100 %** accuracy.") 


    st.markdown("# Score Table")

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/CNN2D_scores.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=600)

    st.markdown("The table shows the precision, recall, and f1 score for all the classes.") 

    st.markdown("# Feature/Activation maps")

    st.markdown("As a sanity check that the CNN model is actually learning the ECG lineshape instead of the irrelevant image features, we trace back our steps and show the feature maps after each CNN layer. We do it for all the four representative images shown at the beginning of this page. One is shown below.") 

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/CNN2D_Normal_featuremap.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=800)

    st.markdown("The feature maps for the ECG of a person with a **normal** heart.")


    # url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/CNN2D_MI_featuremap.png'
    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content))
    # st.image(img,width=800)

    # st.markdown("The feature maps for the ECG of a person having a **heart attack**.")

    # url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/CNN2D_HB_featuremap.png'
    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content))
    # st.image(img,width=800)

    # st.markdown("The feature maps for the ECG of a person having an **abnormal heartbeat**.")


    # url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/CNN2D_PMI_featuremap.png'
    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content))
    # st.image(img,width=800)

    # st.markdown("The feature maps for the ECG of a person having a **history of MI**.") 


if st.sidebar.checkbox('1D CNN (Time Series)', False):

    st.markdown("# Exploratory Visualization of 1D CNN Model")

#    st.markdown("Representative time series ECG images after processing for different classes of cardiological conditions")

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/img1.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    st.image(img,caption="Overlay of ECG image line shape and post-processed time series extraction shape")

    st.markdown("Contrary to using only images, converting the data to time series allows us to explore the ECGs of COVID patients. So all the five classes in the dataset are used for the 1D CNN analysis")

    st.markdown("# Model Summary")
    #model.summary()

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/Model_sum_ati_1.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=500)
 
    st.markdown("# Accuracy - 87%")

#    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/conf_At_1.png'
#    response = requests.get(url)
#    img = Image.open(BytesIO(response.content))
#    st.image(img,width=600)

    st.markdown("The model has an average overall accuracy of **87%** on the test set.")

    st.markdown("# Confusion Matrix")

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/conf_At_1.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=600)

# st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)
    st.markdown("The model also predicts MI ECG images with a whopping **100 %** accuracy.") 

    st.markdown("# Score Table")

    url = 'https://raw.githubusercontent.com/hardikroutray/ECG//main/app/images/Score_table_1.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img,width=600)

    st.markdown("The table shows the precision, recall, and f1 score for all the classes.") 


if st.sidebar.checkbox('KNN (Time Series)', False):

    st.markdown("# Under Development")

if st.sidebar.checkbox('Random Forest (Time Series)', False):

    st.markdown("# Under Development")

if st.sidebar.checkbox('Miscellaneous', False):

    st.markdown("# Under Development")
