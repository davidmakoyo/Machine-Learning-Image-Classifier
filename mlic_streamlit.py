import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from PIL import Image

def main():
    st.title('Machine Learning Image Classifier')
    st.write('Upload any image that fits into one of the available categories and see if the model can classify it correctly.')

    st.subheader('Categories')
    st.write('airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck')

    file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'], key='image')
    if file:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # pre-process the image, load the model, and make a prediction
        resized_image = image.resize((32, 32))
        # need values between 0 and 1
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape((1, 32, 32, 3))

        model = tf.keras.models.load_model('cifar10_model.h5')

        predictions = model.predict(img_array)
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        fig, ax = plt.subplots()
        # we want to display not only the highest value but also the confidence of the model
        y_pos = np.arange((len(cifar10_classes)))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel('Confidence')
        ax.set_title('Predictions')

        st.pyplot(fig)
    else:
        st.text('You have not uploaded an image yet.')
    
    st.markdown("### [GitHub Repo](https://github.com/davidmakoyo/Machine-Learning-Image-Classifier)")

if __name__ == '__main__':
    main()
