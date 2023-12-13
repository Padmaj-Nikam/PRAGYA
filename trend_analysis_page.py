import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import pickle
from PIL import Image

# Load the ResNet50 model for feature extraction
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Load image feature embeddings and filenames from pickle files
feature_list = pickle.load(
    open("/home/paddie/Documents/pragya1/PRAGYA/trend-analysis/embeddings.pkl", "rb")
)
filenames = pickle.load(
    open("/home/paddie/Documents/pragya1/PRAGYA/trend-analysis/filenames.pkl", "rb")
)


# Streamlit App
def main2():
    st.title("Textile Trend Analysis Web App")
    st.header("Retailer Inventory Insights")

    # User uploads an image of a textile product
    uploaded_file = st.file_uploader(
        "Upload an image of a textile product...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(
            uploaded_file,
            caption="Uploaded Textile Product Image.",
            use_column_width=True,
        )

        # Perform feature extraction on the uploaded image
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        # Display the extracted features
        st.subheader("Textile Feature Embeddings:")
        st.write(normalized_result)

        # Provide insights based on the features (dummy insights for illustration)
        st.subheader("Retailer Recommendations:")
        st.write("1. This product aligns with current color trends.")
        st.write("2. Consider bundling this product with complementary items.")
        st.write("3. High texture features indicate a premium quality product.")


if __name__ == "__main__":
    main2()
