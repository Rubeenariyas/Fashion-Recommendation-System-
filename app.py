import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st

st.header("Fashion Recommendation System")

# Load precomputed image features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Initialize feature extraction model
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False
model = tf.keras.models.Sequential([base, GlobalMaxPool2D()])

# Build the Nearest Neighbors model (request up to 7 neighbors)
n_neighbors = 7
neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

def extract_features_from_uploaded_image(upload_file, model):
    """Extract normalized feature vector from an uploaded image (BytesIO)."""
    # Ensure pointer is at start
    upload_file.seek(0)
    img = Image.open(upload_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / (norm(result) + 1e-10)
    return norm_result

# Streamlit UI
upload_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    # Display uploaded image (reset and read)
    upload_file.seek(0)
    image_obj = Image.open(upload_file)
    st.image(image_obj, caption="Uploaded Image", use_container_width=True)
    st.success("Image uploaded successfully!")

    # Extract features and find similar images
    input_img_features = extract_features_from_uploaded_image(upload_file, model)
    distances, indices = neighbors.kneighbors([input_img_features])

    st.subheader("Recommended Images")

    # make sure we don't request more columns than available neighbors
    found = min(n_neighbors, len(indices[0]))
    cols = st.columns(found)
    for i, col in enumerate(cols):
        idx = indices[0][i]
        with col:
            st.caption(f"Rank {i+1}")
            # If filenames store relative/local paths, st.image accepts the path.
            # If you stored raw image bytes or URLs, adapt accordingly.
            st.image(filenames[idx], use_container_width=True)
