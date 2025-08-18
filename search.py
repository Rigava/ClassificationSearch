import streamlit as st
import pandas as pd
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from PIL import Image
import base64
from scipy.spatial.distance import cosine

# Initialize embeddings
key = "AIzaSyAKEaaM7fWIErN3VbikjP_T5m0UfhBy5iE"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
#Functions if not using scipy cosine algo
def dot(A, B): 
    return sum(a * b for a, b in zip(A, B))
def cosine_similarity(a, b):
    return dot(a, b) / ( (dot(a, a) ** 0.5) * (dot(b, b) ** 0.5) )

# Path to the local background image
image_path = r"C:\Users\AKJ064\OneDrive - Maersk Group\SearchApp\static\images\ship.PNG"  # Replace with your image filename
# Function to convert image to base64 for use in CSS
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Add custom CSS for background image
if os.path.exists(image_path):
    image_base64 = get_image_base64(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{image_base64}");
            background-size: 10%;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
            color: blue;  /* Change text color if needed */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Background image not found. Please check the file path.")



# Streamlit app layout
st.title("Incident theme discovery application")

# Input for the search query
query = st.text_input("Enter your search query:")

if query:
    search_term_vector = embeddings.embed_query(query)

    # Load the embeddings dataframe
    # df = pd.read_csv('source_embeddings.csv')
    df = pd.read_csv('HipoThemes_Cluster.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    # Calculate similarities
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    
    # Sort and select top results
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(20)

    cols = ["IncidentNumber","IncidentDescription","Cluster","predicted_probability","predicted_category"]
    frame = sorted_by_similarity[cols]
    st.dataframe(frame)
    
    # Display results
    results = sorted_by_similarity['IncidentDescription'].values.tolist()
    
    st.write(f"Search results for: **{query}**")
    for result in results:
        st.write(f"- {result}")
    
