import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from groq import Groq
from langchain_groq import ChatGroq
import json

# ---------------- STREAMLIT UI ----------------
st.title("📊 Text Clustering + Labelling Tool")

st.write("""
Upload an Excel file with a column of text descriptions.  
This app will:
1. Cluster the text into groups.  
2. Extract top keywords & sample texts for each cluster.  
3. Ask **ChatGroq LLM** to assign **Nature of Operation** and **Subactivity** labels.  
4. Export the results to Excel.  
""")

# API key input (not stored)
groq_api_key = st.text_input("Enter your ChatGroq API Key", type="password", placeholder="gsk_...")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None and groq_api_key:
    # Load Excel
    df = pd.read_excel(uploaded_file)

    # Select text column
    text_column = st.selectbox("Select the text column to analyze", df.columns)

    # Number of clusters
    num_clusters = st.slider("Select number of clusters", 2, 20, 5)

    if st.button("Run Clustering"):


        # Initialize ChatGroq client
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name='llama-3.3-70b-versatile')

        # Vectorize text
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(df[text_column].astype(str))

        # KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X)

        # Extract top keywords per cluster
        terms = vectorizer.get_feature_names_out()
        cluster_keywords = {}
        for i in range(num_clusters):
            centroid = kmeans.cluster_centers_[i]
            top_indices = centroid.argsort()[-5:][::-1]  # top 5
            cluster_keywords[i] = ", ".join([terms[ind] for ind in top_indices])

        # Function to query ChatGroq LLM
        def get_cluster_theme(samples,key):
            prompt = f'''
            You are analyzing industrial operation descriptions. 
            I will give you a cluster of text descriptions along with important keywords.  

            Task:
            - Suggest a short label for the "Nature of Operation" (general category)
            - Suggest a short label for the "Subactivity" (specific task within that category)
            Sample Texts:
            {samples}

            Keywords:
            {key}
            Return only valid JSON in this format (nothing else):
            {{
            "Nature_of_Operation": "....",
            "Subactivity": "...."
            }}
            '''
            response = llm.invoke(prompt)  # just pass plain string to LLM
            try:
                return json.loads(response.content)  # <-- fix: .content
            except:
                return {"Nature_of_Operation": "Unknown", "Subactivity": "Unknown"}


        # Assign themes per cluster
        cluster_themes = {}
        for i in range(num_clusters):
            sample_texts = df[df["Cluster"] == i][text_column].astype(str).head(5).tolist()
            theme = get_cluster_theme(sample_texts,cluster_keywords[i])
            cluster_themes[i] = theme
            st.write(f"Cluster {i}: {theme['Nature_of_Operation']} - {theme['Subactivity']}")
        # Map final labels
        df["Nature_of_Operation"] = df["Cluster"].map(lambda x: cluster_themes[x]["Nature_of_Operation"])
        df["Subactivity"] = df["Cluster"].map(lambda x: cluster_themes[x]["Subactivity"])
        

        # ---------------- Display ----------------
        st.subheader("Clustered Data with Final Labels")
        st.dataframe(df[[text_column, "Cluster", "Nature_of_Operation", "Subactivity"]])

        # Cluster visualization (PCA)
        pca = PCA(n_components=2, random_state=42)
        reduced_X = pca.fit_transform(X.toarray())
        plt.figure(figsize=(8,6))
        plt.scatter(reduced_X[:,0], reduced_X[:,1], c=df["Cluster"], cmap="tab10", alpha=0.7)
        for i in range(num_clusters):
            plt.text(np.median(reduced_X[df["Cluster"]==i,0]),
                     np.median(reduced_X[df["Cluster"]==i,1]),
                     f"{cluster_themes[i]['Nature_of_Operation']}\n{cluster_themes[i]['Subactivity']}",
                     fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
        plt.title("Text Clusters (LLM-Labeled via ChatGroq)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        st.pyplot(plt)

        # Download option
        st.subheader("Download Results")
        output_file = "clustered_labeled_output.xlsx"
        df.to_excel(output_file, index=False)
        with open(output_file, "rb") as f:
            st.download_button("Download Labeled Excel", f, file_name=output_file)

