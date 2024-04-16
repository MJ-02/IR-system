from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
import streamlit as st
import pandas as pd
import nltk


## App configs
OptimumEmbedding.create_and_save_optimum_model(
    "BAAI/bge-small-en-v1.5", "./bge_onnx"
)
nltk.download("stopwords")
Settings.embed_model = OptimumEmbedding(folder_name="./bge_onnx")
st.set_page_config(page_title="IR system", page_icon="🐍", layout="wide")



## Page structure
st.title("IR Engine")

## Query Input
text_search = st.text_input("Enter query here", value="")



df = pd.read_csv("Scrapped_articles.csv")
# st.write(df)
