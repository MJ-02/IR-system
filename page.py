import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
import torch
import os
import textwrap

## Setting page icon, title, etc ...
st.set_page_config(page_title="IR system", page_icon="🔎", layout="wide")



def embed_text(docs):
    """
    Converts text to embeding vectors of size 1024
    """
    ## https://huggingface.co/avsolatorio/GIST-large-Embedding-v0
    model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", revision=None)
    embeddings = model.encode(docs, convert_to_tensor=True)
    return embeddings   

def get_scores(query_embedding, doc_embedding):
    """
    Computes Cosine Similarity Score between query vector and all document vectors
    """
    scores = F.cosine_similarity(query_embedding, doc_embedding.unsqueeze(0), dim=-1)
    return scores.T

def search(query, article_df, embeddings):
    query_embedding = embed_text(query)
    results = article_df.copy()
    results['scores'] = get_scores(query_embedding, embeddings)
    results = results.sort_values(by="scores",ascending=False)
    col1,col2 = st.columns(2)
    for i, row in results.head(10).iterrows():
        content = textwrap.shorten(row["Content"], width = 100, placeholder = "...")
        if i%2 == 0:
            with col1:
                st.subheader(row["Title"])
                st.write(content)
                st.markdown("[Link](%s)"%row["Link"])
                st.divider()

        else:
            with col2:
                st.subheader(row["Title"])
                st.write(content)
                st.markdown("[Link](%s)"%row["Link"])
                st.divider()






def main():
    ### Load Web Scrapped Articles Dataframe
    article_df = pd.read_csv("Scrapped_articles.csv")
    embeddings = None



    ### Page structure
    st.title("IR Engine")
    text_input = st.text_input("Enter query here")


    ### Since Embedding All 50 documents is time consuming, 
    ### we can store a local copy of the embeddings and load them as needed
    ### Define the storage directory
    PERSIST_DIR = "./storage"

    ## Embeddings are torch tensors, using torch.save we can serialize them for later use
    ## Check if storage directory exists, if not then create it and save the embeddings
    if not os.path.exists(PERSIST_DIR+"/embeddings.pt"):
        os.makedirs(PERSIST_DIR)
        embeddings = embed_text(article_df['Content'])
        torch.save(embeddings,PERSIST_DIR+"/embeddings.pt" )
        st.success("Embeddings Done")
    else:
        embeddings = torch.load(PERSIST_DIR+"/embeddings.pt")
        st.success("Loaded Embeddings from storage")

    ## Streamlit Text box, if input is supplied then call the search function
    if text_input:
        search(text_input, article_df, embeddings)




if __name__ == "__main__":
    main()