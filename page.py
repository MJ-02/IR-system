import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
import torch
import os
import textwrap
import numpy as np
## Setting page icon, title, etc ...
st.set_page_config(page_title="IR system", page_icon="🔎", layout="wide")

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
<p>Developed with ❤ by Abdullah and Azd</p>
</div>
"""

def embed_text(docs):
    """
    Converts text to embeding vectors of size 1024
    """
    ## https://huggingface.co/avsolatorio/GIST-large-Embedding-v0
    model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", revision=None)
    embeddings = model.encode(docs, convert_to_tensor=True)
    return embeddings   

def get_scores(query_embedding, doc_embedding, metric = "cosine"):
    """
    Computes Similarity Score between query vector and all document vectors
    metric: default = 'cosine'
    norm -> l2 norm
    dot -> dot product
    inner -> inner product
    l1 -> l1 norm aka manhatan distance
    pearson -> pearson correlation
    spearman -> spearman correlation
    """
    if metric == "cosine":
        scores = F.cosine_similarity(query_embedding, doc_embedding.unsqueeze(0), dim=-1)
        return scores.T
    scores = []
    
    if metric == "l2":
       func = lambda x,y: np.linalg.norm(x-y)
    if metric == "inner":
        func = lambda x,y :1- np.inner(x, y)
    if metric == "l1":
        func = lambda x,y : np.sum(np.abs(x - y))
    if metric == "pearson":
        func = lambda x,y: np.corrcoef(x, y)[0,1]
    if metric == "spearman":
        func = lambda x,y: np.corrcoef(x, y, rowvar=False)[0,1]

    query_embedding = np.array(query_embedding)
    doc_embedding = np.array(doc_embedding)
    for doc in doc_embedding:
        scores.append(func(doc, query_embedding))
    return scores


def search(query, article_df, embeddings, metric = "cosine"):
    query_embedding = embed_text(query)
    results = article_df.copy()
    results['scores'] = get_scores(query_embedding, embeddings, metric)
    
    results = results.sort_values(by="scores",ascending = (metric not in ['cosine', 'pearson', 'spearman']), ignore_index = True)
    col1,col2 = st.columns(2)
    for i, row in results.head(10).iterrows():
        content = textwrap.shorten(row["Content"], width = 200, placeholder = "...")
        if i%2 == 0:
            with col1:
                st.subheader(row["Title"])
                st.write(content)
                st.write(f"Rank: {i}")
                st.write(f"Score: {str(row['scores'])}")
                st.markdown("[Link](%s)"%row["Link"])
                st.divider()
        else:
            with col2:
                st.subheader(row["Title"])
                st.write(content)
                st.write(f"Rank: {i}")
                st.write(f"Score: {row['scores']}")
                st.markdown("[Link](%s)"%row["Link"])
                st.divider()






def main():
    ### Load Web Scrapped Articles Dataframe
    article_df = pd.read_csv("Scrapped_articles.csv")
    embeddings = None

    ### Page structure
    st.title("Wikipedia Search System 🔎")
    col1,col2= st.columns(2)
    with col1:
        text_input = st.text_input("Enter query here")
    with col2:
        metric = st.selectbox("Choose scoring method:", ("cosine","norm", "inner","l1", "spearman", "pearson"), placeholder="Choose a scoring method, defaults to cosine")
        st.write("Scoring method:", metric)
    

    ### Since Embedding All 50 documents is time consuming, 
    ### we can store a local copy of the embeddings and load them as needed
    ### Define the storage directory
    PERSIST_DIR = "./storage"

    ## Embeddings are torch tensors, using torch.save we can serialize them for later use
    ## Check if storage directory exists, if not then create it and save the embeddings
    if not os.path.exists(PERSIST_DIR+"/embeddings.pt"):
        os.makedirs(PERSIST_DIR)
        with st.spinner("Embedding Documents, Hold tight"):
            embeddings = embed_text(article_df['Content'])
        torch.save(embeddings,PERSIST_DIR+"/embeddings.pt" )
        st.toast("Embeddings Done")
    else:
        embeddings = torch.load(PERSIST_DIR+"/embeddings.pt")
        st.toast("Loaded Embeddings from storage")

    ## Streamlit Text box, if input is supplied then call the search function
    if text_input:
        search(text_input, article_df, embeddings, metric)

    st.markdown(footer,unsafe_allow_html=True)





if __name__ == "__main__":
    main()