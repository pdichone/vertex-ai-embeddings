import warnings
import sys

sys.path.append("../")


# Ignore all warnings
warnings.filterwarnings("ignore")

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import os
import pickle


load_dotenv()
key_path = "../vertex-ai-course.json"  # Path to the json key associated with your service account from google cloud

# Create credentials object

credentials = Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

if credentials.expired:
    credentials.refresh(Request())

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

import vertexai

# initialize vertex
vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
import pandas as pd

so_database = pd.read_csv("../data/stackoverflow_data.csv")
# print("Shape: " + str(so_database.shape))
# print(so_database)


# == Load the question embeddings ==
# Encode the stack overflow data

# so_questions = so_database.input_text.tolist()
# print("Embedding the stack overflow questions...")
# question_embeddings = encode_text_to_embedding_batched(
#     sentences=so_questions, api_calls_per_second=20 / 60, batch_size=5
# )

# save embeddings to a file
# Save the embeddings to a pickle file
# with open("question_embeddings_app.pkl", "wb") as file:
#     print("Writing embeddings to 'question_embeds.pkl'...")
#     pickle.dump(question_embeddings, file)

# print("Embeddings have been saved to 'question_embeds.pkl'.")


with open("../data/question_embeddings_app.pkl", "rb") as file:
    # Call load method to deserialze
    question_embeddings = pickle.load(file)

    print(question_embeddings)

so_database["embeddings"] = question_embeddings.tolist()
# print(so_database)

# == Semantic Search ==
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin as distances_argmin

# query = ["How to parse a simple JSON file in python?"]
query = ["How to concat dataframes pandas?"]
query_embedding = model.get_embeddings(query)[0].values

cos_sim_array = cosine_similarity(
    [query_embedding], list(so_database.embeddings.values)
)

print("---> *** Cosine Similarity: *** \n")
print(cos_sim_array)

# Get the index of the most similar question
index_doc_cosine = np.argmax(cos_sim_array)

index_doc_distances = distances_argmin(
    [query_embedding], list(so_database.embeddings.values)
)[0]
# print("\n---> *** Index of the most similar question: *** \n")

# print(so_database.input_text[index_doc_cosine])

# print(so_database.output_text[index_doc_cosine])

# == Question and answer generation ==

from vertexai.language_models import TextGenerationModel

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

context = (
    "Question: "
    + so_database.input_text[index_doc_cosine]
    + "\n Answer: "
    + so_database.output_text[index_doc_cosine]
)


prompt = f"""Here is the context: {context}
             Using the relevant information from the context,
             provide an answer to the query: {query}."
             If the context doesn't provide \
             any relevant information, \
             answer with \
             [I couldn't find a good match in the \
             document database for your query] \
                 make sure to provide the code snippet if necessary.
             
             """

t_value = 0.2
response = generation_model.predict(
    prompt=prompt, temperature=t_value, max_output_tokens=1024
)

# print(" \n ===> **** Generated response: **** === \n")
# print(response.text)


# ==== Scale with approximate nearest neighbor search ====
# We will be using the HNSW algorithm for approximate nearest neighbor search
# through Faiss because it is fast and efficient
# === pip install faiss-cpu ===

import time
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

question_embeddings = np.array(list(so_database.embeddings.values)).astype("float32")

# Step 1: Create a Faiss Index
embedding_dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
index.add(question_embeddings)


# Step 2: Perform the Search
query_embedding = model.get_embeddings(query)[0].values
# query_embedding = model.get_embeddings([query])[0].values.astype("float32")
start = time.time()

# Find the nearest neighbor
D, I = index.search(np.array([query_embedding]), 1)  # Search for 1 nearest neighbor
end = time.time()

# Display results from Faiss
for id, dist in zip(I[0], D[0]):
    print("\n ==>> *** Faiss Results: *** \n")
    print(f"[docid:{id}] [{dist}] -- {so_database.input_text[int(id)][:125]}...")

print("\n ==>>> Faiss Latency (ms):", 1000 * (end - start))


# Step 3: Perform Cosine Similarity for Comparison
start = time.time()
cos_sim_array = cosine_similarity([query_embedding], question_embeddings)
index_doc = np.argmax(cos_sim_array)
end = time.time()


# Display results from Cosine Similarity
print(
    f"\n ==>> [docid:{index_doc}] [{np.max(cos_sim_array)}] -- {so_database.input_text[int(index_doc)][:125]}..."
)
print(" \n ==> Cosine Similarity Latency (ms):", 1000 * (end - start))
