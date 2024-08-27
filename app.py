import warnings

# from utils import plot_2D

# Ignore all warnings
warnings.filterwarnings("ignore")

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import os


load_dotenv()
key_path = "./vertex-ai-course.json"  # Path to the json key associated with your service account from google cloud

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

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

embedding = embedding_model.get_embeddings(["Hello!"])
# print(embedding)

vector = embedding[0].values
print(f"Length = {len(vector)}")  # dimension of the embedding

#  ==== Next -- compare the embeddings of sentences
# == pip install scikit-learn ==
from sklearn.metrics.pairwise import cosine_similarity

# Obtaining embeddings for different statements
embedding_alpha = embedding_model.get_embeddings(["How do airplanes stay in the sky?"])

embedding_beta = embedding_model.get_embeddings(
    ["What's the secret to making perfect coffee?"]
)

embedding_gamma = embedding_model.get_embeddings(["Can you recommend a good book?"])

# Extracting vectors from the embeddings
vector_alpha = [embedding_alpha[0].values]
vector_beta = [embedding_beta[0].values]
vector_gamma = [embedding_gamma[0].values]

print(cosine_similarity(vector_alpha, vector_beta))
print(cosine_similarity(vector_beta, vector_gamma))
print(cosine_similarity(vector_alpha, vector_gamma))
