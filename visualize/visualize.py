import warnings
import sys

sys.path.append("../")

from utils import plot_2D


# Ignore all warnings
warnings.filterwarnings("ignore")

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import os


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

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# == Visulaize the embeddings in 2D space ==
input_1 = "Japan High GDP with one of the oldest populations, challenges in maintaining workforce"
input_2 = "Germany Strong economy with an aging population, balancing innovation with tradition"
input_3 = "Italy Economic struggles in recent years, aging population adds pressure to social systems"
input_4 = "South Korea Rapid economic growth, facing challenges due to low birth rates and aging citizens"
input_5 = "Spain Moderate GDP with significant portion of population over 65, healthcare burden increasing"
input_6 = (
    "Canada High standard of living, facing a future with a growing elderly demographic"
)
input_7 = "France Stable economy, dealing with pension reforms as population ages"

import numpy as np

text_list = [input_1, input_2, input_3, input_4, input_5, input_6, input_7]

embeddings = []

for input_text in text_list:
    emb_aux = embedding_model.get_embeddings([input_text])[0].values
    embeddings.append(emb_aux)
embeddings_array = np.array(embeddings)
print("Shape: " + str(embeddings_array.shape))

from sklearn.decomposition import PCA

PCA_model = PCA(n_components=2)
PCA_model.fit(embeddings_array)

new_embeddings = PCA_model.transform(embeddings_array)

print("Shape: " + str(new_embeddings.shape))
print(new_embeddings)

plot_2D(new_embeddings[:, 0], new_embeddings[:, 1], text_list)
