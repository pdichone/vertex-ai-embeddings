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

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")


# load the BQ Table into a Pandas DataFrame
from google.cloud import bigquery
import pandas as pd
def run_bq_query(sql):

    # Create BQ client
    bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    # Try dry run before executing query to catch any errors
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    bq_client.query(sql, job_config=job_config)

    # If dry run succeeds without errors, proceed to run query
    job_config = bigquery.QueryJobConfig()
    client_result = bq_client.query(sql, job_config=job_config)

    job_id = client_result.job_id

    # Wait for query/job to finish running. then get & return data frame
    df = client_result.result().to_arrow().to_pandas()
    print(f"Finished job_id: {job_id}")
    return df

LIMIT = 500
# define list of programming language tags we want to query
language_list = ["python", "java", "dart"]


so_df = pd.DataFrame()

for language in language_list:

    print(f"generating {language} dataframe")

    query_raw = f"""
    SELECT
        CONCAT(q.title, q.body) as input_text,
        a.body AS output_text
    FROM
        `bigquery-public-data.stackoverflow.posts_questions` q
    JOIN
        `bigquery-public-data.stackoverflow.posts_answers` a
    ON
        q.accepted_answer_id = a.id
    WHERE 
        q.accepted_answer_id IS NOT NULL AND 
        REGEXP_CONTAINS(q.tags, "{language}") AND
        a.creation_date >= "2020-01-01"
    LIMIT 
        {LIMIT}
    """

    query = query_raw.format(limit=LIMIT)
    language_df = run_bq_query(query)
    language_df["category"] = language
    so_df = pd.concat([so_df, language_df], ignore_index=True)
    
    # Save the aggregated DataFrame to 'stackoverflow_data.csv'
# so_df.to_csv("stackoverflow_data.csv", index=False)
# print("All data has been saved to 'stackoverflow_data.csv'.")
# print(so_df.head())

# == Next Generate Embeddings for the data ==
import time
import numpy as np

# Generator function to yield batches of sentences

def generate_batches(sentences, batch_size=5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]


# open the saved data
print("\n --> *** Generating Embeddings for a Batch of Data from saved CSV file *** \n")
so_df = pd.read_csv("../data/stackoverflow_data.csv")

so_questions = so_df[0:200].input_text.tolist()
batches = generate_batches(sentences=so_questions)
batch = next(batches)
print(f"\n\n===>Batch size: {len(batch)} \n\n")


# == Get embeddings on a batch of data ==
def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]


batch_embeddings = encode_texts_to_embeddings(batch)

f"\n ==> *** {len(batch_embeddings)} embeddings of size \
{len(batch_embeddings[0])} *** \n"

from utils import encode_text_to_embedding_batched

print("\n --> *** Generating embeddings for the entire dataset *** \n")
so_questions = so_df.input_text.tolist()
question_embeddings = encode_text_to_embedding_batched(
    sentences=so_questions, api_calls_per_second=20 / 60, batch_size=5
)

print("\n ***Shape of Question Embeddings: ***\n " + str(question_embeddings.shape))
# print(question_embeddings)

# == Cluster the embeddings of the Stack Overflow questions ==
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

clustering_dataset = question_embeddings[:100]  # change to 100

print("\n --> *** Cluster the Embeddings of the SO Questions *** \n")
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
    clustering_dataset
)

kmeans_labels = kmeans.labels_

PCA_model = PCA(n_components=2)
PCA_model.fit(clustering_dataset)
new_values = PCA_model.transform(clustering_dataset)

#

from utils import clusters_2D

print("\n --> *** Generating Clusters *** \n")
clusters_2D(
    x_values=new_values[:, 0],
    y_values=new_values[:, 1],
    labels=so_df[:1000],
    kmeans_labels=kmeans_labels,
)

