import os
from dotenv import load_dotenv
import json
import base64
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import math
from vertexai.language_models import TextEmbeddingModel
import numpy as np
import matplotlib.pyplot as plt
import mplcursors




def plot_2D(x_values, y_values, labels):

    # Create scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values, y_values, alpha=0.5, edgecolors="k", s=40)

    # Create a mplcursors object to manage the data point interaction
    cursor = mplcursors.cursor(scatter, hover=True)

    # aes
    ax.set_title("Embedding visualization in 2D")  # Add a title
    ax.set_xlabel("X_1")  # Add x-axis label
    ax.set_ylabel("X_2")  # Add y-axis label

    # Define how each annotation should look
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(labels[sel.target.index])
        sel.annotation.get_bbox_patch().set(
            facecolor="white", alpha=0.5
        )  # Set annotation's background color
        sel.annotation.set_fontsize(12)

    plt.show()


def generate_batches(sentences, batch_size=5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]


def encode_texts_to_embeddings(sentences):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]


def encode_text_to_embedding_batched(
    sentences, api_calls_per_second=0.33, batch_size=5
):
    # Generates batches and calls embedding API

    embeddings_list = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total=math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return embeddings_list_successful


def clusters_2D(x_values, y_values, labels, kmeans_labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        x_values,
        y_values,
        c=kmeans_labels,
        cmap="Set1",
        alpha=0.5,
        edgecolors="k",
        s=40,
    )  # Change the denominator as per n_clusters

    # Create a mplcursors object to manage the data point interaction
    cursor = mplcursors.cursor(scatter, hover=True)

    # axes
    ax.set_title("Embedding clusters visualization in 2D")  # Add a title
    ax.set_xlabel("X_1")  # Add x-axis label
    ax.set_ylabel("X_2")  # Add y-axis label

    # Define how each annotation should look
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(labels.category[sel.target.index])
        sel.annotation.get_bbox_patch().set(
            facecolor="white", alpha=0.95
        )  # Set annotation's background color
        sel.annotation.set_fontsize(14)

    plt.show()


# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README
# def create_index(embedded_dataset,
#                  num_leaves,
#                  num_leaves_to_search,
#                  training_sample_size):

#     # normalize data to use cosine sim as explained in the paper
#     normalized_dataset = embedded_dataset / np.linalg.norm(embedded_dataset, axis=1)[:, np.newaxis]

#     searcher = (
#         scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product")
#         .tree(
#             num_leaves = num_leaves,
#             num_leaves_to_search = num_leaves_to_search,
#             training_sample_size = training_sample_size,
#         )
#         .score_ah(2, anisotropic_quantization_threshold = 0.2)
#         .reorder(100)
#         .build()
#     )
#     return searcher
