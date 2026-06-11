import os

import numpy as np
import pandas as pd
from openai import OpenAI


def embedding2graph(clean_csv, num_rows=20, dimensions=200):
    """
    read in the clean part of the data, choose at least num_rows(20) and compute the similarity matrix
    dimensions: the dimension of the embedding
    output the similarity matrix
    """
    clean_csv = clean_csv.sample(min(num_rows, len(clean_csv)))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    for col in clean_csv.columns:
        input_list = clean_csv[col].astype(str).tolist()
        input_text = col + ": " + " ".join(input_list)
        response = client.embeddings.create(
            input=input_text,
            model="text-embedding-3-small",
            dimensions=dimensions
        )
        embeddings.append(response.data[0].embedding)

    similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j])

    return similarity_matrix
