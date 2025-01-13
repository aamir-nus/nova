import numpy as np
import pandas as pd

from uuid import uuid4
from typing import List

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def soft_blitz(texts:List[str],
                 encoding_model:SentenceTransformer,
                 threshold:float=0.6,
                 min_size:int=3) -> pd.DataFrame:

    text_embeddings = encoding_model.encode(texts)
    sim_matrix = cosine_similarity(text_embeddings, text_embeddings)

    communities = pd.DataFrame(columns=["node", "cls"])

    #secret sauce - soft mini-blitz clustering
    for idx, row in enumerate(sim_matrix):

        connected_items = np.where(row >= threshold)[0]
        if len(connected_items) < min_size:
            continue

        sim_matrix[connected_items, connected_items] = 0.

        # sim_matrix[:, connected_items] = 0.
        # sim_matrix[connected_items, :] = 0.

        simple_community = pd.DataFrame({"node" : [texts[_] for _ in connected_items]})
        simple_community["cls"] = str(uuid4())

        communities = pd.concat([communities, simple_community], axis=0, ignore_index=True)

    #enable multi-label soft clustering
    text_nodes = pd.DataFrame({"node" : texts})
    unclustered_communities = text_nodes[~text_nodes["node"].isin(communities["node"])].copy()
    # unclustered_communities["cls"] = "-1"

    unclustered_communities["cls"] = unclustered_communities["node"].apply(lambda x: str(uuid4())) # assume community size of 1

    communities = pd.concat([communities,
                             unclustered_communities],
                             axis=0, ignore_index=True)

    return communities

def simple_blitz(texts:List[str],
                 encoding_model:SentenceTransformer,
                 threshold:float=0.6,
                 min_size:int=3) -> pd.DataFrame:
    
    text_embeddings = encoding_model.encode(texts)
    sim_matrix = cosine_similarity(text_embeddings, text_embeddings)

    communities = pd.DataFrame(columns=["node", "cls"])

    #secret sauce - soft mini-blitz clustering
    for idx, row in enumerate(sim_matrix):

        connected_items = np.where(row >= threshold)[0]
        if len(connected_items) < min_size:
            continue

        # sim_matrix[connected_items, connected_items] = 0.

        sim_matrix[:, connected_items] = 0.
        sim_matrix[connected_items, :] = 0.

        simple_community = pd.DataFrame({"node" : [texts[_] for _ in connected_items]})
        simple_community["cls"] = str(uuid4())

        communities = pd.concat([communities, simple_community], axis=0, ignore_index=True)

    #enable multi-label soft clustering
    text_nodes = pd.DataFrame({"node" : texts})
    unclustered_communities = text_nodes[~text_nodes["node"].isin(communities["node"])].copy()
    # unclustered_communities["cls"] = "-1"
    
    unclustered_communities["cls"] = unclustered_communities["node"].apply(lambda x: str(uuid4())) # assume community size of 1

    communities = pd.concat([communities,
                             unclustered_communities],
                             axis=0, ignore_index=True)

    return communities