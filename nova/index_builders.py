import logging
from typing import List, Union, Dict

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from nova.base_retriever import BaseRetriever

class IndexBuilder(BaseRetriever):

    def __init__(self, config:dict={}, logger:logging.Logger=None):
        super().__init__(config, logger)
        self.index = None

    def build_index(self, documents:List[str]) -> None:
        raise NotImplementedError

    def get_document_index(self, document:str) -> List[float]:
        return self.index

    def preprocess_document(self, document) -> str:
        return document

    def retrieve_topk(self, query:str, topk:int=5) -> List[Dict[str, Union[float, int, str]]]:
        raise NotImplementedError
    
class TFIDFIndexBuilder(IndexBuilder):

    def __init__(self, config:dict, logger:logging.Logger=None):

        super().__init__(config, logger)

        self.indexer = TfidfVectorizer(max_features=5_000, ngram_range=(1, 3))
        self.index = None
        self.documents = None

    def build_index(self, documents:List[str]) -> None:

        if self.indexer is None:
            self.indexer = TfidfVectorizer()

        self.documents = documents
        self.index = self.indexer.fit_transform(documents)

    def get_document_index(self, document:str) -> List[float]:
        return self.indexer.transform([document])

    def retrieve_topk(self, query:str, topk:int=5) -> List[Dict[str, Union[float, int, str]]]:

        query_index = self.get_document_index(query)
        scores = self.index.dot(query_index.T).toarray().flatten()
        topk_indices = scores.argsort()[-topk:][::-1]

        # topk_indices, scores[topk_indices]

        return [{"index": idx, "score": score, "document" : self.documents[idx], "method" : "tfidf"}
                for idx, score in zip(topk_indices, scores[topk_indices])
                if score > 0.5]
    
class BM25IndexBuilder(IndexBuilder):

    def __init__(self, config:dict, logger:logging.Logger=None):
        super().__init__(config, logger)
        self.indexer = None
        self.index = None

    def build_index(self, documents: List[str]) -> None:
        tokenized_documents = [doc.split() for doc in documents]
        self.indexer = BM25Okapi(tokenized_documents)
        self.index = tokenized_documents

    def retrieve_topk(self, query: str, topk: int=5) -> List[Dict[str, Union[float, int, str]]]:
        tokenized_query = query.split()
        scores = self.indexer.get_scores(tokenized_query)
        topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        topk_indices = [(i, scores[i]) for i in topk_indices]

        return [{"index": idx, "score": score, "document": " ".join(self.index[idx]).strip(), "method" : "bm25"}
                for idx, score in topk_indices]