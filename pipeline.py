import logging

import numpy as np
import pandas as pd

from typing import List, Union, Dict, Tuple, Any

from nova.index_builders import TFIDFIndexBuilder, BM25IndexBuilder
from nova.contextual_retrievers import DocumentRetriever, EntityRelationRetriever

class DocumentRetrievalPipeline:
    
    def __init__(self, config:dict={}, logger:logging.Logger=None):
        self.config = config
        self.logger = logger

        if self.logger is None:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)

            # Create console handler and set level to INFO
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Create formatter and add it to the handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            
            # Add the handler to the logger
            self.logger.addHandler(ch)
        
        self.index_builders = [
                                TFIDFIndexBuilder(config, self.logger),
                                DocumentRetriever(config, self.logger),
                                # SentenceWindowRetriever(config, self.logger),
                                # EntityRelationRetriever(config, self.logger)
                               ]
        
        self.documents = None
        self.sentence_window_delimiter = "\n [SEP] \n"

    def retrieve_sentence_window(self,
                                 retrieved_document:Dict[str, Union[str, float, int]],
                                 window_size:int=3) -> str:

        retrieved_document_index = retrieved_document["index"]

        if retrieved_document_index < 0 or retrieved_document_index >= len(self.documents):
            return retrieved_document["document"]
        
        n_documents_above_index = max(retrieved_document_index - window_size, 0)
        n_documents_below_index = min(retrieved_document_index + window_size + 1, len(self.documents))

        # self.logger.info(f"for index : {retrieved_document_index}, picking documents from index {n_documents_above_index} to {n_documents_below_index}")

        sentence_window = f"{self.sentence_window_delimiter}".join(self.documents[n_documents_above_index:n_documents_below_index])
        
        return sentence_window
        
    
    def build_index(self, documents:List[str]) -> None:
        """
        Build indices using all index builders

        args:
            - documents:List[str] - list of documents to build indices on

        returns:
            None
        """

        self.documents = documents

        self.logger.info(f"Building indices using {len(self.index_builders)} index builders ...")
        for index_builder in self.index_builders:
            try:
                self.logger.info(f"Building index with {index_builder.__class__.__name__} on {len(documents)} documents ...")
                index_builder.build_index(documents)
            except Exception as e:
                self.logger.error(f"Error building index with {index_builder.__class__.__name__}: {e}\n")

    def retrieve_topk(self,
                      query:str,
                      topk:int=5,
                      sentence_window:int=3) -> List[Dict[str, Union[float, int, str]]]:
        """
        Retrieve top k relevant documents for a given query using all index builders

        args:
            - query:str - query string
            - topk:int - number of relevant documents to retrieve. Defaults to 5.
            - sentence_window:int - number of sentences to retrieve around the relevant document. Defaults to 1.

        returns:
            List[Dict[str, Union[float, int, str]]] - list of relevant documents. Each document is a dictionary with the following keys:
                document:(str) - text of the document
                score:(float) - relevance score of the document
                index:(int) - index of the document in the original document list
                sentence_window:(str) - sentence window around the document. A window is defined as sentence_window sentences before and after the retrieved document.
        """
        
        results = []
        for index_builder in self.index_builders:

            try:
                self.logger.info(f"\nRetrieving top {topk} relevant documents with {index_builder.__class__.__name__}")
                results.extend(index_builder.retrieve_topk(query, topk))
            except Exception as e:
                self.logger.error(f"Error retrieving top {topk} relevant documents with {index_builder.__class__.__name__}: {e}\n")

        #hacky way to remove duplicates
        known_documents = set()
        unique_results = []

        for result in results:
            if result["document"] not in known_documents:
                known_documents.add(result["document"])
                unique_results.append(result)

        #return the sentence-window for each document retrieved
        try:
            for idx in range(len(unique_results)):
                unique_results[idx]["sentence_window"] = self.retrieve_sentence_window(unique_results[idx],
                                                                                       window_size=sentence_window)

        except Exception as e:
            self.logger.error(f"Error retrieving sentence window for document: {e}\nreturning documents without sentence window")

        return unique_results