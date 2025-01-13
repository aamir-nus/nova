import ast
import logging
import networkx as nx

import numpy as np
import pandas as pd

from typing import List, Dict, Union, Any

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

from nova.index_builders import IndexBuilder
from concept_extractor.text_chunker import LineChunker
from concept_extractor.concept_extractor import ConceptExtractor, ConceptValidator

from utils.helpers import soft_blitz, simple_blitz
from utils.llm_pipeline import AsyncLLMPipelineBuilder

nlp = spacy.load("en_core_web_sm")

class DocumentRetriever(IndexBuilder):

    def __init__(self, config: dict={}, logger: logging.Logger = None):

        super().__init__(config, logger)
        
        self.documents = None
        self.index = None
        self.vectorizer = SentenceTransformer("all-mpnet-base-v2")

        self.document_context = None

    def _form_ngrams(self, document: str) -> List[str]:
        vectorizer = CountVectorizer(ngram_range=(1, 3)).build_analyzer()
        return vectorizer(document)
    
    def _extract_keywords(self, documents: List[str]) -> List[Dict[str, Union[int, str, List[str]]]]:
        
        keywords = []

        if isinstance(documents, str):
            documents = [documents]

        for idx, doc in enumerate(nlp.pipe(documents, batch_size=128)):

            entities = [ent.text for ent in doc.ents]

            noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

            # non_pronoun_targets = [f"{token.head.text} {token.text}" for token in doc if token.dep_ in ["nsubj", "dobj", "pobj"]
            #                        and token.pos_ not in ["PRON", "DET", "PUNCT"]
            #                        if token.head.text != token.text]
            non_pronoun_targets = []

            doc_keywords = list(set(entities + noun_chunks + non_pronoun_targets))
            
            #get only the keywords which are not substrings of other keywords
            doc_keywords = [keyword for keyword in doc_keywords
                            if len([kw for kw in doc_keywords if kw != keyword and keyword in kw]) == 0]

            keywords.append({"index" : idx, "document" : doc.text, "keywords" : doc_keywords})
            
        return keywords

    def _extract_ngrams(self, documents: List[str]) -> List[str]:
        all_ngrams = set()
        for doc in documents:
            all_ngrams.update(self._form_ngrams(doc))
            
        return list(all_ngrams)

    def _embed_documents(self, documents: List[str]):
        return self.vectorizer.encode(documents)
    
    def _build_document_context(self, documents:List[str]) -> str:

        if self.document_context is None:

            self.summarizer = AsyncLLMPipelineBuilder(system_prompt="summarize the following texts. Be informative in your summary and mention key points/ideas. Keep the summary to 10 lines or less.",
                                                        model="gemini-1.5-flash", max_timeout_per_request=30)
            
            #limit to 300 'documents'
            self.document_context = self.summarizer.batch_predict([", ".join(documents[:300])])[-1]
        
        return self.document_context
    
    def build_inverse_index(self, document_index:List[Dict[str, Union[str, float, int]]]) -> None:
        
        inverse_index = {}

        for doc in document_index:
            for keyword in doc["keywords"]:
                if keyword['entity'] not in inverse_index:
                    inverse_index[keyword['entity']] = []
                inverse_index[keyword['entity']].append(doc['index'])

        return inverse_index

    def build_index(self, documents: List[str]) -> None:

        if self.index is None:

            self.documents = documents
            self.index = []

            self.all_ngrams = self._extract_ngrams(documents)
            self.all_keywords = self._extract_keywords(documents)

            #change all_keywords to a dictionary for faster lookup
            self.all_keywords = {keyword["index"] : keyword["keywords"] for keyword in self.all_keywords}

            #for ngrams, to make sure they are relevant -- we will use cosine similarity
            self.doc_embeddings = self._embed_documents(documents)
            self.ngram_embeddings = self._embed_documents(self.all_ngrams)
            self.similarities = cosine_similarity(self.doc_embeddings,
                                                  self.ngram_embeddings)
            
            self.inverse_index = {}

            self.logger.info(f"Building document index ...")

            for idx, doc in enumerate(documents):
               
                keywords = []
                doc2ngram_similarities = self.similarities[idx]
                ngrams_similar_to_doc = np.where(doc2ngram_similarities >= 0.85)[0]
                keywords.extend([{"entity" : self.all_ngrams[ngram_idx], "score" : doc2ngram_similarities[ngram_idx]}
                                    for ngram_idx in ngrams_similar_to_doc])

                #add all the keywords extracted via spacy to this index

                # auto_extracted_keywords = [keyword for keyword in self.all_keywords if keyword["index"] == idx][0]["keywords"]
                # keywords.extend([{"entity" : auto_kw, "score" : 0.8} for auto_kw in auto_extracted_keywords])

                keywords.extend([{"entity" : auto_kw, "score" : 0.8}
                                 for auto_kw in self.all_keywords.get(idx, [])])

                #filter out keywords that are not in the document -- but we will lose some keywords that are paraphrasals
                # keywords = [keyword for keyword in keywords if keyword['entity'].lower() in doc.lower()]
                
                self.index.append({"index": idx, "document": doc, "keywords": keywords})

            self.logger.info(f"Built index for {len(documents)} documents | {len(self.index)} keyword pairs extracted")
            self.logger.info(f"Building inverse index ...")

            self.inverse_index = self.build_inverse_index(self.index)
            self.logger.info(f"Built inverse index for {len(self.inverse_index)} keywords\n")

            self.logger.info(f"Building document context for better retrievals ...")
            self.document_context = self._build_document_context(documents=documents)
            self.logger.info(f"Built document context for {len(documents)} documents\n")
        
    def retrieve_topk(self, query: str, topk: int=5) -> List[Dict[str, Union[int, str, List[str]]]]:

        #get simple document embedding matches
        relevant_topk_docs, relevant_topk_ngrams, query_ngrams = [], [], []
        
        query_embedding = self.vectorizer.encode(query)

        query_similarity = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        topk_indices = query_similarity.argsort()[-topk:][::-1]

        relevant_topk_docs = [{"index": idx, "score" : query_similarity[idx], "document": self.documents[idx], "keywords": self.index[idx]["keywords"], "method" : "semantic_match"}
                              for idx in topk_indices]

        query_ngrams = self._extract_keywords(query)[0]["keywords"]

        try:
            self.logger.info(f"Using document context to extract more keywords ...")

            query_expansion_prompt = f"you are an expert analyst working on a transcript. A summary of the transcript is as below:\n{self.document_context}\n"
            with AsyncLLMPipelineBuilder(query_expansion_prompt, model="gemini-1.5-flash") as llm:
                expanded_query = llm.batch_predict([f"give keywords relevant to answering '{query}'. only return a single string of keywords, separated by commas"])[-1]

            expanded_query = expanded_query.replace("\n", " ").replace("'", "").replace("`", "").replace("json", "").replace("python", "").strip()
            expanded_query = expanded_query.split(",")
            query_ngrams += expanded_query

            self.logger.info(f"Expanded the keywords from query! : {expanded_query}\n")

        except:
            self.logger.warning(f"Could not expand query : {query}")

        #if there are no ngrams extracted from the query, return the topk docs based on simple semantic similarity
        if query_ngrams == []:
            return relevant_topk_docs
        
        self.logger.info(f"extracted ngrams from query : {query_ngrams}\n")
        self.logger.info(f"Retrieving documents based on ngram matches in inverse index ...")

        #else, get ngrams and find relevant documents
        query_ngram_embeddings = self.vectorizer.encode(query_ngrams)
        inverse_index_embeddings = self.vectorizer.encode(list(self.inverse_index.keys()))

        query_ngram_similarity = cosine_similarity(query_ngram_embeddings, inverse_index_embeddings)
        
        ngram_topk_indices, matched_inverse_index_ngrams = [], []

        for row in query_ngram_similarity:
            connected_items = np.where(row >= 0.8)[0]
            ngram_topk_indices.extend(list(connected_items))

        for idx in ngram_topk_indices:

            ngram = list(self.inverse_index.keys())[idx]
            
            #for tracking purposes
            matched_inverse_index_ngrams.append(ngram)

            documents_in_ngram_indices = self.inverse_index[ngram]
            documents_in_ngram_indices = [doc for doc in self.index if doc["index"] in documents_in_ngram_indices]
            for doc in documents_in_ngram_indices:
                doc["score"] = 0.8 # query_ngram_similarity[idx] is ideal, but we will use a fixed score for now
                doc["method"] = "semantic_keyword_match"
            relevant_topk_ngrams.extend(documents_in_ngram_indices)

        matched_inverse_index_ngrams = list(set(matched_inverse_index_ngrams))
        self.logger.info(f"matched {len(matched_inverse_index_ngrams)} inverse index ngrams : {matched_inverse_index_ngrams[:10]} ...\n")

        relevant_topk = relevant_topk_docs + relevant_topk_ngrams
        
        relevant_topk = sorted(relevant_topk, key=lambda x: x["score"], reverse=True)
        # relevant_topk = [relevant for relevant in relevant_topk if relevant["score"] > 0.5]

        return relevant_topk

class SentenceWindowRetriever:

    def __init__(self, config: dict={}, logger: logging.Logger = None, base_retriever=None):

        self.base_retriever = base_retriever
        self.config = config
        self.logger = logger

        if self.logger is None:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)

        if self.base_retriever is None:
            self.base_retriever = DocumentRetriever(self.config, self.logger)
            self.logger.info(f"Base retriever not provided, using {self.base_retriever} as base document retriever...")

        self.documents = None
        self.index = None

    def build_index(self, documents: List[str]) -> None:
        self.base_retriever.build_index(documents)
        self.index = self.base_retriever.index

    def retrieve_topk(self, query: str, topk: int=5) -> List[Dict[str, Union[int, str, List[str]]]]:

        #get simple document embedding matches
        retrieved_document_matches = self.base_retriever.retrieve_topk(query, topk)

        #get sentence window matches -- topn sentences above and below the target sentence
        for idx in range(len(retrieved_document_matches)):
            retrieved_document_index = retrieved_document_matches[idx]["index"]
            topn_above_retrieved_document_index = self.index[max(0, retrieved_document_index - 3):retrieved_document_index]
            topn_below_retrieved_document_index = self.index[retrieved_document_index + 1:retrieved_document_index + 4]

            try:
                retrieved_document_matches[idx]["sentence_window"] = "\n".join([match["document"]
                                                for match in topn_above_retrieved_document_index + [retrieved_document_matches[idx]] + topn_below_retrieved_document_index]).strip()
                
            except:
                pass

        return retrieved_document_matches
    
class EntityRelationRetriever(DocumentRetriever):

    def __init__(self, config: dict={}, logger: logging.Logger = None):
        super().__init__(config, logger)

        self.nlp = pipeline("token-classification",
                            tokenizer = AutoTokenizer.from_pretrained("knowledgator/UTC-DeBERTa-large-v2"),
                            model = AutoModelForTokenClassification.from_pretrained("knowledgator/UTC-DeBERTa-large-v2"),
                            aggregation_strategy = 'first')
        
        # Load reranker model and tokenizer
        # self.reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # self.reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _process(self, texts:List[str], prompt, threshold=0.5):
        """
        Processes text by preparing prompt and adjusting indices.

        Args:
        text (str): The text to process
        prompt (str): The prompt to prepend to the text

        Returns:
        list: A list of dicts with adjusted spans and scores
        """
        # Concatenate text and prompt for full input

        if isinstance(texts, str):
            texts = [texts]

        inputs_ = [f"{prompt}\n{text}" for text in texts]
        results = self.nlp(inputs_, batch_size=128) # Run NLP on full input
        processed_results = []
        prompt_length = len(prompt) # Get prompt length
    
        for idx, extracted_relations in enumerate(results):
            # check whether score is higher than treshold
            if extracted_relations is None or extracted_relations == []:
                continue

            processed_result = {"index" : idx, "document" : texts[idx], "keywords" : []}

            for result in extracted_relations:
                
                if result['score']<threshold:
                    continue

                # Adjust indices by subtracting prompt length
                start = result['start'] - prompt_length 
                # If indexes belongs to the prompt - continue
                if start<0:
                    continue
                end = result['end'] - prompt_length
                # Extract span from original text using adjusted indices
                span = texts[idx][start:end]
                # Create processed result dict
                _processed_result = {
                    'score': result['score'],
                    'entity': str(span).strip()
                }
                processed_result["keywords"].append(_processed_result)
                # processed_results.append(processed_result)
            processed_results.append(processed_result)

        return processed_results
    
    def build_index(self, documents: List[str]) -> None:

        #build entity-relation graph using UTC models, GNN models, etc.
        if not hasattr(self, "index") or not hasattr(self, "inverse_index"):
            self.index = []
            self.inverse_index = {}

            self.index.extend(self._process(documents, prompt="""Extract all named entities in the following texts:"""))
            self.logger.info(f"Built index for {len(documents)} documents | {len(self.index)} entities extracted")

            # Build inverse index

            # for doc in self.index:
            #     for keyword in doc["keywords"]:
            #         if keyword['entity'] not in self.inverse_index:
            #             self.inverse_index[keyword['entity']] = []
            #         self.inverse_index[keyword['entity']].append(doc['index'])

            self.inverse_index = self.build_inverse_index(self.index)
            self.logger.info(f"Built inverse index for {len(self.inverse_index)} entities")

        return None
    
    def _rerank_documents(self,
                          query:str,
                          retrieved_documents:List[Dict[str, Union[int, str, List[str]]]]) -> List[Dict[str, Union[int, str, List[str]]]]:

        #TODO : rerank documents based on entity relations extracted from the query
        reranker_inputs = [self.reranker_tokenizer(query, doc["document"], return_tensors='pt', truncation=True, padding=True) for doc in retrieved_documents]

        # Get reranker scores
        reranker_scores = []
        for inputs in reranker_inputs:
            with torch.no_grad():
                outputs = self.reranker_model(**inputs)
                scores = outputs.logits.squeeze().tolist()
                reranker_scores.append(scores)

        # Attach scores to documents
        for doc, score in zip(retrieved_documents, reranker_scores):
            doc["reranker_score"] = score

        # Sort documents by reranker score
        reranked_documents = sorted(retrieved_documents, key=lambda x: x["reranker_score"], reverse=True)

        return reranked_documents
    
    def _get_documents_relevant_to_query_entities(self, query_entities:Dict[str, Union[str, float]], topk:int=5):

        print(f"query entities : {[entity['entity'] for entity in query_entities]}\n")

        query_entity_embeddings = self._embed_documents([entity["entity"] for entity in query_entities])
        inverse_index_embeddings = self._embed_documents(list(self.inverse_index.keys()))
        similarities = cosine_similarity(query_entity_embeddings, inverse_index_embeddings)

        topk_indices = []
        for row in similarities:
            
            sim_indices = np.where(row >= 0.7)[0]

            if len(sim_indices) > 0:
                sim_indices = sim_indices.tolist()
                topk_indices.extend(sim_indices)

        topk_indices = list(set(topk_indices))

        entities_relevant_to_query = [list(self.inverse_index.keys())[idx] for idx in topk_indices]

        self.logger.info(f"Retrieved entities relevant to query : {entities_relevant_to_query}")
        print(f"entities relevant to query : {entities_relevant_to_query}\n")

        documents_with_entities_relevant_to_query = [self.inverse_index[entity_name]
                                             for entity_name in entities_relevant_to_query]
        documents_with_entities_relevant_to_query = list(set([arr for sublist in documents_with_entities_relevant_to_query for arr in sublist]))

        #TODO : make this a dataframe slice or an np.where(), sub-optimal for now
        documents_with_entities_relevant_to_query = [doc for doc in self.index if doc["index"] in documents_with_entities_relevant_to_query]
        
        for doc in documents_with_entities_relevant_to_query:
            doc["score"] = 0.6
            doc["method"] = "semantic_entity_match"

        self.logger.info(f"Retrieved documents where entities are relevant to query : {documents_with_entities_relevant_to_query}\n")
        print(f"documents with entities relevant to query : {documents_with_entities_relevant_to_query}\n")

        return documents_with_entities_relevant_to_query
    
    def retrieve_topk(self, query: str, topk: int=5) -> List[Dict[str, Union[int, str, List[str]]]]:

        #get entities from the query using UTC models first
        query_entities = self._process(query, prompt="""Extract all named entities in the following texts:""")
        self.logger.info(f"\nExtracted entities from the query using UTC : {query_entities}\n")

        #get entities from the query using gemini
        
        er_graph_builder = AsyncLLMPipelineBuilder(system_prompt="you are an expert knowledge graph engineer, you have been tasked with traversing a knowledge graph for the given query. split a given query into its entities and relations. output should be a list of dictionaries, where each dictionary has keys 'entity_1', 'relation', 'entity_2' 'score'",
                                                model="gemini-1.5-flash")
        query_er = er_graph_builder.batch_predict([f"return only a list of dictionaries, where each dictionary has keys 'entity_1', 'relation', 'entity_2' 'score' for the given query : {query}"])[0]
        query_er = query_er.replace("`", "").replace("json", "").replace("python", "").replace("\n", "").strip()
        query_er = ast.literal_eval(query_er)

        #TODO: refactor
        query_entities += [{"entity" : entity["entity_1"],
                       "score" : 0.8} for entity in query_er] + [{"entity" : entity["entity_2"],
                                                                  "score" : 0.8} for entity in query_er]
        
        if query_entities == []:
            self.logger.info(f"Could not extract any entities from the query, returning top hybrid matches to the query ...")
            query_entities = [{"keywords" : [{"entity" : query, "score" : 1.0}]}]
            query_entities = query_entities[0]["keywords"]

        #get documents that contain the entities
        relevant_documents = []
        for entity in query_entities:
            if entity["entity"] in self.inverse_index:
                document_indices_with_keywords = self.inverse_index[entity["entity"]]
                documents_in_indices = [doc for doc in self.index if doc["index"] in document_indices_with_keywords]
                for doc in documents_in_indices:
                    doc["score"] = 0.8
                    doc["method"] = "entity_match"
                relevant_documents.extend(documents_in_indices)

        #also get documents that are 'similar' to the entities in the query
        documents_with_entities_relevant_to_query = self._get_documents_relevant_to_query_entities(query_entities,
                                                                                                  topk)
        
        # documents_with_entity_relations_relevant_to_query = self._get_documents_relevant_to_query_entity_relations(query_entities,
        #                                                                                           topk)

        relevant_documents.extend(documents_with_entities_relevant_to_query)

        # try:
        #     relevant_documents = self._rerank_documents(query, relevant_documents)[:topk]
        # except Exception as e:
        #     self.logger.error(f"Could not rerank documents : {e}")
        #     relevant_documents = relevant_documents[:topk]

        return relevant_documents

class GraphIndexRetriever(DocumentRetriever):

    def __init__(self,
                 config:dict={},
                 logger:logging.Logger=None):
        
        super().__init__(config=config, logger=logger)

        self.concepts = None
        self.valid_concepts = None

        self.connected_nodes = pd.DataFrame(columns=["node", "originating_texts", "summary"])
        self.communities = pd.DataFrame(columns=["node", "cls", "type", "originating_texts", "summary"])

        self.graph_context = None
        self.sentence_window_separator = "\n [SEP] \n"
        
        self.encoding_model = SentenceTransformer("all-mpnet-base-v2")
        
        self.node_embeddings = None
        self.community_embeddings = None

        self.min_community_size = 3
        self.node_level_similarity = 0.7
        self.edge_level_similarity = 0.85
        self.relation_level_similarity = 0.9

    def _extract_keywords(self, documents: List[str]) -> List[Dict[str, Union[int, str, List[str]]]]:
        
        keywords = []

        if isinstance(documents, str):
            documents = [documents]

        for idx, doc in enumerate(nlp.pipe(documents, batch_size=128)):

            entities = [ent.text for ent in doc.ents]
            noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

            doc_keywords = list(set(entities + noun_chunks))
            
            #get only the keywords which are not substrings of other keywords
            doc_keywords = [keyword for keyword in doc_keywords
                            if len([kw for kw in doc_keywords if kw != keyword and keyword in kw]) == 0]

            keywords.append({"index" : idx, "document" : doc.text, "keywords" : doc_keywords})
            
        return keywords

    def _get_keywords(self, document:str, concepts:List[Dict[str, Any]]) -> List[str]:
        
        keywords = []

        try:
            concepts = pd.DataFrame(concepts)
            
            #keywords are just nodes in the graph -- either a source node (node_1) or a target node (node_2)
            graph_lookup = concepts[concepts["originating_text"] == document]
            keywords = graph_lookup["node_1"].tolist() + graph_lookup["node_2"].tolist()

            keywords = list(set(keywords))
            keywords = [{"entity": keyword,
                         "score" : 1.0,
                         "extraction_type" : "concept_graph"}
                         for keyword in keywords]
        except:
            pass
        
        return keywords

    def _build_graph_context(self, graph:pd.DataFrame) -> str:

        if self.graph_context is None:

            self.summarizer = AsyncLLMPipelineBuilder(system_prompt="summarize the following texts. Be informative in your summary and mention key points/ideas. Keep the summary to 5 lines or less.",
                                                        model="gemini-1.5-flash")
            
            #limit to 500 edges
            self.graph_context = self.summarizer.batch_predict([", ".join(graph.sample(frac=1)['edge'].tolist()[:500])])[-1]
        
        return self.graph_context
    
    def _find_connected_nodes(self, concepts:List[Dict[str, Any]]) -> pd.DataFrame:

        concepts = pd.DataFrame(concepts)
        G = nx.from_pandas_edgelist(concepts, 'node_1', 'node_2', edge_attr='weight')

        _min_connections = max(3, len(concepts)//1000) #minimum number of connections to be considered 'connected'
        connected_nodes = [node for node in G.nodes() if G.degree[node] >= _min_connections]
        connected_nodes = pd.DataFrame(connected_nodes,
                                    columns=["node"]).drop_duplicates(subset=["node"])
        
        connected_nodes["originating_texts"] = connected_nodes["node"].apply(lambda x: list(set(concepts[concepts["node_1"] == x]["originating_text"].unique().tolist() + concepts[concepts["node_2"] == x]["originating_text"].unique().tolist())))
        connected_nodes["originating_texts"] = connected_nodes["originating_texts"].apply(lambda x: self.sentence_window_separator.join(x))

        self.logger.info(f"generating summaries for {connected_nodes.shape[0]} connected nodes ...")

        summaries_to_generate = [f"summarize the following texts around ideas/opinions related to {row['node']}:\n{row['originating_texts'][:5000]}" #limit to 5000 characters at most
                                for _, row in connected_nodes.iterrows()]
        
        summaries = self.summarizer.batch_predict(summaries_to_generate)
        connected_nodes["summary"] = summaries
        
        return connected_nodes

    def _find_communities(self, concepts:List[Dict[str, Any]]) -> pd.DataFrame:

        concepts = pd.DataFrame(concepts)

        if concepts.empty:
            return concepts
        
        #update the minimum community size
        self.min_community_size = max(3, len(concepts)//1000)
        self.logger.info(f"Changed min community size to {self.min_community_size} ...")

        #if there are more than 1000 concepts, we will use simple_blitz otherwise we will use soft_blitz
        community_detection = soft_blitz
        if len(concepts) > 1000:
            community_detection = simple_blitz

        #update minimum node->node sim, node->rel->node sim, and edge sim thresholds
        
        entities_only_communities = concepts["node_1"].unique().tolist() + concepts["node_2"].unique().tolist()
        entities_only_communities = list(set(entities_only_communities))
        self.logger.info(f'number of entities : {len(entities_only_communities)}')

        edge_communities = concepts["edge"].unique().tolist()
        self.logger.info(f'number of edges : {len(edge_communities)}')

        relation_only_communities = [f"{row['node_1']} {row['relation']} {row['node_2']}"
                                    for _, row in concepts.iterrows()]
        relation_only_communities = list(set(relation_only_communities))
        
        self.logger.info(f'number of relations : {len(relation_only_communities)}')

        entities_only_communities = community_detection(entities_only_communities,
                                                self.encoding_model,
                                                min_size=self.min_community_size,
                                                threshold=self.node_level_similarity)
        entities_only_communities["type"] = "similar_nodes"
        entities_only_communities["originating_texts"] = entities_only_communities["node"].apply(lambda x: list(set(concepts[concepts["node_1"] == x]["originating_text"].unique().tolist() + concepts[concepts["node_2"] == x]["originating_text"].unique().tolist())))
        entities_only_communities["originating_texts"] = entities_only_communities["originating_texts"].apply(lambda x: "\n".join(x))
        
        edge_communities = community_detection(edge_communities,
                                        self.encoding_model,
                                        min_size=self.min_community_size,
                                        threshold=self.edge_level_similarity)
        edge_communities["type"] = "similar_edges"
        edge_communities["originating_texts"] = edge_communities["node"].apply(lambda x: concepts[concepts["edge"] == x]["originating_text"].unique().tolist())
        edge_communities["originating_texts"] = edge_communities["originating_texts"].apply(lambda x: "\n".join(x))

        relation_only_communities = community_detection(relation_only_communities,
                                                self.encoding_model,
                                                min_size=self.min_community_size,
                                                threshold=self.relation_level_similarity)
        relation_only_communities["type"] = "similar_relations"
        relation_only_communities["originating_texts"] = relation_only_communities["node"]
            
        communities = pd.concat([entities_only_communities[entities_only_communities["cls"] != "-1"],
                                edge_communities[edge_communities["cls"] != "-1"],
                                relation_only_communities[relation_only_communities["cls"] != "-1"]
                                ],
                                axis=0, ignore_index=True)
            
        #get cls level summaries
        # unique_clusters = communities.drop_duplicates(subset=["cls"])
        # unique_clusters["group_summaries"] = unique_clusters["cls"].apply(lambda x : communities[communities["cls"] == x]["originating_texts"].tolist())

        # print(unique_clusters.head(), unique_clusters.shape)
        # communities_to_summarize = [f"summarize the following texts around ideas/opinions related to the keywords {communities[communities['cls'] == row['cls']]['node'].unique().tolist()}:\n\n{', '.join(row['group_summaries'])}"
        #                             for idx, row in unique_clusters.iterrows()]
        
        # cls_summaries = self.summarizer.batch_predict(communities_to_summarize)
        # unique_clusters["summary"] = cls_summaries

        # communities = pd.merge(communities,
        #                         unique_clusters[["cls", "summary"]],
        #                         on="cls",
        #                         how="left")
        
        # communities.reset_index(inplace=True, drop=True)
        communities["summary"] = communities["node"].tolist()

        return communities
    
    def search_nodes(self,
                    query:str,
                    graph_context:str,
                    node_subgraph:pd.DataFrame) -> List[Dict[str, Union[str, List[str]]]]:
        """
        'search' nodes that are frequently mentioned, reason which of the nodes have relevant information for the query, summarize information

        Args:
            - query : str : the query to search for
            - graph_context : str : the summary of the graph
            - node_subgraph : pd.DataFrame : the subgraph of the nodes
            - index_builder : DocumentRetriever : the index builder object

        Returns:
            - relevant_to_query : List[Dict[str, Union[str, List[str]]] : the relevant nodes to the query
        """
        
        if self.node_embeddings is None:
            self.node_embeddings = self.encoding_model.encode(node_subgraph["node"].tolist())
            self.logger.info(f"encoded {len(node_subgraph)} node embeddings ...")

        extracted_entity_keywords = self._extract_keywords(query)[0]["keywords"]
        extracted_entity_keywords = [{"node_1" : entity, "relation" : query, "node_2" : entity, "edge" : query}
                                for entity in extracted_entity_keywords]
        self.logger.info(f"Extracted entities and relations from the query : {extracted_entity_keywords}\n")
        
        try:

            er_query = f"""you are an expert analyst working on a transcript. A summary of the transcript is as below:\n{graph_context}\n
                        Given this context, extract the entities and relationships between them as a list of dictionaries. each dictionary
                        should have the keys 'node_1', 'relation', 'node_2', 'edge'."""
        
            graph_query_builder = AsyncLLMPipelineBuilder(system_prompt=er_query,
                                                        model="gemini-1.5-flash")
            query_entity_relations = graph_query_builder.batch_predict([f"only return a list of dictionaries with the keys 'node_1', 'relation', 'node_2', 'edge' for the following query : {query}"])[0]
            query_entity_relations = query_entity_relations.replace("\n", " ").replace("'", "").replace("`", "").replace("json", "").replace("python", "").strip()


            self.logger.info(f"Expanded query : {query_entity_relations}")
            query_entity_relations = ast.literal_eval(query_entity_relations)
            query_entity_relations += extracted_entity_keywords
        
        except:
            query_entity_relations = extracted_entity_keywords

        if query_entity_relations == []:
            return pd.DataFrame(columns=node_subgraph.columns).to_dict('records')
        
        entities_in_query = pd.DataFrame(query_entity_relations)["node_1"].tolist() + pd.DataFrame(query_entity_relations)["node_2"].tolist()

        entities_in_query = list(set(entities_in_query))

        relevant_to_query = node_subgraph.copy()

        #find entities that are in the node-subgraph that might be relevant to the query
        query_entity_embeddings = self.encoding_model.encode(entities_in_query)
        # node_subgraph_embeddings = self.encoding_model.encode(node_subgraph["node"].tolist())
        node_subgraph_embeddings = self.node_embeddings
        query2node_sim_matrix = cosine_similarity(query_entity_embeddings, node_subgraph_embeddings)

        similar_subgraph_nodes = []
        for row in query2node_sim_matrix:
            connected_items = np.where(row >= 0.7)[0]
            similar_subgraph_nodes.extend(list(connected_items))

        similar_subgraph_nodes = list(set(similar_subgraph_nodes))
        similar_subgraph_nodes = node_subgraph.iloc[similar_subgraph_nodes]["node"].tolist()

        self.logger.info(f"found {len(similar_subgraph_nodes)} nodes in the subgraph that are similar to the query entities ...")
        self.logger.info(f"similar nodes : {similar_subgraph_nodes}\n")

        #get the originating texts for these nodes
        relevant_to_query = relevant_to_query[relevant_to_query["node"].isin(similar_subgraph_nodes)]
        relevant_to_query["type"] = "node_search"

        if relevant_to_query.empty:
            self.logger.info(f"no relevant nodes found in the subgraph for the query")

        return relevant_to_query.to_dict('records')
    
    def search_communities(self,
                           query:str,
                           graph_context:str,
                           communities:pd.DataFrame) -> List[Dict[str, Union[str, List[str]]]]:
        """
        search the communities formed in the graph for relevant information.

        - get relevant keywords to 'answer' a query using the graph context
        - search the community nodes for relevant information
        - for every unique 'cls' of the community, retrieve all 'connected' nodes

        Args:
            - query : str : the query to search for
            - graph_context : str : the summary of the graph
            - communities : pd.DataFrame : the communities formed in the graph

        Returns:
            - communities : List[Dict[str, Union[str, List[str]]] : the connected nodes in the communities
        """
        if self.community_embeddings is None:
            self.community_embeddings = self.encoding_model.encode(communities["node"].tolist())
            self.logger.info(f"encoded {len(communities)} community nodes ...")

        try:
            query_keywords = self._extract_keywords(documents=[query])[0]["keywords"]
        except:
            query_keywords = []
        
        #simple query expansion -- ask an LLM what 'keywords' are required to answer the query
        query_expansion_prompt = f"you are an expert analyst working on a transcript. A summary of the transcript is as below:\n{graph_context}\n"
        with AsyncLLMPipelineBuilder(query_expansion_prompt, model="gemini-1.5-flash") as llm:
            expanded_query = llm.batch_predict([f"give keywords relevant to answering '{query}'. only return a single string of keywords, separated by commas"])[-1]

        try:
            expanded_query = expanded_query.replace("\n", " ").replace("'", "").replace("`", "").replace("json", "").replace("python", "").replace("indeterminate", "").strip().lower()
            expanded_query = expanded_query.split(",")
        except:
            expanded_query = []
        finally:
            expanded_query += query_keywords
            expanded_query = list(set(expanded_query))

        if expanded_query == [] or expanded_query == ['']:
            return pd.DataFrame(columns=communities.columns).to_dict('records')
        
        self.logger.info(f"expanded query keywords : {expanded_query}")
        
        #search the communities for relevant information
        query_embeddings = self.encoding_model.encode(expanded_query)
        # community_embeddings = self.encoding_model.encode(communities["node"].tolist())
        community_embeddings = self.community_embeddings
        query2community_sim_matrix = cosine_similarity(query_embeddings, community_embeddings)

        relevant_communities = []
        for row in query2community_sim_matrix:
            connected_items = np.where(row >= 0.8)[0]
            relevant_communities.extend(list(connected_items))

        relevant_communities = list(set(relevant_communities))
        relevant_communities = [communities["node"].tolist()[rel_idx]
                                for rel_idx in relevant_communities]
        relevant_communities = list(set(relevant_communities))
        
        self.logger.info(f"found {len(relevant_communities)} communities relevant to the query ...")
        self.logger.info(f"relevant communities : {relevant_communities[:5]} ... \n")

        #for every connected node, get the cluster it belongs to
        connected_nodes = communities[communities["node"].isin(relevant_communities)]["cls"].unique().tolist()
        connected_nodes = communities[communities["cls"].isin(connected_nodes)]
        connected_nodes["score"] = 0.8

        connected_nodes.drop_duplicates(subset=["node", "cls"], inplace=True)

        return connected_nodes.to_dict('records')
    
    def _search_index(self, query:str):
        #simple semantic search in the index and community search
        pass

    def build_index(self, documents: List[str]) -> None:

        if self.index is None:

            self.logger.info(f"Building a graph-index using LLMs for {len(documents)} documents ...")

            data_chunks = LineChunker(docs=documents).get_output()
            self.logger.info(f"Chunked {len(documents)} documents into {len(data_chunks)} chunks ...\n")
            
            concept_extractor = ConceptExtractor(chunks=data_chunks, comprehensive=False)
            self.concepts = concept_extractor.get_output()

            self.logger.info(f"Extracted {len(self.concepts)} concepts from {len(documents)} documents ...\n")

            validator = ConceptValidator(concepts=self.concepts)
            self.valid_concepts = validator.validate()

            self.logger.info(f"Found {len(self.valid_concepts)} valid concepts from {len(documents)} documents ...\n")

            self.index = [{"index": idx,
                        "document": doc,
                        "keywords": self._get_keywords(doc, self.valid_concepts)}
                        for idx, doc in enumerate(documents)]
            
            self.logger.info(f"Built index for {len(documents)} documents | {len(self.index)} keyword pairs extracted")
            self.logger.info(f"Building inverse index ...")

            #build graph context -- this is a summary of the graph.
            self._build_graph_context(pd.DataFrame(self.valid_concepts))

            self.logger.info(f"Finding connected nodes ...")
            self.connected_nodes = self._find_connected_nodes(self.valid_concepts)

            self.logger.info(f"Finding communities ...")
            self.communities = self._find_communities(self.valid_concepts)

            self.inverse_index = self.build_inverse_index(self.index)
            self.logger.info(f"Built inverse index for {len(self.inverse_index)} keywords\n")
        
    def retrieve_topk(self, query: str, topk: int=5) -> List[Dict[str, Union[int, str, List[str]]]]:

        relevant_nodes, relevant_docs = [], []

        # relevant_nodes = self.search_nodes(query,
        #                                   graph_context=self.graph_context,
        #                                   node_subgraph=self.connected_nodes)
        
        relevant_nodes += self.search_communities(query,
                                                graph_context=self.graph_context,
                                                communities=self.communities)
        if relevant_nodes == []:
            return pd.DataFrame(columns=['index', 'document', 'node', 'type', 'keywords'])
        
        # relevant_nodes = pd.merge(pd.DataFrame(relevant_nodes)[["originating_texts", "node", "type"]].rename(columns={"originating_texts" : "document"}),
        #                         pd.DataFrame(self.index)[["document", "keywords", "index"]],
        #                         on="document",
        #                         how="left").dropna(subset=["index"]).drop_duplicates(subset=["document", "index"])
        # relevant_nodes['index'] = relevant_nodes['index'].astype(int)
        
        # summary_docs = [{'index' : 'aux_doc',
        #                  'document' : rel_document.get('summary', ''),
        #                  'keywords' : [],
        #                  'score' : 0.5}
        #                  for rel_document in relevant_nodes]
        # summary_docs = [doc for doc in summary_docs if doc.get('document') != '']
        
        #get all documents associated with the node -- overkill for number of documents retrieved

        # documents_for_nodes = [self.inverse_index.get(keyword['node'], [])
        #                        for keyword in relevant_nodes]
        # documents_for_nodes = [arr for sublist in documents_for_nodes for arr in sublist]
        # documents_for_nodes = [self.index[document_idx] for document_idx in documents_for_nodes]

        # documents_for_nodes += summary_docs

        # relevant_docs += self.search_communities(query, topk=topk)

        # documents_for_nodes = relevant_nodes.to_dict('records')
        documents_for_nodes = relevant_nodes

        relevant_docs += documents_for_nodes
        
        return relevant_docs