import ast

from typing import List, Dict

from concept_extractor.text_chunker import DataChunk

from utils.custom_llm import CustomLLMBuilder
from utils.llm_pipeline import AsyncLLMPipelineBuilder
from utils.validators import StatementValidator

class ConceptExtractor:

    def __init__(self,
                 chunks: List[DataChunk],
                 comprehensive:bool=False) -> None:
        """
        Given a list of 'chunks' of text, extracts concepts from the text using an LLM model
        A chunk is a DataChunk object that contains the text and an id

        args:
            - chunks : List[DataChunk] : a list of DataChunk objects containing the text to extract concepts from
            - comprehensive : bool : if True, the concepts extracted will be more comprehensive by using an API-access LLM model. If False, the concepts will be extracted using a local LLM model
        """
        
        self.chunks = chunks
        self.concepts = []

        self.build_comprehensive_graph = comprehensive

    def _parse_results(self,
                       results: List[str]) -> List[Dict[str, str]]:
        """
        Parse the results from the LLM model into a list of dictionaries

        args:
            - results : List[str] : list of strings containing the results from the LLM model

        returns:
            - concepts : List[Dict[str, str]] : list of dictionaries containing the extracted concepts
        """

        concepts = []
        error_count = 0

        for idx, result in enumerate(results):

            try:
                result = result.replace("output:", "").replace("```python", "").replace("```", "").replace("json", "").strip()
                result = ast.literal_eval(result)

                #add the text the result was generated from
                if isinstance(result, list):
                    for res in result:
                        res["originating_text"] = self.chunks[idx].text
                        res["originating_chunk_id"] = self.chunks[idx].id

                concepts.extend(result)

            except Exception as e:
                error_count += 1
                print(f"Error parsing result for chunk {self.chunks[idx].id} : {e}")
                print(f"Result : {result}")

        print(f"Error count : {error_count}")

        return concepts

    def extract_concepts(self) -> None:
        """
        Extract concepts from the given text chunks using an LLM model. The behaviour (system prompt) and results are both prompted appropriately to extract 'concepts' in this method
        The extracted concepts are stored in the 'concepts' attribute of the class

        returns:
            - None
        """

        system_prompt = "You are an expert relationship and entity extractor, using these to build a concept ontology. You are to extract entity pairs that share a relationship in the given input text. An entity can be a single word, a term, concept, organization, brand name, product, idea, action, reaction, emotion, etc. Your task is to extract the key terms and concepts from the given context only."
        system_prompt += f"Given a text, return a list of concepts that are related in the text and the relationships between them. The output should be a simple list of dictionaries, where each dictionary has keys 'node_1': <term/concept extracted from the input>, 'node_2' : <term/concept that is related to node_1>, 'edge': <explanation of how node_1 and node_2 are related in the input using one or two sentences>, 'weight' : <number from 1-5 to depict how strongly related node_1 and node_2 are>, 'relation': <relationship between node_1 and node_2>. Note that the output should be a valid list of dictionaries parse-able by ast.literal_eval()"
        system_prompt += f"\nIf there are no concepts to be extracted from input, return an empty list"

        GraphBuilder = CustomLLMBuilder

        if self.build_comprehensive_graph:

            GraphBuilder = AsyncLLMPipelineBuilder
            print(f"Using the comprehensive graph builder for extracting concepts...")

        with GraphBuilder(system_prompt=system_prompt) as llm:

            user_prompts = [f"extract concepts/terms in the list of dictionaries format described above from this text ```{chunk.text}``` \n\n output: "
                            for chunk in self.chunks]

            extracted_concepts = llm.batch_predict(user_prompts)

        self.concepts = self._parse_results(extracted_concepts)
    
    def get_output(self) -> List[Dict[str, str]]:
        """
        Get the extracted concepts using the extract_concepts() method

        returns:
            - concepts : List[Dict[str, str]] : a list of dictionaries containing the extracted concepts. Each dictionary contains the keys 'node_1', 'node_2', 'relation' and 'originating_text'
        """
        
        self.extract_concepts()

        return self.concepts

class ConceptValidator:

    def __init__(self, concepts:List[Dict[str, str]]):
        """
        Given a list of 'concepts' that have been extracted from text, this class will validate the concepts using a language model

        args:
            concepts : List[Dict[str, str]] : a list of dictionaries containing the extracted concepts. Each dictionary should have the keys 'node_1', 'node_2', 'relation' and 'originating_text'

        returns:
            - None
        """

        self.concepts = concepts

    def validate(self) -> List[Dict[str, str]]:
        """
        Validates the concepts using the StatementValidator class. Forms statements that can be answered as 'Yes' or 'No', then validates them using a language model

        returns:
            - concepts : List[Dict[str, str]] : all input concepts with two new keys 'valid' and 'explanation' added to each dictionary. 'valid' is a boolean indicating if the concept is valid or not, 'explanation' is a string containing the explanation for the validation
        """

        statements = [f"`{concept.get('node_1', None)}` and `{concept.get('node_2')}` share the relationship `{concept.get('relation', None)}` in the context of ```{concept.get('originating_text', None)}```"
                        for concept in self.concepts]
        
        validator = StatementValidator(statements = statements)
        evaluated_statements = validator.validate()

        invalid_statements = []

        for idx, concept in enumerate(self.concepts):

            concept["valid"] = False
            concept["explanation"] = None

            try:
                concept["valid"] = evaluated_statements[idx]["valid"]
                concept["explanation"] = evaluated_statements[idx]["explanation"]

            except:
                pass

            if not concept["valid"]:
                invalid_statements.append(concept)

        print(f"Starting secondary evaluation on {len(invalid_statements)} potentially invalid concepts...\n")

        #validate edges for the concepts that are invalid
        statements = [f"`{concept.get('relation', None)}` is a insight from the text `{concept.get('originating_text', None)}`"
                        for concept in invalid_statements]
        
        validator = StatementValidator(statements = statements)
        evaluated_statements = validator.validate()

        for idx, concept in enumerate(invalid_statements):
                
            concept["valid"] = False
            concept["explanation"] = None

            try:
                concept["valid"] = evaluated_statements[idx]["valid"]
                concept["explanation"] = evaluated_statements[idx]["explanation"]

            except:
                pass
        
        #update the concepts list
        self.concepts = [concept for concept in self.concepts if concept["valid"]] + invalid_statements

        return self.concepts