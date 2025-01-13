import logging
class BaseRetriever:

    def __init__(self, config={}, logger: logging.Logger=None):
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

    def preprocess_document(self, document:str):
        return document
    
    def retrieve_topk(self, query:str, topk:int=5):
        raise NotImplementedError
    
    def build_index(self):
        raise NotImplementedError