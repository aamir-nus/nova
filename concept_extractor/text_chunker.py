from uuid import uuid4
from typing import List, Dict

from utils.constants import PUNCTUATIONS

class DataChunk:
    def __init__(self,
                 id: int,
                 text: str,
                 size: int):
        """
        A class to represent a DataChunk.

        args:
            id: UUID - unique identifier for the chunk
            text: str - the text content of the chunk
            size: int - number of words in the chunk
        
        returns:
            None
        """
        self.id = id
        self.text = text
        self.size = size

class BaseTextChunker:

    def __init__(self,
                 docs:List[str]):
        """
        Simple base class to chunk text data into smaller pieces.

        args:
            docs:List[str] - list of strings to be chunked
        
        returns:
            None
        """
        
        self.docs = [str(doc) for doc in docs]
        self.docs = [doc for doc in self.docs
                    if doc.strip() not in [None, 'None', "", 'nan', 'NaN']]

        self.chunks = []

    def chunk_text(self,
                   chunk_size:int=200) -> List[DataChunk]:
        """
        Abstract method to chunk text data into smaller pieces.

        args:
            chunk_size:int - maximum number of words in each chunk. Defaults to 200.

        returns:
            List[DataChunk] - list of DataChunk objects. Each DataChunk object has the following attributes:
                id:(UUID) - unique id for the chunk
                text:(str) - text of the chunk
                size:(int) - number of words in the chunk
        """

        return NotImplementedError

    def get_output(self,
                   chunk_size:int=200) -> List[Dict[str, str]]:
        """
        calls chunk_text() method to chunk text data, returns the chunks formed

        args:
            chunk_size:int - maximum number of words in each chunk. Defaults to 200.
        """

        self.chunks = self.chunk_text(chunk_size=chunk_size)
        
        return self.chunks

class SimpleChunker(BaseTextChunker):

    def __init__(self,
                 docs:List[str]):

        super().__init__(docs=docs)

    def chunk_text(self,
                    chunk_size:int=200) -> List[DataChunk]:
        """
        Method to chunk text data into smaller pieces. The chunks are formed based on the chunk_size parameter as a 'guideline' to prioritize keeping the entire document in the same chunk.
        If the chunk_size is exceeded, the current document is NOT split into two chunks. Instead, the current document is added to the current chunk.

        args:
            chunk_size:int - number of words in each chunk. Defaults to 200.

        returns:
            List[DataChunk] - list of DataChunk objects. Each DataChunk object has the following attributes:
                id:(UUID) - unique id for the chunk
                text:(str) - text of the chunk
                size:(int) - number of words in the chunk
        """

        chunks = []
        current_chunk_size, current_chunk = 0, ""

        for doc in self.docs:

            if current_chunk_size >= chunk_size:

                chunks.append(DataChunk(id=str(uuid4()),
                                        text=current_chunk,
                                        size=len(current_chunk.split(" "))
                                        )
                            )
                
                #reset the current_chunk and current_chunk_size
                current_chunk = ""
                current_chunk_size = 0

            else:

                #add a period at the end of the document if it doesn't end with a punctuation
                if doc.strip()[-1] not in PUNCTUATIONS:
                    doc+". "

                current_chunk += doc
                current_chunk_size += len(doc.strip().split(" "))

        if current_chunk:
            chunks.append(DataChunk(id=str(uuid4()),
                                    text=current_chunk,
                                    size=len(current_chunk.split(" "))
                                    )
                        )
        return chunks
    
class LineChunker(BaseTextChunker):
    
    def __init__(self,
                 docs:List[str]):
        """
        Line/record based data chunking. Each line/record is considered a separate chunk.
        """
        super().__init__(docs=docs)

    def chunk_text(self,
                    chunk_size:int=200) -> List[DataChunk]:
        """
        Method to chunk text data into smaller pieces. Each line/record is considered a separate chunk.

        args:
            chunk_size:int - number of words in each chunk. Defaults to 200. Dummy argument for compatibility with the BaseTextChunker class.
        
        returns:
            List[DataChunk] - list of DataChunk objects. Each DataChunk object has the following attributes:
                id:(UUID) - unique id for the chunk
                text:(str) - text of the chunk
                size:(int) - number of words in the chunk
        """
        
        chunks = []
        for doc in self.docs:
            chunks.append(DataChunk(id=str(uuid4()),
                                    text=doc,
                                    size=len(doc.split(" "))
                                    )
                        )
        return chunks