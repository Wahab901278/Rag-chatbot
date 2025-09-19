import faiss
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:


    def __init__(self,model_name='all-MiniLM-L6-v2',vector_db_path="vector_db"):
        self.model_name=model_name
        self.vector_db_path=vector_db_path
        self.embedding_model=SentenceTransformer(model_name)
        self.dimension=self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)

        self.documents=[]
        self.ensure_vector_db_directory()

    def ensure_vector_db_directory(self):
        if not os.path.exists(self.vector_db_path):
            os.makedirs(self.vector_db_path)
            logger.info(f"Created Vector database directory: {self.vector_db_path}")

    def add_documents(self,documents):
        if not documents:
            logger.warning('No Documents to add to Vector Store')
            return
        texts=[]
        for doc in documents:
            texts.append(doc['content'])
        logger.info('Generating Embeddings....')

        embeddings=self.embedding_model.encode(texts,show_progress_bar=True)

        embeddings=embeddings/np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index.add(embeddings.astype('float32'))

        self.documents.extend(documents) # Metadata storage

        logger.info(f'Added {len(documents)} documents to vector storage')

    def search(self,query,k=5): # k represents that it is gonna look for the top k documentsd
        if self.index.ntotal==0:
            logger.warning("Vector store si empty")

            return []

        query_embedding=self.embedding_model.encode([query])
        query_embedding=query_embedding/np.linalg.norm(query_embedding, axis=1, keepdims=True)

        scores,indices=self.index.search(query_embedding.astype('float32'),k)

        results=[]

        for i, (score,idx) in enumerate(zip(scores[0],indices[0])):
            if idx < len(self.documents):
                doc=self.documents[idx].copy()
                doc['similarity_score']=score
                doc['rank']=i+1
                results.append(doc)
        return results
    

    def save_index(self):
        try:
            index_path=os.path.join(self.vector_db_path,"faiss_index.index")
            faiss.write_index(self.index,index_path)

            docs_path=os.path.join(self.vector_db_path,"documents.pkl")

            with open(docs_path,'wb') as f:
                pickle.dump(self.documents,f)

            logger.info("Vector Store Saved Successfully")
        except Exception as e:
            logger.error(f"Error saving Vector Store: {str(e)}")

    def load_index(self):
        try:
            index_path=os.path.join(self.vector_db_path,"faiss_index.index")
            docs_path=os.path.join(self.vector_db_path,"documents.pkl")

            if os.path.exists(index_path) and os.path.exists(docs_path):
                self.index= faiss.read_index(index_path)

                with open(docs_path,'rb') as f:
                    self.documents=pickle.load(f)

                logger.info(f"Loaded Vector store with {len(self.documents)} documents")

            else:
                logger.info("No Existing vector strore found")

        except Exception as e:
            logger.error(f"Error Loading Vector store : {str(e)}")
            self.index = faiss.IndexFlatIP(self.dimension)

            self.documents=[]

    def get_stats(self):
        return {
            'total_documents' : len(self.documents),
            'index_size': self.index.ntotal,
            'dimension': self.dimension
        }