import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    """RAG CHATBOT USING QWEN3-4B-INSTRUCT"""

    def __init__(self,model_name):
        self.model_name=model_name
        self.tokenizer=None
        self.model=None
        self.vector_store=VectorStore()
        self.document_processor=DocumentProcessor()
        self.load_model()
        self.initialize_documents()
    def load_model(self):
        try:
            logger.info(f"Loading model:{self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            device= "cuda" if torch.cuda_is_available() else "cpu"
            logging.info(f'Using Device : {device}')

            self.model= AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
                device_map="auto" if device=="cuda" else None,
                trust_remote_code=True
            )
            if device=='cpu':
                self.model= self.model.to(device)

            logger.info('Model Loaded Successfully')

        except Exception as e:
            logger.error(f"Error loading Model: {str(e)}")
            raise
    def initialize_documents(self):
        try:
            documents=self.document_processor.load_documents()

            if documents:
                chunks=self.document_processor.chunk_documents(documents)
            
                if chunks and self.vector_store.get_stats()['total_documents']==0:
                    self.vector_store.add_documents(chunks)
                    logger.info("Documents Preprocessed and added to vector store")
                else:
                    logger.info("Using existing vector store")
            else:
                logger.info("No documents found in the documents directory")
        except Exception as e:
            logger.error(f"Error Initializing Documents : {str(e)}")
    def retrieve_context(self,query,k=3):

        results=self.vector_store.search(query,k=k)

        if not results:
            return "No relevant context found"
        
        context_parts=[]

        for i , result in enumerate(results):
            context_parts.append(f"Context {i} (from {result['source']}):\n{result['content']}")

        return "\n".join(context_parts)
    
    def generate_response(self,query,context):
        try:
            system_prompt="""
                            You are a Helpful AI assistant. Use the Provided Context to answer the user's question accurately and comprehensively. If the context doesn't contain relevant information, Say so clearly.                        
                           """
            prompt=f"""<|im_start|> system : {system_prompt}<|im_end|>
                        <|im_start|> user Context: {context}
                        Question: {query}<|im_end|>
                        <|im_start|> assistant
                    """
            inputs=self.tokenizer(prompt,return_tensors="pt",truncation=True,max_length=2048)
            device = next(self.model.parameters()).device

            inputs={k: v.to(device)for k, v in inputs.items()}
            with torch.no_grad():
                outputs=self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temprature=0.7
                )
                full_response=self.tokenizer.decode(outputs[0],skip_special_tokens=True)

                assistant_start=full_response.find("<|im_start|> assistant\n")

                if assistant_start != -1 :
                    response=full_response[assistant_start + len("<|im_start|> assistant\n"):].strip()

                else:
                    response= full_response[len(prompt):].strip()

                return response
        except Exception as e:
            logger.error(f"Error generating response: str(e)")
            return f"Sorry I encountered an error while generating a response : { str(e)}"
