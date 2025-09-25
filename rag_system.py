import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:

    def __init__(self, model_name="Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        self.load_model()
        self.initialize_documents()
    def load_model(self):
        try:
            logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f'Using Device: {device}')

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == 'cpu':
                self.model = self.model.to(device)

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
    
    def generate_response(self, query, context):
        try:

            prompt = f"""Context: {context}

Question: {query}

Answer:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.7
                )
                

                input_length = inputs['input_ids'].shape[1]
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                

                if response.startswith("Answer:"):
                    response = response[7:].strip()
                

                if not response or len(response) < 10:
                    full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    prompt_end = full_response.find("Answer:")
                    if prompt_end != -1:
                        response = full_response[prompt_end + 7:].strip()
                    else:
                        response = full_response[len(prompt):].strip()
                
                return response if response else "I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Sorry I encountered an error while generating a response: {str(e)}"
        
    def chat(self,query):
        try:
            logger.info(f"Processing query: {query}")
            context=self.retrieve_context(query)

            response=self.generate_response(query,context)

            return {
                "query": query,
                "context":context,
                "response": response,
                "status":"success"
            }
        except Exception as e:
            logger.error(f'Error in chat: {str(e)}')
            return {
                "query": query,
                "context":"",
                "response": f"Sorry, I encountered an error : {str(e)}",
                "status":"error"
            }
    def add_document(self, file_path):
        try:
            if file_path.endswith('.pdf'):
                text = self.document_processor.extract_text_from_pdf(file_path)
            elif file_path.endswith('.txt'):
                text = self.document_processor.extract_text_from_txt(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            if not text.strip():
                logger.warning(f"No text content found in {file_path}")
                return False

            document = {
                'content': text,
                'source': os.path.basename(file_path),
                'type': 'document'
            }

            chunks = self.document_processor.chunk_documents([document])
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False

            self.vector_store.add_documents(chunks)
            logger.info(f"Successfully added document: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            return False

