import os
import PyPDF2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:

    def __init__(self, document_path):
        self.documents_path=document_path
        self.ensure_documents_directory()
    def ensure_documents_directory(self):
        if not os.path.exists(self.document_path):
            os.makedirs(self.document_path)

            logger.info(f"Created Documents directory:{self.documents_path}")

    def load_documents(self):
        documents=[]
        if not os.path.exists(self.document_path):
            logger.warning(f'Documents Directory {self.documents_path} does not exist')
            return documents
        for filename in os.listdir(self.documents_path):
            filepath=os.path.join(self.documents_path,filename)
            if filename.lower().endswith('.pdf'):
                text=self.extract_text_from_pdf(filepath)
            elif filename.lower().endswith('.txt'):
                text=self.extract_text_from_txt(filepath)
            else:
                logger.warning(f'Unsupported file content')
                continue
            if text.strip():
                documents.append({
                    'content':text,
                    'source':filename,
                    'type':'document'

                })
                logger.info(f"Loaded Document: {filename}")

        return documents

    def extract_text_from_pdf(filepath):
        try:
            with open(filepath,'rb') as file:
                pdf_reader=PyPDF2.PdfReader(file)
                text=""
                for page in pdf_reader.pages:
                    text += page.extract_text()+'\n'
                return text
        except Exception as e:
            logger.error(f"Error reading file {filepath} : {str(e)}")
            return ""
    def extract_text_from_txt(filepath):
        try:
            with open(filepath,'r',encoding="UTF-8") as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {filepath} : {str(e)}")
            return ""
    def chunk_documents(documents,chunk_size=1000,overlap=200):
        try:
            chunks=[]

            for doc in documents:
                text=doc['content']

                source=doc['source']
                for i in range(0,len(text),chunk_size-overlap):
                    chunk_text=text[i:i+chunk_size]
                    if chunk_text.strip():
                        chunks.append({
                            'content':chunk_text,
                            'source':source,
                            'chunk_id':len(chunks),
                            'type':'chunk'
                        })      
            logger.info(f'Created {len(chunks)} chunks form {len(documents)} documents')

            return chunks
        except Exception as e:
            logger.info('Error Creating Chunk')
            return 
        


