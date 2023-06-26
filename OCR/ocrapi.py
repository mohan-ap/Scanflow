from io import BytesIO
from flask import Flask, request, jsonify, session
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS    
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
from flask_cors import CORS
from paddleocr import PaddleOCR
import tempfile
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from pymongo import MongoClient
import faiss
import secrets

load_dotenv()  # take environment variables from .env.

app = Flask(__name__)
cors = CORS(app)
app.secret_key = secrets.token_hex(16)

client = MongoClient('mongodb://localhost:27017')
db = client['qa']
collection=db['files']

os.environ["OPENAI_API_KEY"] =os.getenv("OPEN_API_KEY")
ocr = PaddleOCR(use_angle_cls=True,use_gpu=False, lang='en')

question_answers = []
docsearch = None
chain = None
uploaded_pdf_data = None


@app.route("/", methods=["POST"])
def process_request():
    global docsearch, chain, uploaded_pdf_data

    if 'pdf' in request.files:

        pdf_file = request.files["pdf"]
        uploaded_file_name = pdf_file.filename
        session['session_id'] = secrets.token_hex(16)
        session_id = session['session_id']
        collection.insert_one({'session_id': session_id, 'file_name': uploaded_file_name})
        uploaded_pdf_data = pdf_file.read()

        if not uploaded_pdf_data:
            return jsonify({'error': 'Empty file'})
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(uploaded_pdf_data)
            pdf_path = temp_pdf.name

        pdf_reader = PdfReader(BytesIO(uploaded_pdf_data))
        raw_text = ""
        extracted_text = []
        for page_num in range(len(pdf_reader.pages)):
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, output_folder=temp_dir, first_page=page_num+1, last_page=page_num+1)
                image = np.array(images[0])
                image_path = f"page_{page_num+1}.jpg"
                Image.fromarray(image).save(image_path)
                result = ocr.ocr(image, cls=True)
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        content = line[1][0]
                        content = content.replace("'", "")
                        extracted_text.append(content)
        raw_text = ' '.join(extracted_text)
        text_file_path = "raw_text.txt"  

        with open(text_file_path, "w") as text_file:
            text_file.write(raw_text)

        print("Raw text has been stored as a text file.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

        texts = splitter.create_documents([raw_text])

        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_documents(texts, embeddings)

        uploaded_filename=pdf_file.filename

        return jsonify({'message': 'File uploaded successfully', 'file_name':'{}'.format(uploaded_filename)})

    elif 'question' in request.form:
        if docsearch is None:
            return jsonify({'error': 'File not uploaded'})

        question = request.form["question"]
        
        docs = docsearch.similarity_search(question, k=4)
        print(docs)
        # docs_page_content = " ".join([d.page_content for d in docs])
        # template = """
        #   You are a helpful assistant that can answer questions about Uploaded file
        #   based on the uploaded file contents: {docs}

        #   Only use the factual information from the uploaded file content to answer the question.

        #   If you feel like you don't have enough information to answer the question, say "I don't know".

        #   Your answers should be short.
        # """
        # system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        # human_template = "answer the following question: {question}"
        # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            
        # chat_prompt = ChatPromptTemplate.from_messages(
        #  [system_message_prompt, human_message_prompt]
        # )  
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        # chain = LLMChain(llm=OpenAI(), prompt=chat_prompt)    
        # answer = chain.run(question=question, docs=docs_page_content)
        answer = chain.run(input_documents=docs, question=question)

        question_answers.append((question, answer))

        return jsonify({
            'question': question,
            'answer': answer,
            'question_answers': question_answers
        })

    else:
        return jsonify({'error': 'Invalid request'})


if __name__ == "__main__":
    app.run(debug=True)
