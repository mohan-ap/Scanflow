from io import BytesIO
from flask import Flask, request, jsonify, session
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS    
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from flask_cors import CORS
from paddleocr import PaddleOCR
import tempfile
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import secrets
from werkzeug.utils import secure_filename
from s3 import s3, BUCKET_NAME 
from mongo import files, store_message, retrieve_conversation

load_dotenv()  # take environment variables from .env.

app = Flask(__name__)
cors = CORS(app)
app.secret_key = "123"


os.environ["OPENAI_API_KEY"] =os.getenv("OPEN_API_KEY")
ocr = PaddleOCR(use_angle_cls=True,use_gpu=False, lang='en')

docsearch = None
chain = None
uploaded_pdf_data = None
embeddings = OpenAIEmbeddings()

@app.route("/", methods=["POST"])
def process_request():
    global docsearch, chain, uploaded_pdf_data

    if 'pdf' in request.files:

        pdf_file = request.files["pdf"]
        uploaded_file_name = pdf_file.filename
        uploaded_pdf_data = pdf_file.read()

        if not uploaded_pdf_data:
            return jsonify({'error': 'Empty file'})

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:   #The with statement ensures that the file is automatically closed when the block of code is exited, regardless of any exceptions
            temp_pdf.write(uploaded_pdf_data)
            pdf_path = temp_pdf.name         #name attribute of the temp_pdf object returns the path of the temporary file on the filesystem

        s3.upload_file(
            Bucket=BUCKET_NAME,
            Filename=temp_pdf.name,
            Key=uploaded_file_name
        )   
        session['session_id'] = secrets.token_hex(16)                                   
        session_id = session['session_id']
        files.insert_one({'session_id': session_id, 'file_name': uploaded_file_name})
        
        pdf_reader = PdfReader(BytesIO(uploaded_pdf_data))
        raw_text = ""
        extracted_text = []
        for page_num in range(len(pdf_reader.pages)):
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, output_folder=temp_dir, first_page=page_num+1, last_page=page_num+1)
                image = np.array(images[0])
                # image_path = f"page_{page_num+1}.jpg"
                # Image.fromarray(image).save(image_path)
                result = ocr.ocr(image, cls=True)
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        content = line[1][0]
                        extracted_text.append(content)
        raw_text = ' '.join(extracted_text)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

        texts = splitter.create_documents([raw_text])

        docsearch = FAISS.from_documents(texts, embeddings)
        docsearch.save_local(f"{uploaded_file_name}")   

        return jsonify({'message': 'File uploaded successfully', 'file_name':'{}'.format(uploaded_file_name)})
    
    elif 'image' in request.files:
            image_file = request.files["image"]
            uploaded_file_name = image_file.filename
            uploaded_image_data = image_file.read()

            if not uploaded_image_data:
                return jsonify({'error': 'Empty file'})

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
                temp_image.write(uploaded_image_data)
                image_path = temp_image.name

            s3.upload_file( 
                Bucket=BUCKET_NAME,
                Filename=temp_image.name,
                Key=uploaded_file_name
            )   
            session['session_id'] = secrets.token_hex(16)                                   
            session_id = session['session_id']
            files.insert_one({'session_id': session_id, 'file_name': uploaded_file_name})
            image = Image.open(image_path).convert("RGB")
            result = ocr.ocr(np.array(image), cls=True)
            extracted_text = [line[1][0] for res in result for line in res]
            raw_text = ' '.join(extracted_text)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

            texts = splitter.create_documents([raw_text])

            docsearch = FAISS.from_documents(texts, embeddings)
            docsearch.save_local(f"{uploaded_file_name}")

            return jsonify({'message': 'image uploaded successfully', 'file_name': '{}'.format(uploaded_file_name)})


@app.route("/ask", methods=["POST"])
def question_request():
    session_id = session.get('session_id')
    print(session_id)
    result = files.find_one({'session_id': session_id})
    file_name = result['file_name']
    new_db = FAISS.load_local(f"{file_name}",embeddings=embeddings) 

    if 'question' in request.form:
        if new_db is None:
            return jsonify({'error': 'File not uploaded'})

        question = request.form["question"]
        
        docs = new_db.similarity_search(question, k=4)
        # print(docs)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")
    
        answer = chain.run(input_documents=docs, question=question)

        user_id = "12345"  #we will modify with user id
        store_message(user_id, question)
        store_message(user_id, answer)
        conversation = retrieve_conversation(user_id)

        return jsonify({
            'question': question,
            'answer': answer,
            'question_answers': conversation
        })

if __name__ == "__main__":
    app.run(debug=True)
