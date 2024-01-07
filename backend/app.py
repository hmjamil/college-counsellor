from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os
import openai
import sys
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import uuid


# sys.path.append("../..")


old_db_path = "./docs/chroma"
if os.path.exists(old_db_path):
    os.system(f"rm -rf {old_db_path}")

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            target_directory = "./docs"
            os.makedirs(target_directory, exist_ok=True)
            unique_filename = str(uuid.uuid4()) + ".pdf"
            file_path = os.path.join(target_directory, unique_filename)
            uploaded_file.save(file_path)
            return jsonify({
                "message": "PDF file uploaded and saved successfully",
                "original_filename": uploaded_file.filename,
                "file_path": file_path
            })
        else:
            return jsonify({"error": "No file selected"})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query')

        print(f"Query: {query}")

        uploaded_file_path = data.get('file_path')
        print(uploaded_file_path)
        if uploaded_file_path:
            # Convert to absolute path
            uploaded_file_path = os.path.abspath(uploaded_file_path)
            print(f"Using PDF from: {uploaded_file_path}")
            loader = PyPDFLoader(uploaded_file_path)
            pages = loader.load()
            print("PDF loaded successfully")

        # Load the PDF and perform text processing
        # pdf_file_path = "docs/the-tortoise-and-the-hare-story.pdf"
        # print(f"Attempting to load PDF from: {pdf_file_path}")
        # loader = PyPDFLoader(pdf_file_path)
        # pages = loader.load()

            page_len = len(pages)
            print(page_len)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(pages)
            split_len = len(splits)
            print(split_len)
            print("saving result in chromadb")

            persist_directory = "docs/chroma/"
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding= embedding_function,       #OpenAIEmbeddings(),
                persist_directory=persist_directory
            )

            # memory = ConversationBufferMemory(
            #     memory_key="chat_history",
            #     return_messages=True
            # )

            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
            retriever = vectordb.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=retriever,
                # memory=memory
            )

            response_schemas = [
            ResponseSchema(name="English_profiency_score", description="It will contain the english proficiency score (IELTS, TOEFL or DUOLINGO). Just get the exact score. It will be just numbers. IELTS score will be between 1 to 9 and may contain decimal. TOEFL score will be between 1 to 120. DUOLINGO score will be between 10 to 160."),
            ResponseSchema(name="intented_program", description="It will contain the intended program or major for studying. Example: Computer Science, Data Science, Psychology etc."),
            ResponseSchema(name="intented_degree", description="It will contain the degree name the user is planning to pursue. It can be Undergraduated, Master or PhD."),
            ResponseSchema(name="scholarship_preference", description="Answer if the user is looking for full scholarship, partial scholarships or no scholarships/ Self fund. Scholarship can be also addressed as funding. Only have one of these values: full, partial, self"),
            ResponseSchema(name="skills", description="It will contain comma separated skill names. For example, the user can have these skills: python, c++, javascript, machine learning etc."),
            ResponseSchema(name="publications", description="Mention if the user has any publications or research experience. For example, the user can have 3 conference papers or 2 journal articles."),
            ResponseSchema(name="research_interest", description="Mention the research interests of the user in comma separated format. For example, the user's research interests are : machine learning in chemical engineering, cyber security in credit card fraud"),
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

            template = """query: : {query}.
            Extract all these information from the database.
            {format_instructions}
            
            Do not use external information. keep the value of the keys none if the information is not present in the database.
            Answer: """
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template= template,
                input_variables=["query"],
                partial_variables={"format_instructions": format_instructions}
            )

            _input = prompt.format_prompt(query=query)

            
            # llm_openai = OpenAI(temperature=0)
            # output = llm_openai(_input.to_string())
            # response = output_parser.parse(output)
                    
                    
            result = qa_chain({"query": _input.to_string()})
            answer = output_parser.parse(result.get("result"))
            # answer = result.get("result")
            # answer = "Thisis a pdf file "
            print(answer)

        # similar_documents = vectordb.similarity_search(query, k=3)
        # similar_documents_json = []
        # for document in similar_documents:
        #     document_dict = {
        #         "title": document.title,
        #         "content": document.content,
        # # Include any other relevant fields here
        #     }
        #     similar_documents_json.append(document_dict)

            # conversation_chain = ConversationalRetrievalChain.from_llm(
            #     llm,
            #     retriever=retriever,
            #     memory=memory
            # )

            # conversation_result = conversation_chain({"question": query})
            # conversation_answer = conversation_result["answer"]

            conversation_answer = "Whole conversation"

            response = {
                "result": answer,
                "conversation_result": conversation_answer
                
                # "similar_documents": similar_documents_json
            }

            return jsonify(response)
        else:
            print("Error")
            return jsonify({"error": "No PDF file path provided in the request"})

    except Exception as e:
        print("Error: ", e)
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
