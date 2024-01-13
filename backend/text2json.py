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
from langchain.document_loaders import TextLoader
import os
import openai
import sys
import json
import glob


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# def load_text_files_from_directory(directory_path):
#     texts = {}
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(directory_path, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 text = file.read()
#                 texts[filename] = text
#     return texts

def create_text2json(output_filepath, file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )
    splits = text_splitter.split_documents(documents)

    persist_directory = "docs/chroma_text2json/" + os.path.basename(file_path)[:-4]
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding= embedding_function,       #OpenAIEmbeddings(),
        persist_directory=persist_directory
    )

 

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        # memory=memory
    )

    response_schemas = [
    ResponseSchema(name="university_name", description="Mention the name of the university for which the post is created. Do not use any other words. Just mention the university name. Keep it blank if you did not find the expected output."),
    ResponseSchema(name="university_name", description="Mention the name of the professor who is looking for students for this job. Do not use any other words. Keep it blank if you did not find the expected output."),
    ResponseSchema(name="Required_English_proficiency_score", description="It will contain the required english proficiency score (IELTS, TOEFL or DUOLINGO). Just get the exact score. It will be just numbers. IELTS score will be between 1 to 9 and may contain decimal. TOEFL score will be between 1 to 120. DUOLINGO score will be between 10 to 160. Keep it blank if you did not find the expected output."),
    ResponseSchema(name="intended_program", description="It will contain the department name for the recruitment. Example: Computer Science, Data Science, Psychology etc. Keep it blank if you did not find the expected output."),
    ResponseSchema(name="intended_degree", description="It will contain the degree name. It will define what type of student the recruiter is planning to hire. It can be Undergraduated, Master or PhD. Keep it blank if you did not find the expected output."),
    # ResponseSchema(name="scholarship_preference", description="Answer if the user is looking for full scholarship, partial scholarships or no scholarships/ Self fund. Scholarship can be also addressed as funding. Only have one of these values: full, partial, self"),
    ResponseSchema(name="Required_skills", description="It will contain comma separated required skill names to be qualified for the job. For example, the user can have these skills: python, c++, javascript, machine learning etc. Keep it blank if you did not find the expected output."),
    # ResponseSchema(name="publications", description="Mention if the user has any publications or research experience. For example, the user can have 3 conference papers or 2 journal articles."),
    ResponseSchema(name="research_field", description="Mention the research field of the recruiter or the job. If the fields are multiple, put it in a comma separated format. For example, the user's research fields are : machine learning in chemical engineering, cyber security in credit card fraud. Keep it blank if you did not find the expected output."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    template = """query: : {query}.
    {format_instructions}
    
    Do not use external information. keep the value of the keys none if the information is not present in the database.
    Answer: """
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template= template,
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions}
    )
    query = "Extract all these information from the text."
    _input = prompt.format_prompt(query=query)
        
    result = qa_chain({"query": _input.to_string()})

    # answer = {}
    answer = output_parser.parse(result.get("result"))
    if answer:
        answer['text'] = text

    print(answer)

    # output_json_path = "output.json"
    with open(output_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(answer, json_file, ensure_ascii=False, indent=2)
    print(f"Output saved to {output_filepath}")




if __name__ == '__main__':
    data_path = "text_data"
    output_path = "json_data"
    # texts = load_text_files_from_directory(data_path)
    files = glob.glob(data_path + "/*.txt")
    print(files)
    
    for file  in files:
        filename = os.path.basename(file)
        output_filepath = os.path.join(output_path, filename[:-4] + ".json")
        create_text2json(output_filepath, file)


