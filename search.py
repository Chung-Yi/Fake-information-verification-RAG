import numpy as np
import os
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from service.rag.database.embedding import encoder
from configure.config_loader import config
from langchain_openai import OpenAIEmbeddings
from service.rag.database.qdrant_client import BaseQdrantClient
from langchain_community.vectorstores import Qdrant
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def main():

    client = BaseQdrantClient(url=config.get_database_parameter("URL"), api_key=config.get_database_parameter("API_KEY"))
    qdrant_store = QdrantVectorStore(client, config.get_database_parameter("COLLECTION_NAME"), embedding=encoder)
    
    query_text = "以色列人躲飛彈畫面"

    

    results = qdrant_store.similarity_search_with_score(query=query_text, score_threshold=0.6)
    
    retriver = qdrant_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k':4}
    )

    prompt_template = """
        你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。注意：請確保答案的準確性。
    """

    system_message = SystemMessagePromptTemplate.from_template("文件: \n\n {documents}")
    human_message = HumanMessagePromptTemplate.from_template("問題: {question}")



    prompt = ChatPromptTemplate.from_messages(
        [
            # ("system", prompt_template.encode('utf-8').strip()),
            # ("system","文件: {documents}"),
            # ("human","問題: {question}"),

            ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{documents}'),
            ('user', 'Question: {question}'),
            # prompt_template.encode('utf-8').strip(),
            # system_message,
            # human_message
        ]
    )

    prompt = PromptTemplate(
        template=prompt_template,
        # input_variables=["question"]
    )

    chain_type_kwargs = {"prompt": prompt}


    # llm and chain
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = Ollama(model='llama3', url='http://localhost:11434')
    print("llm: ", llm)

    os._exit(0)
    rag_chain = prompt | llm | StrOutputParser()


    documents = retriver.invoke(query_text)

    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": query_text})
    print("generation: ", generation)

    # print(len(document))

    # print("document: ", document)
    


  


if __name__ == "__main__":
    main()