import os
import google.generativeai as gemini_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
# from service.rag.database.data_struct import SocialMediaData
from service.rag.database.qdrant_client import BaseQdrantClient
from service.rag.database.data_process import *
from configure.config_loader import config

CONFIG_PATH = "configure/configure.ini"


def main():
    
    # intialize qdrant client
    client = BaseQdrantClient(url=config.get_database_parameter("URL"), api_key=config.get_database_parameter("API_KEY"))
    
    # data process
    data = concat_dataset()
    data_dict = data.to_dict(orient='records')
    is_collection_existed = client.collection_exists(collection_name=config.get_database_parameter("COLLECTION_NAME"))

    # client.recreate_collection(collection_name=COLLECTION_NAME, 
    #                             #  vectors_config= vectors_config,
    #                             vectors_config=VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    #                             shard_number=4
    #                             )
   
    # check db is existed
    if not is_collection_existed:


        client.create_collection(collection_name=config.get_database_parameter("COLLECTION_NAME"), 
                                #  vectors_config= vectors_config,
                                vectors_config=VectorParams(size=encoder.encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                                # vectors_config=VectorParams(size=4, distance=Distance.COSINE),
                                shard_number=2
                                )



    ids, vectors, payloads = qdrant_db_format(data_dict)

    # chunk_vectors, chunk_payloads = chunk_content(ids, payloads)
    page_content, metadatas = chunk_content(ids, payloads)

    client.upload(data_dict, ids, vectors, metadatas)
    # client.upload(data_dict, ids, chunk_vectors, chunk_payloads)


if __name__ == "__main__":
    main()