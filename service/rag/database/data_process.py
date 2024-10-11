import glob
import os
import uuid
import pandas as pd
from tqdm.auto import tqdm
from qdrant_client.http import models
from configure.config_loader import config
# from sentence_transformers import SentenceTransformer
from service.rag.database.embedding import encoder
from service.rag.database.data_struct import SocialMediaData
from langchain.text_splitter import RecursiveCharacterTextSplitter

# SentenceTransformer for embedding
# encoder = SentenceTransformer("all-MiniLM-L6-v2")

def batch_data(ids, vectors, payloads, start_index, end_index):
    # print("payloads: ", payloads)
    # print("vectors: ", vectors)
    # os._exit(0)
    batch = models.Batch(
        ids = ids[start_index:end_index],
        vectors=vectors[start_index:end_index],
        payloads=payloads[start_index:end_index]

    )
    # print("payloads: ", payloads[0][:])
    # print("vectors: ", vectors[0])
    # os._exit(0)

    # batch = [ models.PointStruct(
    #     id=id,
    #     payload=payloads[i][:],
    #     vector=vectors[i][:]
    # ) for i, id in enumerate(ids)]

    return batch

def get_chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    # print("chunks: ", chunks)
    return chunks


def chunk_content(ids, payloads):
    
    chunk_ids = []
    # chunk_texts = []
    # chunk_vectors = []｀
    vectors = []
    chunk_payloads = []
    metadatas = []
    texts = []
    
    for idx, payload in tqdm(zip(ids, payloads)):
        
        chunk_text = get_chunk_text(payload["content"])

        
        
        # chunk_payload = [
        #             {
        #                 "chunk": i,
        #                 "content": text,
        #                 "subject": payload["subject"],
        #                 "link": payload["link"],
        #                 "time": payload["time"],
        #                 "name": payload["name"],
        #                 "vector": encoder.embed_query(text)
        #             }
        #         for i, text in enumerate(chunk_text)
        #     ]

        metadata = {"id": idx, 
                    "subject": payload["subject"], 
                    "link": payload["link"], 
                    "time": payload["time"],
                    "name": payload["name"],
                    # "content": text,
                    }
        
        record_metadatas = [{
            "chunk": j, "content": text
        } for j, text in enumerate(chunk_text)]

        metadata["content"] = record_metadatas

        chunk_payloads.append({
            "metadata": metadata
        })
        vectors.append(encoder.embed_query(payload["subject"]))
        # chunk_ids.append(str(uuid.uuid4()))
        
        texts.extend(chunk_text)
        metadatas.extend(record_metadatas)

   

    c = []
    for i in range(len(metadatas)):
        c.append(metadatas[i]['content'])
    # print("c: ", c)
    # os._exit(0)
    
    page_content = c
    # return vectors, chunk_payloads
    return page_content, chunk_payloads


    

def read_dataset():
    files = glob.glob(config.get_data_process_parameter("DATA_PATH"))
    dataframe_list = []
    for file in files:
         if "$" not in file:
            data = pd.read_excel(file)
            dataframe_list.append(data)

    return dataframe_list
    

def concat_dataset():
    dataframe_list = read_dataset()
    result = pd.concat(dataframe_list)

    return result

def qdrant_db_format(data_dict):
    ids = [str(uuid.uuid4()) for idx, data in enumerate(data_dict)]
    vectors = [encoder.embed_query(data["subject"]) for idx, data in enumerate(data_dict)]
    data_object = [ SocialMediaData(data, vector) for idx, (data, vector) in enumerate(zip(data_dict, vectors))]
    payloads = [ {
                     "subject": data.subject, 
                     "content": data.content,
                     "link": data.link,
                     "time": data.time,
                     "name": data.name
                 } 
                for idx, data in enumerate(data_object)
               ]
    
    return ids, vectors, payloads