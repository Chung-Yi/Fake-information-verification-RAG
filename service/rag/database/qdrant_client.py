import time
from qdrant_client import QdrantClient
from service.rag.database.data_process import *
from configure.config_loader import config

class Singleton(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instance[cls] = instance
        else:
            instance = cls._instance[cls]
            if hasattr(cls, '__allow_reinitialization') and cls.__allow_reinitialization:
                instance.__init__(*args, **kwargs)

        return instance
    
class BaseQdrantClient(QdrantClient, metaclass=Singleton):

    def __init__(self, url, api_key):
        # self.url = url
        # self.api_key = api_key
        super().__init__(url=url, api_key=api_key)

    def upload(self, data_dict, ids, vectors, payloads):

        batch_size = int(config.get_data_process_parameter("BATCH_SIZE"))
        count = int(config.get_data_process_parameter("COUNT"))

        for i in range(0, len(data_dict), batch_size):
            batch = batch_data(ids, vectors, payloads, i, min(len(data_dict), batch_size+i))
            # batch = batch_data(chunk_ids, chunk_vectors, chunk_payloads, i, min(len(data_dict), BATCH_SIZE+i))

            for c in range(count):
                try:

                    self.upsert(
                        collection_name=config.get_database_parameter("COLLECTION_NAME"),
                        points = batch,
                        # points=[
                        #     models.PointStruct(
                        #         id=1,
                        #         vector=[0.05, 0.61, 0.76, 0.74],
                        #         payload={
                        #             "city": "Berlin",
                        #             "price": 1.99,
                        #         },
                        #     ),
                        #     models.PointStruct(
                        #         id=2,
                        #         vector=[0.19, 0.81, 0.75, 0.11],
                        #         payload={
                        #             "city": ["Berlin", "London"],
                        #             "price": 1.99,
                        #         },
                        #     ),
                        #     models.PointStruct(
                        #         id=3,
                        #         vector=[0.36, 0.55, 0.47, 0.94],
                        #         payload={
                        #             "city": ["Berlin", "Moscow"],
                        #             "price": [1.99, 2.99],
                        #         },
                        #     ),
                        # ],
                        # wait=True
                    
                        
                    ),
                    print(f"Batch {i//batch_size + 1} succeeded")
                    break

                except Exception as e:
                    print(f"Batch {i//batch_size + 1} failed on attempt {c + 1} with error: {e}")
                    time.sleep(2 ** c) 