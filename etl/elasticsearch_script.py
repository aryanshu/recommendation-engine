from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from datetime import datetime
from elasticsearch.helpers import bulk
import logging
import yaml

logging.basicConfig(level=logging.DEBUG)

class elasticsearch_script:

    def push_recommendation(self, result):
        with open('resources/application.yml', 'r') as file:
            config = yaml.safe_load(file)

        elastic_host = config['elasticsearch']['elastic_host']
        elastic_port = config['elasticsearch']['elastic_port']

        url = elastic_host+":"+elastic_port
        try:
            es = Elasticsearch(url)
            logging.info(es.info().body)

        except Exception as e:
            logging.error(e)

        try:
            if es.indices.exists:
                es.indices.delete(index='user_data')
            df = pd.read_csv("resources/user_profile.csv")

            df['dob'] = pd.to_datetime(df['dob'])

            # Calculate age based on current date
            current_date = pd.Timestamp('now')
            df['age'] = np.floor_divide((current_date - df['dob']).dt.days, 365)

            mappings = {
                "properties": {
                    "preferance": {"type": "text", "analyzer": "english"},
                    "profile_score": {"type": "text", "analyzer": "standard"},
                    "age": {"type": "integer"},
                    "global": {"type": "text", "analyzer": "standard"},
                    "higher_range": {"type": "integer"},
                    "lower_range": {"type": "integer"},
                    "similar_user":{"type": "keyword"}
                }
            }

            es.indices.create(index="user_data", mappings=mappings)


            similarity_score = []
            for i in range(len(result)):
                user_i=[]
                for j in range(len(result[i])):
                    if i!=j:
                        user_i.append([result[i][j],j+1])
                sorted(user_i, key=lambda x: x[0], reverse=True)
                similarity_score.append(user_i)

            logging.info(similarity_score)

            score_list =[]
            for i in range(len(similarity_score)):
                users_list = []
                for j in range(len(similarity_score[i])):
                    users_list.append(similarity_score[i][j][1])
                score_list.append(users_list)
            logging.info(score_list)

            bulk_data = []

            for i, row in df.iterrows():
                similar_user = score_list[i]
                bulk_data.append(
                    {
                        "_index": "movies",
                        "_id": df["id"][i],
                        "_source": {
                            "preferance": {"type": "text", "analyzer": "english"},
                            "profile_score": {"type": "text", "analyzer": "standard"},
                            "age": {"type": "integer"},
                            "global": {"type": "text", "analyzer": "standard"},
                            "higher_range": {"type": "integer"},
                            "lower_range": {"type": "integer"},
                            "similar_user": similar_user
                        }
                    }
                )

            logging.info(bulk(es, bulk_data))

        except Exception as e:
            logging.error(e)

        finally:
            if es is not None:
                es.close()