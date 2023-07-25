from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from datetime import datetime
from elasticsearch.helpers import bulk
import logging
import yaml

import utils.Constants

logging.basicConfig(level=logging.DEBUG)


class elasticsearch_script:

    def push_recommendation(self, result):
        with open('resources/application.yml', 'r') as file:
            config = yaml.safe_load(file)

        elastic_host = config['elasticsearch']['elastic_host']
        elastic_port = config['elasticsearch']['elastic_port']

        url = elastic_host+":"+elastic_port
        es = None
        try:
            es = Elasticsearch(url)
            logging.info(es.info().body)

        except Exception as e:
            logging.error(e)

        try:
            if es.indices.exists(index='userdata'):
                es.indices.delete(index='userdata')
                print("deleted")

            mappings = {
                "properties": {
                    "preferance": {"type": "text", "analyzer": "english"},
                    "profile_score": {"type": "keyword", "analyzer": "standard"},
                    "age": {"type": "integer"},
                    "global": {"type": "text", "analyzer": "standard"},
                    "higher_range": {"type": "integer"},
                    "lower_range": {"type": "integer"},
                    "lat": {"type": "double"},
                    "long": {"type": "double"},
                    "similar_user": {"type": "integer"}
                }
            }

            es.indices.create(index="userdata", mappings=mappings, body={"settings": {"number_of_shards": 6}})
            print("created shards successfully")
            for k in range(len(result)):
                print("shard no:{}, len:{}".format(k,len(result[k])))
                similarity_score = []
                for i in range(len(result[k])):
                    user_i= []
                    for j in range(len(result[k][i])):
                        if i!= j:
                            user_i.append([result[k][i][j],j+1])
                    sorted(user_i, key=lambda x: x[0], reverse=True)
                    similarity_score.append(user_i)

                # logging.info(similarity_score)

                score_list =[]
                for i in range(len(similarity_score)):
                    users_list = []
                    for j in range(len(similarity_score[i])):
                        users_list.append(similarity_score[i][j][1])
                    score_list.append(users_list)
                # logging.info(score_list)

                bulk_data = []

                path = "resources/"+utils.Constants.df_names[k]+".csv"
                df = pd.read_csv(path)
                df['dob'] = pd.to_datetime(df['dob'])

                # Calculate age based on current date
                current_date = pd.Timestamp('now')
                df['age'] = np.floor_divide((current_date - df['dob']).dt.days, 365)

                for i, row in df.iterrows():
                    similar_user = score_list[i]
                    bulk_data.append(
                        {
                            "_index": "userdata",
                            "_id": df["id"][i],
                            "_source": {
                                "preferance": row["preferance"],
                                "profile_score": row["profile_score"],
                                "age": row["age"],
                                "global": row["global"],
                                "higher_range": row["higher_range"],
                                "lower_range": row["lower_range"],
                                "similar_user": similar_user
                            },
                            "routing": k+1
                        }
                    )

                bulk(es, bulk_data)


        except Exception as e:
            logging.error(e)

        finally:
            if es is not None:
                es.close()