from etl.data_synchronization import data_synchronization
from etl.data_transform import data_transform
from etl.similarity import similarity
from etl.elasticsearch_script import elasticsearch_script
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    synchronize = data_synchronization()
    synchronize.synchronization()

    transform = data_transform()
    transform.transform()

    cosine_similarity = similarity()
    result = cosine_similarity.user_similarity()
    logging.debug(result)
    print(result)
    pushing_recommendation = elasticsearch_script()
    pushing_recommendation.push_recommendation(result=result)







