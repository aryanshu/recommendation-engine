from sqlalchemy import text
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from sqlalchemy.orm import declarative_base
import yaml
import pandas as pd
import numpy as np
import sklearn
import os

class data_transform:
    def transform(self):
        with open('resources/application.yml', 'r') as file:
            config = yaml.safe_load(file)


        db_host_destination = config['database_destination']['hostname_destination']
        db_name_destination = config['database_destination']['database_destination']
        db_user_destination = config['database_destination']['username_destination']
        db_password_destination = config['database_destination']['password_destination']
        db_port_destination = config['database_destination']['port_id_destination']

        # Source database
        destination_db_url = 'postgresql://'+str(db_user_destination)+':'+str(db_password_destination)+'@'+str(db_host_destination)+':'+str(db_port_destination)+'/'+db_name_destination
        destination_engine = create_engine(destination_db_url)
        DestinationSession = sessionmaker(bind=destination_engine)
        destination_session = DestinationSession()

        required_table = ['user_profile', 'user_interests', 'swipe', 'user_image']

        try:
            inspector = inspect(destination_engine)
            schemas = inspector.get_schema_names()

            for schema in schemas:
                for table_name in inspector.get_table_names(schema='public'):
                    if table_name in required_table:
                        required_table.remove(table_name)
                        query = text(f'SELECT * FROM {table_name}')
                        df = pd.read_sql_query(query, destination_engine)

                        # Export the DataFrame to a CSV file
                        path = 'resources/'+table_name+".csv"
                        df.to_csv(path, index=False)
                        print("data saved successfully: {}".format(path))

        except Exception as e:
            # Rollback the changes if any error occurs
            print(e)
        finally:
            # Close the sessions
            destination_session.close()
            destination_session.close()

