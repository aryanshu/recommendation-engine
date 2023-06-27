from sqlalchemy import text
from sqlalchemy import create_engine,MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from sqlalchemy.orm import declarative_base
import yaml
import sqlalchemy as sa

class data_synchronization:
    def synchronization(self):
        with open('resources/application.yml', 'r') as file:
            config = yaml.safe_load(file)

        Base = declarative_base()
        db_host_source = config['database_source']['hostname_source']
        db_name_source = config['database_source']['database_source']
        db_user_source = config['database_source']['username_source']
        db_password_source = config['database_source']['password_source']
        db_port_source = config['database_source']['port_id_source']

        db_host_destination = config['database_destination']['hostname_destination']
        db_name_destination = config['database_destination']['database_destination']
        db_user_destination = config['database_destination']['username_destination']
        db_password_destination = config['database_destination']['password_destination']
        db_port_destination = config['database_destination']['port_id_destination']

        # Source database
        source_db_url = 'postgresql://'+str(db_user_source)+':'+str(db_password_source)+'@'+str(db_host_source)+':'+str(db_port_source)+'/'+db_name_source
        source_engine = create_engine(source_db_url)

        # Destination database
        destination_db_url = 'postgresql://'+str(db_user_destination)+':'+str(db_password_destination)+'@'+str(db_host_destination)+':'+str(db_port_destination)+'/'+db_name_destination
        destination_engine = create_engine(destination_db_url)

        source_connection = source_engine.connect()
        destination_connection = destination_engine.connect()

        # Get a session for both databases
        SourceSession = sessionmaker(bind=source_engine)
        source_session = SourceSession()

        DestinationSession = sessionmaker(bind=destination_engine)
        destination_session = DestinationSession()

        # Get a transaction for the destination database
        destination_trans = destination_session.begin()
        required_table = ['user_profile','user_interests', 'swipe', 'user_image']

        try:
            inspector = inspect(source_engine)
            schemas = inspector.get_schema_names()

            for schema in schemas:
                for table_name in inspector.get_table_names(schema='public'):
                    if table_name in required_table:
                        query = text(f'SELECT * FROM {table_name}')

                        result = source_session.execute(query)
                        rows = result.fetchall()

                        delete_query = text(f'DELETE FROM {table_name}')
                        destination_session.execute(delete_query)

                        metadata = MetaData()
                        target_table = Table(table_name, metadata, autoload_with=destination_engine)

                        for row in rows:
                            new_row_data = {}
                            for column in target_table.columns:
                                column_value = getattr(row, column.name, None)
                                new_row_data[column.name] = column_value

                            new_row = target_table.insert().values(**new_row_data)
                            destination_session.execute(new_row)

            destination_session.commit()
            print("Database copied successfully!")
        except Exception as e:
            # Rollback the changes if any error occurs
            destination_trans.rollback()
            print(e)
        finally:
            # Close the sessions
            source_session.close()
            destination_session.close()
