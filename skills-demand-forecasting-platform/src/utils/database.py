"""Database Module"""
from sqlalchemy import create_engine
import pandas as pd

class DatabaseManager:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)

    def save_dataframe(self, df, table_name):
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)

    def load_dataframe(self, table_name):
        return pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
