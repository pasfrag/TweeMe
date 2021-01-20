from pymongo import MongoClient
import pandas as pd


class MongoHandler(object):
    client = None
    db = None

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["thesis_db"]

    def mongo_to_df(self, col_name):
        collection = self.db[col_name]
        cursor = collection.find(no_cursor_timeout=True)
        df = pd.DataFrame(list(cursor))
        return df
