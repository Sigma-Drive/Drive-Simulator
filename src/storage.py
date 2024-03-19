from datetime import datetime
from typing import List

from google.cloud import bigquery
from google.cloud.bigquery import Table
from google.cloud.exceptions import NotFound

from src.models.bounding_box import BoundingBox
from src.models.user import User

client = bigquery.Client()

schema = [
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("age", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("country", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("object_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("x_top_left", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("y_top_left", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("x_bottom_right", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("y_bottom_right", "INTEGER", mode="REQUIRED")
]


def create_table_if_not_exists(project_id: str, dataset_id: str, table_name: str) -> Table:
    table_id = f'{project_id}.{dataset_id}.{table_name}'
    try:
        # Make an API request to check if the table exists
        table = client.get_table(table_id)
        print("Table {} already exists.".format(table_id))
        return table
    except NotFound:
        print("Table {} is not found. Creating now...".format(table_id))
        table = bigquery.Table(table_id, schema=schema)
        # Make an API request to create the table
        table = client.create_table(table)
        print("Table {} created.".format(table_id))
        return table


def insert_data(user: User, objects: List[BoundingBox], table: Table):
    rows_to_insert = []
    for object in objects:
        detection = {
            "name": user.name,
            "user_id": user.id,
            "age": user.age,
            "country": user.country,
            "timestamp": datetime.now().isoformat(),
            "object_name": object.name,
            "x_top_left": object.x_top_left,
            "y_top_left": object.y_top_left,
            "x_bottom_right": object.x_bottom_right,
            "y_bottom_right": object.y_bottom_right
        }
        rows_to_insert.append(detection)
    # Make an API request to insert the data
    if len(rows_to_insert) == 0:
        print("No detection to insert")
    else:
        errors = client.insert_rows_json(table, rows_to_insert)
        if errors == []:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
