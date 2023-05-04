import snowflake.connector
from dotenv import load_dotenv
import os

load_dotenv()

conn = snowflake.connector.connect(
    user=os.environ["SF_USER"],
    password=os.environ["SF_PWD"],
    account=os.environ["SF_ACCOUNT"],
    database=os.environ["SF_DB"],
    schema=os.environ["SF_SCHEMA"]
)

# create a cursor object
cursor = conn.cursor()

select_stuff = 'select category, text from hackathon.POST_MODERN_DATA_STACK.INTENT_DATASET limit 10'
cursor.execute(select_stuff)
for (category, text) in cursor: print('{0}, {1}'.format(category, text))
