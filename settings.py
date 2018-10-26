import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

JP = os.environ.get("JSON_PATH")
DB = os.environ.get("DATABASE_URL")
