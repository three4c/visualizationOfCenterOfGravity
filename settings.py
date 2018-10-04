import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

JP = os.environ.get("JSON_PATH")
AP = os.environ.get("API_KEY")
AD = os.environ.get("AUTH_DOMAIN")
DB = os.environ.get("DATABASE_URL")
SB = os.environ.get("STORAGE_BUCKET")
