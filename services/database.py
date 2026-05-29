import os
import re
from urllib.parse import quote_plus
from pymongo import MongoClient

_mongo_client = None


def _build_mongo_url(raw_url):
    match = re.match(r"(mongodb\+srv://)([^:]+):([^@]+)@(.+)", raw_url)
    if match:
        prefix, username, password, rest = match.groups()
        return f"{prefix}{quote_plus(username)}:{quote_plus(password)}@{rest}"
    return raw_url


def get_db():
    global _mongo_client
    if _mongo_client is None:
        raw_url = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
        db_name = os.environ.get("DATABASE_NAME", "docuverse")
        url = _build_mongo_url(raw_url)
        _mongo_client = MongoClient(
            url,
            connectTimeoutMS=20000,
            serverSelectionTimeoutMS=30000,
            tls=True,
            tlsAllowInvalidCertificates=True,
        )
    return _mongo_client[os.environ.get("DATABASE_NAME", "docuverse")]
