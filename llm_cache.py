from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os

def setup_sqlite_cache(cache_dir: str = ".cache") -> None:
    """
    Set up SQLite cache for LangChain LLM calls
    
    Args:
        cache_dir (str): Directory to store the SQLite cache file
    """
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Set up SQLite cache
    cache_path = os.path.join(cache_dir, "langchain.db")
    set_llm_cache(SQLiteCache(database_path=cache_path)) 