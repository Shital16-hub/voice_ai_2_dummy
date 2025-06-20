# performance_tuning_config.py
"""
Apply these changes to your config.py to improve performance based on test results
"""

# Update your config.py with these optimized values:

class QdrantConfig(BaseSettings):
    # ... existing settings ...
    
    # PERFORMANCE IMPROVEMENTS based on test results
    search_limit: int = Field(default=2, env="SEARCH_LIMIT")  # Reduced from 5 to 2 for speed
    similarity_threshold: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")  # Lowered from 0.25 to 0.3
    rag_timeout_ms: int = Field(default=700, env="RAG_TIMEOUT_MS")  # Increased from 1500 to 700ms
    
    # CACHE IMPROVEMENTS
    max_cache_size: int = Field(default=50, env="MAX_CACHE_SIZE")  # Smaller cache for better performance
    
    # EMBEDDING OPTIMIZATIONS
    chunk_size: int = Field(default=150, env="CHUNK_SIZE")  # Smaller chunks for better matching
    chunk_overlap: int = Field(default=25, env="CHUNK_OVERLAP")  # Less overlap


# Also update your .env file with these values:
"""
# Optimized performance settings
SEARCH_LIMIT=2
SIMILARITY_THRESHOLD=0.3
RAG_TIMEOUT_MS=700
MAX_CACHE_SIZE=50
CHUNK_SIZE=150
CHUNK_OVERLAP=25
"""