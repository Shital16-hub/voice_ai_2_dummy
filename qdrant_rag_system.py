# qdrant_rag_system.py - FINAL PERFORMANCE FIX
"""
Ultra-Fast Qdrant RAG System for LiveKit Telephony with Local Docker
FINAL FIX: More aggressive caching with similarity-based embedding cache
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import uuid
import requests
import hashlib
import difflib

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchParams, OptimizersConfigDiff,
    HnswConfigDiff
)
import openai
from sentence_transformers import SentenceTransformer

from config import config

logger = logging.getLogger(__name__)

class QdrantRAGSystem:
    """
    Ultra-fast Qdrant RAG system optimized for telephony with local Docker
    FINAL FIX: Aggressive similarity-based caching to minimize API calls
    """
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.aclient: Optional[AsyncQdrantClient] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.ready = False
        
        # 🚀 SEARCH RESULT CACHE
        self.cache = {}
        self.max_cache_size = 100
        
        # 🚀 AGGRESSIVE EMBEDDING CACHE FOR MASSIVE SPEED IMPROVEMENT
        self.embedding_cache = {}
        self.embedding_query_cache = {}  # Maps similar queries to existing embeddings
        self.max_embedding_cache_size = config.embedding_cache_size
        
        # 🚀 PRE-COMPUTED EMBEDDINGS FOR COMMON QUERIES
        self.common_query_embeddings = {}
        
        self.local_mode = True
        
    async def initialize(self) -> bool:
        """Initialize the Qdrant RAG system with Docker health checks"""
        try:
            start_time = time.time()
            
            # Check Docker health first
            if not await self._check_docker_health():
                logger.error("❌ Qdrant Docker container not healthy")
                return False
            
            # Initialize clients with local optimizations
            await self._init_clients()
            
            # Initialize embedding model
            await self._init_embedding_model()
            
            # Setup collection with local optimizations
            await self._setup_collection()
            
            # Load existing data if available
            await self._load_existing_data()
            
            # 🚀 WARM UP CACHE FOR BETTER PERFORMANCE
            await self._warm_up_cache()
            
            self.ready = True
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"✅ OPTIMIZED Qdrant RAG initialized in {elapsed:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Qdrant RAG initialization failed: {e}")
            return False
    
    async def _check_docker_health(self) -> bool:
        """Check if Qdrant Docker container is healthy using correct endpoint"""
        try:
            for attempt in range(config.docker_health_check_retries):
                try:
                    # Use root endpoint instead of /health
                    response = requests.get(f"{config.qdrant_url}/", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Qdrant Docker container is healthy")
                        return True
                except requests.exceptions.RequestException:
                    if attempt < config.docker_health_check_retries - 1:
                        logger.warning(f"🔄 Qdrant health check attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(1)
                    
            logger.error("❌ Qdrant Docker container health check failed")
            logger.error("   Please run: docker-compose up -d")
            return False
            
        except Exception as e:
            logger.error(f"❌ Docker health check error: {e}")
            return False
    
    async def _init_clients(self):
        """Initialize Qdrant clients with local Docker optimizations"""
        try:
            # Simple HTTP clients (no gRPC complications)
            self.client = QdrantClient(
                url=config.qdrant_url,
                timeout=config.qdrant_timeout
            )
            
            self.aclient = AsyncQdrantClient(
                url=config.qdrant_url,
                timeout=config.qdrant_timeout
            )
            
            # OpenAI async client
            self.openai_client = openai.AsyncOpenAI(
                api_key=config.openai_api_key
            )
            
            logger.info("✅ Optimized Qdrant clients initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            raise
    
    async def _init_embedding_model(self):
        """Initialize embedding model"""
        try:
            if config.embedding_model.startswith("text-embedding"):
                logger.info("✅ Using OpenAI embeddings with caching")
            else:
                self.embedding_model = await asyncio.to_thread(
                    SentenceTransformer,
                    config.embedding_model
                )
                logger.info(f"✅ Using local embedding model: {config.embedding_model}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    async def _setup_collection(self):
        """Setup optimized Qdrant collection for local Docker with FIXED dimensions"""
        try:
            collection_name = config.qdrant_collection_name
            
            # Check if collection exists
            collections = await asyncio.to_thread(
                self.client.get_collections
            )
            
            collection_exists = any(
                col.name == collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                # Create collection with FIXED 1536 dimensions for OpenAI
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # FIXED: Always use 1536 for OpenAI embeddings
                        distance=Distance.COSINE,
                        # Optimized HNSW for local Docker
                        hnsw_config=HnswConfigDiff(
                            m=16,
                            ef_construct=200,
                            full_scan_threshold=10000,
                            max_indexing_threads=0,
                        )
                    ),
                    # Optimize storage for local telephony
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=1,
                        max_segment_size=None,
                        memmap_threshold=0,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=1
                    )
                )
                logger.info(f"✅ Created collection with FIXED 1536 dimensions: {collection_name}")
            else:
                logger.info(f"✅ Using existing collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup collection: {e}")
            raise
    
    async def _load_existing_data(self):
        """Load existing data from data directory"""
        try:
            data_dir = config.data_dir
            if not data_dir.exists():
                logger.info("📁 No data directory found, skipping data load")
                return
            
            # Check if collection has data
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                config.qdrant_collection_name
            )
            
            if collection_info.points_count > 0:
                logger.info(f"✅ Collection has {collection_info.points_count} existing points")
                return
            
            # Load and index data
            documents = await self._load_documents_from_directory(data_dir)
            if documents:
                await self.add_documents(documents)
                logger.info(f"✅ Loaded {len(documents)} documents into collection")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing data: {e}")
    
    async def _load_documents_from_directory(self, data_dir: Path) -> List[Dict[str, Any]]:
        """Load documents from data directory"""
        documents = []
        
        # Load JSON files
        for json_file in data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    for key, value in data.items():
                        documents.append({
                            "id": f"{json_file.stem}_{key}",
                            "text": str(value),
                            "metadata": {
                                "source": str(json_file),
                                "category": key,
                                "type": "json_entry"
                            }
                        })
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        content = str(item.get("content", item) if isinstance(item, dict) else item)
                        documents.append({
                            "id": f"{json_file.stem}_{i}",
                            "text": content,
                            "metadata": {
                                "source": str(json_file),
                                "category": "list_item",
                                "type": "json_list"
                            }
                        })
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {json_file}: {e}")
        
        # Load text files
        for txt_file in data_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        chunks = self._chunk_text(content)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                "id": f"{txt_file.stem}_chunk_{i}",
                                "text": chunk,
                                "metadata": {
                                    "source": str(txt_file),
                                    "category": "document",
                                    "type": "text_chunk"
                                }
                            })
                            
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {txt_file}: {e}")
        
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks optimized for telephony"""
        if len(text) <= config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + config.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + config.chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - config.chunk_overlap
        
        return [c for c in chunks if c.strip()]
    
    def _get_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding"""
        # Normalize text and create hash for consistent caching
        normalized = text.lower().strip()[:200]  # First 200 chars
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _find_similar_query(self, query: str, threshold: float = 0.8) -> Optional[str]:
        """🚀 NEW: Find similar cached query to avoid API call"""
        normalized_query = query.lower().strip()
        
        # Check for exact matches first
        if normalized_query in self.embedding_query_cache:
            return self.embedding_query_cache[normalized_query]
        
        # Check for similar queries using fuzzy matching
        for cached_query in self.embedding_query_cache.keys():
            similarity = difflib.SequenceMatcher(None, normalized_query, cached_query).ratio()
            if similarity >= threshold:
                logger.debug(f"⚡ Similar query found: {similarity:.2f} similarity")
                return self.embedding_query_cache[cached_query]
        
        return None
    
    async def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text"""
        try:
            if config.embedding_model.startswith("text-embedding"):
                response = await self.openai_client.embeddings.create(
                    model=config.embedding_model,
                    input=text[:8000]
                )
                return response.data[0].embedding
            else:
                embedding = await asyncio.to_thread(
                    self.embedding_model.encode,
                    text,
                    show_progress_bar=False
                )
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"❌ Failed to create embedding: {e}")
            raise
    
    async def _create_embedding_cached(self, text: str) -> List[float]:
        """🚀 SUPER AGGRESSIVE embedding caching - Major performance improvement"""
        if not config.enable_embedding_cache:
            return await self._create_embedding(text)
        
        # Get cache key
        cache_key = self._get_embedding_cache_key(text)
        normalized_query = text.lower().strip()
        
        # Check direct cache hit
        if cache_key in self.embedding_cache:
            logger.debug("⚡ Direct embedding cache hit!")
            return self.embedding_cache[cache_key]
        
        # 🚀 NEW: Check for similar queries to avoid API call
        similar_cache_key = self._find_similar_query(normalized_query)
        if similar_cache_key and similar_cache_key in self.embedding_cache:
            logger.debug("⚡ Similar query embedding cache hit!")
            # Store this query mapping for future use
            self.embedding_query_cache[normalized_query] = similar_cache_key
            return self.embedding_cache[similar_cache_key]
        
        # Create embedding (API call required)
        start_time = time.time()
        embedding = await self._create_embedding(text)
        api_time = (time.time() - start_time) * 1000
        
        logger.debug(f"📡 OpenAI API call: {api_time:.1f}ms")
        
        # Cache management - Remove oldest 20% if cache is full
        if len(self.embedding_cache) >= self.max_embedding_cache_size:
            oldest_keys = list(self.embedding_cache.keys())[:int(self.max_embedding_cache_size * 0.2)]
            for key in oldest_keys:
                del self.embedding_cache[key]
        
        # Store in cache
        self.embedding_cache[cache_key] = embedding
        self.embedding_query_cache[normalized_query] = cache_key
        
        return embedding
    
    async def _warm_up_cache(self):
        """🔥 Pre-warm cache with common telephony queries"""
        if not config.enable_embedding_cache:
            return
        
        common_queries = [
            "towing service", "battery help", "membership", "pricing", 
            "cost", "emergency", "roadside assistance", "tire change",
            "fuel delivery", "jump start", "lockout service", "business hours",
            "contact information", "services offered", "location address",
            "phone number", "email contact", "appointment booking",
            # Add variations that users might say
            "how much does it cost", "what services do you offer",
            "do you tow cars", "can you help with my battery",
            "what are your prices", "emergency help"
        ]
        
        logger.info("🔥 Warming up embedding cache...")
        start_time = time.time()
        
        for query in common_queries:
            try:
                await self._create_embedding_cached(query)
            except Exception as e:
                logger.warning(f"Cache warm-up failed for '{query}': {e}")
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"✅ Cache warmed with {len(self.embedding_cache)} embeddings in {elapsed:.1f}ms")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to Qdrant collection with optimized embedding creation"""
        try:
            points = []
            
            for doc in documents:
                embedding = await self._create_embedding_cached(doc["text"])
                point_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "original_id": doc["id"],
                        **doc.get("metadata", {})
                    }
                )
                points.append(point)
            
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=config.qdrant_collection_name,
                points=points
            )
            
            logger.info(f"✅ Added {len(points)} documents to local Qdrant")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    async def search(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """🚀 ULTRA-FAST search optimized for local Docker with aggressive caching"""
        if not self.ready:
            return []
        
        try:
            start_time = time.time()
            
            # 🚀 STEP 1: Check search result cache first
            cache_key = f"{query.lower().strip()}_{limit or config.search_limit}"
            if cache_key in self.cache:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⚡ FULL CACHE HIT: {elapsed:.1f}ms")
                return self.cache[cache_key]
            
            # 🚀 STEP 2: Use super aggressive cached embedding creation
            query_embedding = await self._create_embedding_cached(query)
            
            # 🚀 STEP 3: Optimized Qdrant search with reduced parameters
            search_result = await asyncio.wait_for(
                self.aclient.search(
                    collection_name=config.qdrant_collection_name,
                    query_vector=query_embedding,
                    limit=limit or config.search_limit,
                    score_threshold=config.similarity_threshold,
                    search_params=SearchParams(
                        hnsw_ef=config.qdrant_hnsw_ef,  # Reduced for speed
                        exact=config.qdrant_exact_search
                    )
                ),
                timeout=config.rag_timeout_ms / 1000.0
            )
            
            # 🚀 STEP 4: Quick result formatting
            results = []
            for point in search_result:
                text = point.payload.get("text", "")
                # Truncate for telephony
                if len(text) > config.max_response_length:
                    text = text[:config.max_response_length] + "..."
                
                results.append({
                    "id": str(point.id),
                    "text": text,
                    "score": float(point.score),
                    "metadata": {
                        k: v for k, v in point.payload.items() 
                        if k != "text"
                    }
                })
            
            # Cache results with size management
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = results
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"⚡ OPTIMIZED search completed in {elapsed:.1f}ms, found {len(results)} results")
            return results
            
        except asyncio.TimeoutError:
            elapsed = (time.time() - start_time) * 1000
            logger.warning(f"⚠️ Search timeout after {elapsed:.1f}ms")
            return []
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "search_cache_size": len(self.cache),
            "search_cache_max": self.max_cache_size,
            "embedding_cache_size": len(self.embedding_cache),
            "embedding_cache_max": self.max_embedding_cache_size,
            "embedding_cache_enabled": config.enable_embedding_cache,
            "query_mapping_cache_size": len(self.embedding_query_cache)
        }
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
            if self.aclient:
                await self.aclient.close()
            logger.info("✅ Optimized Qdrant RAG system closed")
        except Exception as e:
            logger.error(f"❌ Error closing Qdrant RAG system: {e}")

# Global instance
qdrant_rag = QdrantRAGSystem()