# qdrant_rag_system.py
"""
Ultra-Fast Qdrant RAG System for LiveKit Telephony with Local Docker
FIXED: Optimized search performance and timeout handling
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import uuid
import requests

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
    FIXED: Better timeout handling and search optimization
    """
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.aclient: Optional[AsyncQdrantClient] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.ready = False
        self.cache = {}
        self.max_cache_size = 100  # Increased cache size
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
            
            self.ready = True
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"✅ Local Qdrant RAG initialized in {elapsed:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Qdrant RAG initialization failed: {e}")
            return False
    
    async def _check_docker_health(self) -> bool:
        """FIXED: Check if Qdrant Docker container is healthy using correct endpoint"""
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
            
            logger.info("✅ Simple Qdrant clients initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            raise
    
    async def _init_embedding_model(self):
        """Initialize embedding model"""
        try:
            if config.embedding_model.startswith("text-embedding"):
                logger.info("✅ Using OpenAI embeddings")
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
        """Setup optimized Qdrant collection for local Docker"""
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
                # Create ultra-optimized collection for local Docker
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=config.embedding_dimensions,
                        distance=Distance.COSINE,
                        # Optimized HNSW for local Docker (speed over accuracy)
                        hnsw_config=HnswConfigDiff(
                            m=16,  # Increased for better recall
                            ef_construct=200,  # Increased for better indexing
                            full_scan_threshold=10000,  # Use full scan for small collections
                            max_indexing_threads=0,  # Use all available threads
                        )
                    ),
                    # Optimize storage for local telephony
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=1,  # Single segment for small datasets
                        max_segment_size=None,
                        memmap_threshold=0,  # Disable mmap for speed
                        indexing_threshold=20000,  # Index sooner
                        flush_interval_sec=5,  # Regular flushes
                        max_optimization_threads=1  # Don't over-optimize
                    )
                )
                logger.info(f"✅ Created ultra-optimized local collection: {collection_name}")
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
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to Qdrant collection"""
        try:
            points = []
            
            for doc in documents:
                embedding = await self._create_embedding(doc["text"])
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
        """FIXED: Ultra-fast search optimized for local Docker with better timeout handling"""
        if not self.ready:
            return []
        
        try:
            start_time = time.time()
            
            # Check cache with better cache key
            cache_key = f"{query.lower().strip()}_{limit or config.search_limit}"
            if cache_key in self.cache:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⚡ Local cache hit in {elapsed:.1f}ms")
                return self.cache[cache_key]
            
            # Create query embedding
            query_embedding = await self._create_embedding(query)
            
            # FIXED: Search with proper timeout handling
            search_result = await asyncio.wait_for(
                self.aclient.search(
                    collection_name=config.qdrant_collection_name,
                    query_vector=query_embedding,
                    limit=limit or config.search_limit,
                    score_threshold=config.similarity_threshold,
                    search_params=SearchParams(
                        hnsw_ef=128,  # Increased for better recall
                        exact=False
                    )
                ),
                timeout=config.rag_timeout_ms / 1000.0  # Use config timeout
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    "id": str(point.id),
                    "text": point.payload.get("text", ""),
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
            logger.info(f"⚡ Local search completed in {elapsed:.1f}ms, found {len(results)} results")
            return results
            
        except asyncio.TimeoutError:
            elapsed = (time.time() - start_time) * 1000
            logger.warning(f"⚠️ Local search timeout after {elapsed:.1f}ms")
            return []
        except Exception as e:
            logger.error(f"❌ Local search error: {e}")
            return []
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
            if self.aclient:
                await self.aclient.close()
            logger.info("✅ Local Qdrant RAG system closed")
        except Exception as e:
            logger.error(f"❌ Error closing Qdrant RAG system: {e}")

# Global instance
qdrant_rag = QdrantRAGSystem()