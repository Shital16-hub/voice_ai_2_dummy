# services/rag_service.py
"""
RAG service for Qdrant integration in LiveKit 1.1 multi-agent system
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import openai
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SearchParams

logger = logging.getLogger(__name__)


class RAGService:
    """
    Async RAG service for knowledge base integration
    Optimized for LiveKit 1.1 multi-agent telephony
    """
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "telephony_knowledge",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        timeout_ms: int = 1500,
        similarity_threshold: float = 0.25,
        search_limit: int = 3
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.timeout_ms = timeout_ms
        self.similarity_threshold = similarity_threshold
        self.search_limit = search_limit
        
        # Clients (initialized in initialize())
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        
        # Performance cache
        self.cache: Dict[str, List[Dict]] = {}
        self.max_cache_size = 100
        
        # State
        self.ready = False
        
        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
    
    async def initialize(self) -> bool:
        """Initialize the RAG service"""
        try:
            logger.info("🧠 Initializing RAG service...")
            
            # Initialize Qdrant client
            self.qdrant_client = AsyncQdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=self.timeout_ms / 1000.0
            )
            
            # Test connection
            await self._test_connection()
            
            self.ready = True
            logger.info("✅ RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG service initialization failed: {e}")
            return False
    
    async def _test_connection(self):
        """Test Qdrant connection"""
        try:
            collections = await self.qdrant_client.get_collections()
            logger.info(f"📊 Connected to Qdrant, found {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"❌ Qdrant connection test failed: {e}")
            raise
    
    async def search_knowledge(
        self, 
        query: str, 
        limit: Optional[int] = None,
        context_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant information
        Optimized for telephony latency requirements
        """
        if not self.ready or not self.qdrant_client:
            logger.warning("⚠️ RAG service not ready")
            return []
        
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{query.lower().strip()}_{limit or self.search_limit}"
            if cache_key in self.cache:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"⚡ Cache hit in {elapsed_ms:.1f}ms")
                return self.cache[cache_key]
            
            # Create embedding
            query_embedding = await self._create_embedding(query)
            if not query_embedding:
                return []
            
            # Search with timeout
            search_limit = limit or self.search_limit
            search_result = await asyncio.wait_for(
                self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=search_limit,
                    score_threshold=self.similarity_threshold,
                    search_params=SearchParams(
                        hnsw_ef=128,
                        exact=False
                    )
                ),
                timeout=self.timeout_ms / 1000.0
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
            
            # Cache results (with size management)
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = results
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"⚡ Knowledge search completed in {elapsed_ms:.1f}ms, found {len(results)} results")
            return results
            
        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"⚠️ Knowledge search timeout after {elapsed_ms:.1f}ms")
            return []
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return []
    
    async def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for text"""
        try:
            if not self.openai_client:
                logger.warning("⚠️ OpenAI client not configured")
                return None
            
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000]  # Truncate to avoid token limits
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return None
    
    def clean_content_for_voice(self, content: str) -> str:
        """Clean content for voice response"""
        try:
            # Remove formatting characters
            content = content.replace("Q: ", "").replace("A: ", "")
            content = content.replace("■", "").replace("●", "").replace("•", "")
            content = content.replace("- ", "").replace("* ", "")
            
            # Handle multi-line content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if lines:
                for line in lines:
                    if len(line) > 15 and not line.startswith(('Q:', 'A:', '#', '-', '*', '■')):
                        content = line
                        break
                else:
                    content = lines[0]
            
            # Limit length for voice (telephony optimization)
            if len(content) > 200:
                sentences = content.split('.')
                if len(sentences) > 1:
                    content = sentences[0] + "."
                else:
                    content = content[:200] + "..."
            
            return content
            
        except Exception:
            return content[:150] if len(content) > 150 else content
    
    async def get_relevant_context(
        self, 
        service_type: str, 
        customer_issue: Optional[str] = None
    ) -> Optional[str]:
        """Get context relevant to specific service type"""
        try:
            # Build context-aware query
            query_parts = [service_type]
            if customer_issue:
                query_parts.append(customer_issue)
            
            query = " ".join(query_parts)
            results = await self.search_knowledge(query, limit=1)
            
            if results and len(results) > 0:
                best_result = results[0]
                if best_result["score"] >= self.similarity_threshold:
                    return self.clean_content_for_voice(best_result["text"])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Context retrieval error: {e}")
            return None
    
    async def get_pricing_info(self, service_type: str) -> Optional[str]:
        """Get pricing information for specific service"""
        try:
            query = f"pricing cost {service_type}"
            results = await self.search_knowledge(query, limit=1)
            
            if results and len(results) > 0:
                return self.clean_content_for_voice(results[0]["text"])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Pricing info error: {e}")
            return None
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.qdrant_client:
                await self.qdrant_client.close()
            logger.info("✅ RAG service closed")
        except Exception as e:
            logger.error(f"❌ RAG service cleanup error: {e}")


# Global instance for dependency injection
rag_service: Optional[RAGService] = None


def get_rag_service() -> Optional[RAGService]:
    """Get the global RAG service instance"""
    return rag_service


def initialize_rag_service(
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "telephony_knowledge",
    openai_api_key: Optional[str] = None,
    **kwargs
) -> RAGService:
    """Initialize global RAG service"""
    global rag_service
    rag_service = RAGService(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        openai_api_key=openai_api_key,
        **kwargs
    )
    return rag_service