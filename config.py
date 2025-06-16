# config.py - Updated for Local Qdrant Docker with FIXED timeouts
"""
Optimized Configuration for LiveKit RAG Agent with Local Qdrant Docker
FIXED: Proper timeout handling and performance settings
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class QdrantConfig(BaseSettings):
    """Qdrant-specific configuration optimized for ultra-low latency with local Docker"""
    
    # ✅ REQUIRED: LiveKit Settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # ✅ REQUIRED: AI Service API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    
    # ✅ OPTIONAL: Enhanced TTS
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    eleven_api_key: Optional[str] = Field(default=None, env="ELEVEN_API_KEY")
    
    # ✅ TWILIO/SIP SETTINGS
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_phone_number: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    twilio_trunk_sid: Optional[str] = Field(default=None, env="TWILIO_TRUNK_SID")
    sip_username: Optional[str] = Field(default=None, env="SIP_USERNAME")
    sip_password: Optional[str] = Field(default=None, env="SIP_PASSWORD")
    sip_domain: Optional[str] = Field(default=None, env="SIP_DOMAIN")
    sip_trunk_id: Optional[str] = Field(default=None, env="SIP_TRUNK_ID")
    livekit_sip_uri: Optional[str] = Field(default=None, env="LIVEKIT_SIP_URI")
    
    # ✅ LOCAL QDRANT DOCKER SETTINGS (FIXED for proper timeout handling)
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="telephony_knowledge", env="QDRANT_COLLECTION")
    qdrant_collection: str = Field(default="telephony_knowledge", env="QDRANT_COLLECTION")
    
    # ✅ PERFORMANCE SETTINGS (Optimized for local Docker)
    qdrant_prefer_grpc: bool = Field(default=False, env="QDRANT_PREFER_GRPC")
    qdrant_timeout: int = Field(default=5, env="QDRANT_TIMEOUT")
    qdrant_grpc_port: int = Field(default=6334, env="QDRANT_GRPC_PORT")
    
    # ✅ EMBEDDING SETTINGS
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")
    
    # ✅ RAG SETTINGS (FIXED: Proper timeout values)
    chunk_size: int = Field(default=300, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_tokens: int = Field(default=50, env="MAX_TOKENS")
    
    # ✅ FIXED PERFORMANCE SETTINGS (Match actual search times)
    rag_timeout_ms: int = Field(default=1500, env="RAG_TIMEOUT_MS")  # FIXED: 1.5 seconds
    search_limit: int = Field(default=5, env="SEARCH_LIMIT")  # FIXED: More results
    similarity_threshold: float = Field(default=0.25, env="SIMILARITY_THRESHOLD")  # FIXED: Lower threshold
    
    # ✅ LOCAL DOCKER OPTIMIZATION
    use_local_docker: bool = Field(default=True, env="USE_LOCAL_DOCKER")
    docker_health_check_retries: int = Field(default=3, env="DOCKER_HEALTH_CHECK_RETRIES")
    
    # ✅ TELEPHONY
    transfer_sip_address: str = Field(
        default="sip:voiceai@sip.linphone.org", 
        env="TRANSFER_SIP_ADDRESS"
    )
    
    # ✅ PATHS
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def qdrant_storage_dir(self) -> Path:
        return self.project_root / "qdrant_storage"
    
    @property
    def qdrant_config_dir(self) -> Path:
        return self.project_root / "qdrant_config"
    
    @property
    def qdrant_grpc_url(self) -> str:
        """Get Qdrant gRPC URL if prefer_grpc is enabled"""
        if self.qdrant_prefer_grpc:
            return f"http://localhost:{self.qdrant_grpc_port}"
        return self.qdrant_url
    
    def ensure_directories(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.qdrant_storage_dir.mkdir(exist_ok=True)
        if hasattr(self, 'qdrant_config_dir'):
            self.qdrant_config_dir.mkdir(exist_ok=True)
    
    def is_docker_healthy(self) -> bool:
        """Check if Qdrant Docker container is healthy"""
        try:
            import requests
            response = requests.get(f"{self.qdrant_url}/", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Global configuration instance
config = QdrantConfig()
config.ensure_directories()

def validate_config():
    """Validate essential configuration with Docker health check"""
    required_fields = [
        ("OPENAI_API_KEY", config.openai_api_key),
        ("DEEPGRAM_API_KEY", config.deepgram_api_key),
    ]
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    print("✅ Configuration validated")
    print(f"📞 Transfer destination: {config.transfer_sip_address}")
    print(f"🔍 Qdrant URL: {config.qdrant_url}")
    print(f"🚀 Qdrant gRPC: {config.qdrant_prefer_grpc}")
    print(f"⚡ RAG timeout: {config.rag_timeout_ms}ms (FIXED)")
    print(f"🔍 Search limit: {config.search_limit}")
    print(f"📊 Similarity threshold: {config.similarity_threshold}")
    print(f"🐳 Local Docker mode: {config.use_local_docker}")
    
    # Check Docker health
    if config.use_local_docker:
        if config.is_docker_healthy():
            print("✅ Qdrant Docker container is healthy")
        else:
            print("⚠️  Warning: Qdrant Docker container not responding")
            print("   Run: docker-compose up -d to start Qdrant")

if __name__ == "__main__":
    validate_config()