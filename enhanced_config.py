# enhanced_config.py
"""
Enhanced configuration for intelligent monitoring system
Cross-platform compatible (Windows development + Linux/AWS production)
"""
import os
import platform
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class EnhancedConfig(BaseSettings):
    """Enhanced configuration with NLP and monitoring settings"""
    
    # Platform detection
    platform: str = platform.system().lower()
    
    # LiveKit settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # AI API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    
    # Twilio/SIP (from your existing config)
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_phone_number: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    
    # Transfer destination (from your existing config)
    transfer_sip_address: str = Field(
        default="sip:voiceai@sip.linphone.org", 
        env="TRANSFER_SIP_ADDRESS"
    )
    
    # Enhanced NLP Settings
    spacy_model: str = Field(default="en_core_web_md", env="SPACY_MODEL")
    use_spacy_llm: bool = Field(default=True, env="USE_SPACY_LLM")
    
    # Entity Extraction Settings
    extract_phone_numbers: bool = Field(default=True)
    extract_addresses: bool = Field(default=True)
    extract_vehicle_info: bool = Field(default=True)
    extract_names: bool = Field(default=True)
    extract_job_numbers: bool = Field(default=True)
    extract_costs: bool = Field(default=True)
    
    # Monitoring Settings
    save_transcripts: bool = Field(default=True)
    save_extracted_data: bool = Field(default=True)
    real_time_analysis: bool = Field(default=True)
    
    # Data Storage Settings
    transcript_format: str = Field(default="json", env="TRANSCRIPT_FORMAT")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Performance Settings (adjusted for AWS)
    max_concurrent_extractions: int = Field(default=5, env="MAX_CONCURRENT_EXTRACTIONS")
    extraction_timeout_seconds: int = Field(default=30, env="EXTRACTION_TIMEOUT")
    
    # Quality Settings
    minimum_confidence_score: float = Field(default=0.7, env="MIN_CONFIDENCE_SCORE")
    require_phone_validation: bool = Field(default=True)
    require_address_validation: bool = Field(default=False)
    
    # AWS/Production Settings
    aws_region: Optional[str] = Field(default=None, env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_to_file: bool = Field(default=True, env="LOG_TO_FILE")
    
    # Paths (cross-platform)
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent
    
    @property
    def call_recordings_dir(self) -> Path:
        return self.project_root / "call_recordings"
    
    @property
    def extracted_data_dir(self) -> Path:
        return self.project_root / "extracted_data"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    # Platform-specific settings
    @property
    def is_production(self) -> bool:
        """Check if running in production (AWS/Linux)"""
        return self.platform == "linux" or os.getenv("ENVIRONMENT") == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development (Windows)"""
        return self.platform == "windows" or os.getenv("ENVIRONMENT") == "development"
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.call_recordings_dir,
            self.extracted_data_dir,
            self.logs_dir,
            self.data_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)

# Global enhanced config instance
enhanced_config = EnhancedConfig()
enhanced_config.ensure_directories()

def validate_enhanced_config():
    """Validate enhanced configuration"""
    required_fields = [
        ("OPENAI_API_KEY", enhanced_config.openai_api_key),
        ("DEEPGRAM_API_KEY", enhanced_config.deepgram_api_key),
    ]
    
    # LiveKit required only for production
    if enhanced_config.is_production:
        required_fields.extend([
            ("LIVEKIT_URL", enhanced_config.livekit_url),
            ("LIVEKIT_API_KEY", enhanced_config.livekit_api_key),
            ("LIVEKIT_API_SECRET", enhanced_config.livekit_api_secret),
        ])
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields and enhanced_config.is_production:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    print(f"Platform: {enhanced_config.platform}")
    print(f"Environment: {'Production' if enhanced_config.is_production else 'Development'}")
    print(f"NLP Model: {enhanced_config.spacy_model}")
    print(f"Real-time Analysis: {enhanced_config.real_time_analysis}")
    print(f"Save Transcripts: {enhanced_config.save_transcripts}")
    
    if missing_fields and not enhanced_config.is_production:
        print(f"Warning: Missing API keys for development: {', '.join(missing_fields)}")
    
    return True

if __name__ == "__main__":
    validate_enhanced_config()
