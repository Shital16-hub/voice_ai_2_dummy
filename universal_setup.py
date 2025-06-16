# universal_setup.py
"""
Universal cross-platform setup script for enhanced monitoring system
Works on Windows (development) and Linux/AWS (production)
"""
import os
import platform
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_platform():
    """Detect the current platform"""
    system = platform.system().lower()
    logger.info(f"Detected platform: {system}")
    return system

def install_requirements():
    """Install enhanced requirements"""
    logger.info("Installing enhanced requirements...")
    
    # Use the requirements file that exists
    req_files = ["enhanced_requirements.txt", "requirements.txt"]
    req_file = None
    
    for rf in req_files:
        if Path(rf).exists():
            req_file = rf
            break
    
    if not req_file:
        logger.error("No requirements file found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", req_file
        ])
        logger.info(f"Requirements installed successfully from {req_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def download_spacy_models():
    """Download required spaCy models"""
    logger.info("Downloading spaCy models...")
    
    models = [
        "en_core_web_sm",  # Small model
        "en_core_web_md",  # Medium model for better extraction
    ]
    
    success_count = 0
    for model in models:
        try:
            logger.info(f"Downloading {model}...")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", model
            ])
            logger.info(f"Successfully downloaded {model}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to download {model}: {e}")
    
    return success_count > 0

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        "call_recordings",
        "extracted_data", 
        "logs",
        "models",
        "temp",
        "data"  # For your existing system
    ]
    
    try:
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False

def create_enhanced_config():
    """Create enhanced configuration file (cross-platform)"""
    logger.info("Creating enhanced configuration...")
    
    config_content = '''# enhanced_config.py
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
'''
    
    try:
        with open("enhanced_config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        logger.info("Enhanced configuration created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create enhanced config: {e}")
        return False

def create_environment_files():
    """Create environment files for different environments"""
    logger.info("Creating environment files...")
    
    # Development .env (Windows)
    dev_env_content = '''# Development Environment (Windows)
ENVIRONMENT=development

# LiveKit Configuration (optional for development)
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_api_key_here
LIVEKIT_API_SECRET=your_api_secret_here

# AI Service API Keys (required)
OPENAI_API_KEY=your_openai_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Twilio/SIP Settings
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_PHONE_NUMBER=your_twilio_number
TWILIO_AUTH_TOKEN=your_twilio_token
TRANSFER_SIP_ADDRESS=sip:voiceai@sip.linphone.org

# Enhanced NLP Settings
SPACY_MODEL=en_core_web_md
USE_SPACY_LLM=true

# Monitoring Settings
TRANSCRIPT_FORMAT=json
REAL_TIME_ANALYSIS=true
SAVE_TRANSCRIPTS=true
SAVE_EXTRACTED_DATA=true

# Quality Settings
MIN_CONFIDENCE_SCORE=0.7
REQUIRE_PHONE_VALIDATION=true
REQUIRE_ADDRESS_VALIDATION=false

# Development Settings
LOG_LEVEL=DEBUG
LOG_TO_FILE=true
MAX_CONCURRENT_EXTRACTIONS=3
EXTRACTION_TIMEOUT=30
'''
    
    # Production .env (AWS/Linux)
    prod_env_content = '''# Production Environment (AWS/Linux)
ENVIRONMENT=production

# LiveKit Configuration (required for production)
LIVEKIT_URL=wss://your-production-livekit-server.com
LIVEKIT_API_KEY=your_production_api_key
LIVEKIT_API_SECRET=your_production_api_secret

# AI Service API Keys (required)
OPENAI_API_KEY=your_production_openai_key
DEEPGRAM_API_KEY=your_production_deepgram_key
ELEVENLABS_API_KEY=your_production_elevenlabs_key

# Twilio/SIP Settings
TWILIO_ACCOUNT_SID=your_production_twilio_sid
TWILIO_PHONE_NUMBER=your_production_twilio_number
TWILIO_AUTH_TOKEN=your_production_twilio_token
TRANSFER_SIP_ADDRESS=sip:production@your-sip-provider.com

# Enhanced NLP Settings
SPACY_MODEL=en_core_web_md
USE_SPACY_LLM=true

# Monitoring Settings
TRANSCRIPT_FORMAT=json
REAL_TIME_ANALYSIS=true
SAVE_TRANSCRIPTS=true
SAVE_EXTRACTED_DATA=true

# AWS Settings
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# Database Settings (production)
DATABASE_URL=postgresql://user:password@your-rds-endpoint/monitoring_db

# Production Quality Settings
MIN_CONFIDENCE_SCORE=0.8
REQUIRE_PHONE_VALIDATION=true
REQUIRE_ADDRESS_VALIDATION=true

# Production Performance Settings
LOG_LEVEL=INFO
LOG_TO_FILE=true
MAX_CONCURRENT_EXTRACTIONS=10
EXTRACTION_TIMEOUT=60
'''
    
    try:
        # Create development environment file
        if not Path(".env.dev").exists():
            with open(".env.dev", "w", encoding="utf-8") as f:
                f.write(dev_env_content)
            logger.info("Created .env.dev for development")
        
        # Create production environment file
        if not Path(".env.prod").exists():
            with open(".env.prod", "w", encoding="utf-8") as f:
                f.write(prod_env_content)
            logger.info("Created .env.prod for production")
        
        # Create general example
        if not Path(".env.example").exists():
            with open(".env.example", "w", encoding="utf-8") as f:
                f.write(dev_env_content)
            logger.info("Created .env.example")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create environment files: {e}")
        return False

def create_aws_deployment_files():
    """Create AWS deployment configuration files"""
    logger.info("Creating AWS deployment files...")
    
    # Docker file for AWS deployment
    dockerfile_content = '''FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt enhanced_requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r enhanced_requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p call_recordings extracted_data logs models temp data

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Start command
CMD ["python", "main.py"]
'''
    
    # Docker Compose for local testing
    docker_compose_content = '''version: '3.8'

services:
  monitoring-system:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.prod
    volumes:
      - ./call_recordings:/app/call_recordings
      - ./extracted_data:/app/extracted_data
      - ./logs:/app/logs
    restart: unless-stopped
    
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped
'''
    
    # AWS ECS task definition
    ecs_task_content = '''{
  "family": "livekit-monitoring-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "monitoring-system",
      "image": "YOUR_ECR_URI/livekit-monitoring:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "AWS_REGION", "value": "us-east-1"}
      ],
      "secrets": [
        {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-key"},
        {"name": "DEEPGRAM_API_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:deepgram-key"},
        {"name": "LIVEKIT_API_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:livekit-key"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/livekit-monitoring",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}'''
    
    try:
        # Create Dockerfile
        with open("Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        logger.info("Created Dockerfile for AWS deployment")
        
        # Create docker-compose.yml
        with open("docker-compose.yml", "w", encoding="utf-8") as f:
            f.write(docker_compose_content)
        logger.info("Created docker-compose.yml")
        
        # Create ECS task definition
        with open("ecs-task-definition.json", "w", encoding="utf-8") as f:
            f.write(ecs_task_content)
        logger.info("Created ECS task definition")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create AWS deployment files: {e}")
        return False

def test_system():
    """Test the enhanced system"""
    logger.info("Testing enhanced system...")
    
    try:
        # Test spaCy
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp("Hello, my name is John Smith. Phone: 555-123-4567.")
            logger.info(f"spaCy test passed - Found {len(doc.ents)} entities")
        except OSError:
            logger.warning("spaCy model not found - run download_spacy_models()")
        
        # Test LiveKit imports
        from livekit.agents import Agent, function_tool
        logger.info("LiveKit imports working")
        
        # Test enhanced config
        from enhanced_config import enhanced_config, validate_enhanced_config
        validate_enhanced_config()
        logger.info("Enhanced configuration working")
        
        logger.info("All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main setup function"""
    platform_name = detect_platform()
    logger.info(f"Setting up Enhanced LiveKit Monitoring System for {platform_name}")
    
    steps = [
        ("Install requirements", install_requirements),
        ("Download spaCy models", download_spacy_models),
        ("Setup directories", setup_directories),
        ("Create enhanced config", create_enhanced_config),
        ("Create environment files", create_environment_files),
        ("Create AWS deployment files", create_aws_deployment_files),
        ("Test system", test_system),
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        logger.info(f"Running: {step_name}...")
        try:
            if step_func():
                logger.info(f"SUCCESS: {step_name}")
                success_count += 1
            else:
                logger.error(f"FAILED: {step_name}")
        except Exception as e:
            logger.error(f"EXCEPTION in {step_name}: {e}")
    
    logger.info(f"Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count >= len(steps) - 1:  # Allow one failure
        logger.info("SETUP COMPLETE!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("DEVELOPMENT (Windows):")
        logger.info("  1. Copy .env.dev to .env")
        logger.info("  2. Fill in your API keys")
        logger.info("  3. Test: python test_monitoring_system.py")
        logger.info("  4. Demo: python demo_transcript_analysis.py")
        logger.info("")
        logger.info("PRODUCTION (AWS):")
        logger.info("  1. Copy .env.prod to .env")
        logger.info("  2. Configure AWS credentials")
        logger.info("  3. Build: docker build -t livekit-monitoring .")
        logger.info("  4. Deploy: Use ECS task definition")
        logger.info("")
        logger.info("Features ready:")
        logger.info("  - Cross-platform compatibility")
        logger.info("  - Intelligent entity extraction")
        logger.info("  - Real-time monitoring")
        logger.info("  - AWS deployment ready")

if __name__ == "__main__":
    main()