# main.py - ULTRA LOW LATENCY VERSION WITH OPENAI REALTIME API
"""
Ultra Low Latency Roadside Assistance Agent - TARGET: <2 SECONDS
✅ OpenAI Realtime API (300ms speech-to-speech latency)
✅ No STT→LLM→TTS pipeline delays
✅ Background RAG integration for knowledge enhancement
✅ Automatic information extraction via conversation context
✅ Human transfer functionality
✅ All monitoring features preserved
✅ GOAL: Total latency < 2 seconds
"""
import asyncio
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from livekit import rtc, api
from livekit.agents import (
    Agent, AgentSession, JobContext, RunContext,
    function_tool, get_job_context, ChatContext, ChatMessage,
    WorkerOptions, cli
)
from livekit.plugins import openai, silero

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES (PRESERVED)
# ============================================================================

class ServiceType(Enum):
    """Service types based on call transcript analysis"""
    TOWING = "towing"
    JUMP_START = "jump_start"
    TIRE_CHANGE = "tire_change"
    TIRE_REPLACEMENT = "tire_replacement"
    WINCH_OUT = "winch_out"
    LOCKOUT = "lockout"
    FUEL_DELIVERY = "fuel_delivery"
    UNKNOWN = "unknown"

@dataclass
class VehicleInfo:
    """Structured vehicle information"""
    year: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    neutral_functional: Optional[bool] = None

@dataclass
class LocationInfo:
    """Structured location information"""
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    landmarks: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None

@dataclass
class CustomerInfo:
    """Structured customer information"""
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    insurance_info: Optional[str] = None

@dataclass
class ServiceRequest:
    """Complete service request structure"""
    service_type: ServiceType = ServiceType.UNKNOWN
    customer: CustomerInfo = None
    vehicle: VehicleInfo = None
    location: LocationInfo = None
    special_requirements: List[str] = None
    estimated_cost: Optional[str] = None
    job_number: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.customer is None:
            self.customer = CustomerInfo()
        if self.vehicle is None:
            self.vehicle = VehicleInfo()
        if self.location is None:
            self.location = LocationInfo()
        if self.special_requirements is None:
            self.special_requirements = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class CallTranscript:
    """Individual transcript entry with intelligence"""
    speaker: str
    text: str
    timestamp: datetime
    is_final: bool
    confidence: Optional[float] = None
    extracted_entities: Optional[Dict] = None
    intent: Optional[str] = None

# ============================================================================
# ULTRA-FAST MONITORING AGENT (SIMPLIFIED)
# ============================================================================

class UltraFastMonitoringAgent:
    """
    Ultra-fast monitoring agent optimized for minimal latency
    SIMPLIFIED: Only essential monitoring, fast pattern matching
    """
    
    def __init__(self, room_name: str = ""):
        self.transcripts: List[CallTranscript] = []
        self.service_request = ServiceRequest()
        self.room_name = room_name
        self.start_time = datetime.now()
        
        # Pre-compiled regex patterns for instant extraction
        self.extraction_patterns = {
            'name': [
                re.compile(r'(?:my name is|i\'m|this is|call me)\s+([a-zA-Z\s]+)', re.IGNORECASE),
                re.compile(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)$'),  # "Mike Anderson"
            ],
            'phone': [
                re.compile(r'(\d{3})\s*[-.]?\s*(\d{3})\s*[-.]?\s*(\d{4})'),
                re.compile(r'(?:eight|eight)\s*(?:one|one)\s*(?:eight|eight)', re.IGNORECASE),
            ],
            'service': [
                (re.compile(r'\b(?:tow|towing|stuck|won\'t start|can\'t start)\b', re.IGNORECASE), ServiceType.TOWING),
                (re.compile(r'\b(?:jump\s*start|battery|dead battery)\b', re.IGNORECASE), ServiceType.JUMP_START),
                (re.compile(r'\b(?:tire|flat tire)\b', re.IGNORECASE), ServiceType.TIRE_CHANGE),
                (re.compile(r'\b(?:locked out|lockout|keys)\b', re.IGNORECASE), ServiceType.LOCKOUT),
            ]
        }
    
    def extract_fast(self, text: str) -> Dict[str, str]:
        """Ultra-fast pattern-based extraction"""
        extracted = {}
        
        # Extract name
        if not self.service_request.customer.name:
            for pattern in self.extraction_patterns['name']:
                match = pattern.search(text)
                if match:
                    if len(match.groups()) == 1:
                        name = match.group(1).strip().title()
                    else:
                        name = f"{match.group(1)} {match.group(2)}".strip().title()
                    
                    if len(name) > 2 and not any(char.isdigit() for char in name):
                        self.service_request.customer.name = name
                        extracted['customer_name'] = name
                        logger.info(f"⚡ EXTRACTED - Name: {name}")
                        break
        
        # Extract phone (handle spoken numbers)
        if not self.service_request.customer.phone:
            # Convert spoken numbers to digits
            phone_text = text.lower()
            spoken_to_digit = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
            }
            for word, digit in spoken_to_digit.items():
                phone_text = phone_text.replace(word, digit)
            
            for pattern in self.extraction_patterns['phone']:
                match = pattern.search(phone_text)
                if match:
                    if len(match.groups()) == 1:
                        phone = match.group(1)
                    else:
                        phone = ''.join(match.groups())
                    
                    if len(phone) >= 10:
                        phone = phone[-10:]  # Last 10 digits
                        self.service_request.customer.phone = phone
                        extracted['customer_phone'] = phone
                        logger.info(f"⚡ EXTRACTED - Phone: {phone}")
                        break
        
        # Extract service type
        if self.service_request.service_type == ServiceType.UNKNOWN:
            for pattern, service_type in self.extraction_patterns['service']:
                if pattern.search(text):
                    self.service_request.service_type = service_type
                    extracted['service_type'] = service_type.value
                    logger.info(f"⚡ EXTRACTED - Service: {service_type.value}")
                    break
        
        return extracted
    
    def add_transcript(self, speaker: str, text: str, is_final: bool = True):
        """Add transcript entry"""
        transcript = CallTranscript(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            is_final=is_final
        )
        self.transcripts.append(transcript)
        
        if speaker == "customer":
            logger.info(f"👤 USER: {text}")
        elif speaker == "agent":
            logger.info(f"🤖 AGENT: {text}")
    
    def get_conversation_summary(self) -> Dict:
        """Generate conversation summary"""
        return {
            "room_name": self.room_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "service_request": {
                "service_type": self.service_request.service_type.value,
                "customer": asdict(self.service_request.customer),
                "vehicle": asdict(self.service_request.vehicle),
                "location": asdict(self.service_request.location),
                "special_requirements": self.service_request.special_requirements,
                "estimated_cost": self.service_request.estimated_cost,
                "job_number": self.service_request.job_number,
            },
            "transcript_count": len(self.transcripts),
            "full_transcript": [
                {
                    "speaker": t.speaker,
                    "text": t.text,
                    "timestamp": t.timestamp.isoformat(),
                    "is_final": t.is_final
                }
                for t in self.transcripts
            ]
        }
    
    async def save_conversation_data(self):
        """Save conversation data"""
        summary = self.get_conversation_summary()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"roadside_call_{self.room_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"💾 TRANSCRIPT SAVED: {filename}")
        except Exception as e:
            logger.error(f"❌ FAILED TO SAVE TRANSCRIPT: {e}")

# ============================================================================
# BACKGROUND RAG SYSTEM (NON-BLOCKING)
# ============================================================================

class BackgroundRAGSystem:
    """Background RAG system that doesn't block the main conversation"""
    
    def __init__(self):
        self.ready = False
        self.qdrant_rag = None
        self.cache = {}
        
    async def initialize_background(self):
        """Initialize RAG in background (non-blocking)"""
        try:
            from qdrant_rag_system import qdrant_rag
            from config import config
            
            await qdrant_rag.initialize()
            self.qdrant_rag = qdrant_rag
            self.ready = True
            logger.info("🧠 RAG: ✅ Background initialization complete")
        except Exception as e:
            logger.warning(f"⚠️ RAG background init failed: {e}")
    
    async def search_fast(self, query: str) -> Optional[str]:
        """Fast RAG search with timeout and caching"""
        if not self.ready or not self.qdrant_rag:
            return None
        
        # Check cache first
        if query in self.cache:
            return self.cache[query]
        
        try:
            # Ultra-fast search with short timeout
            results = await asyncio.wait_for(
                self.qdrant_rag.search(query, limit=1),
                timeout=0.5  # 500ms max
            )
            
            if results and len(results) > 0:
                content = results[0]["text"]
                # Clean and shorten for voice
                if len(content) > 150:
                    sentences = content.split('.')
                    content = sentences[0] + "." if sentences else content[:150]
                
                # Cache result
                self.cache[query] = content
                return content
                
        except asyncio.TimeoutError:
            logger.debug("⚡ RAG timeout (continuing without)")
        except Exception as e:
            logger.debug(f"⚡ RAG error: {e}")
        
        return None

# ============================================================================
# ULTRA LOW LATENCY REALTIME AGENT
# ============================================================================

class UltraLowLatencyAgent(Agent):
    """
    Ultra Low Latency Agent using OpenAI Realtime API
    TARGET: <2 seconds total latency
    ✅ Direct speech-to-speech (300ms)
    ✅ Background RAG integration
    ✅ Automatic extraction
    ✅ All monitoring preserved
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance agent.

CRITICAL REQUIREMENTS:
- Keep ALL responses under 30 words for ultra-low latency
- Use natural, conversational tone
- Always acknowledge what the customer says immediately
- Collect information efficiently: name, phone, location, vehicle, service type
- For towing, ask if neutral gear works
- Provide cost estimates when appropriate

RESPONSE STYLE:
- Short, direct responses: "Got it, Mike. What's your phone number?"
- Acknowledge: "Thanks for that. Now, where are you located?"
- Be empathetic but efficient: "Sorry to hear that. Let's get you help."

HUMAN TRANSFER:
- If customer asks for human, use transfer_to_human function immediately

Your goal is to help customers quickly and professionally while keeping responses brief for speed."""
        )
        
        self.monitoring_agent: Optional[UltraFastMonitoringAgent] = None
        self.rag_system = BackgroundRAGSystem()
        self.transfer_sip_address = "sip:voiceai@sip.linphone.org"
    
    def set_room_name(self, room_name: str):
        """Initialize monitoring agent"""
        self.monitoring_agent = UltraFastMonitoringAgent(room_name)
        logger.info(f"⚡ MONITORING INITIALIZED: {room_name}")
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        ULTRA-FAST extraction and optional RAG enhancement
        """
        try:
            if not self.monitoring_agent:
                return
            
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 2:
                return
            
            # INSTANT pattern-based extraction
            extracted = self.monitoring_agent.extract_fast(user_text)
            
            if extracted:
                # Add extraction context for better responses
                context_parts = []
                if 'customer_name' in extracted:
                    context_parts.append(f"Customer: {extracted['customer_name']}")
                if 'customer_phone' in extracted:
                    context_parts.append(f"Phone: {extracted['customer_phone']}")
                if 'service_type' in extracted:
                    context_parts.append(f"Service: {extracted['service_type']}")
                
                if context_parts:
                    context = f"[INFO: {', '.join(context_parts)}]"
                    turn_ctx.add_message(role="system", content=context)
            
            # OPTIONAL: Background RAG enhancement (non-blocking)
            if self.rag_system.ready:
                # Start RAG search but don't wait for it
                asyncio.create_task(self._enhance_with_rag(turn_ctx, user_text))
            
        except Exception as e:
            logger.error(f"❌ Turn completion error: {e}")
    
    async def _enhance_with_rag(self, turn_ctx: ChatContext, user_text: str):
        """Background RAG enhancement (non-blocking)"""
        try:
            rag_content = await self.rag_system.search_fast(user_text)
            if rag_content:
                # Add RAG context (this won't delay the main response)
                rag_context = f"[KNOWLEDGE: {rag_content}]"
                turn_ctx.add_message(role="system", content=rag_context)
                logger.info("🧠 RAG enhanced response")
        except Exception as e:
            logger.debug(f"RAG enhancement failed: {e}")
    
    # ========================================================================
    # TRANSFER FUNCTIONS (PRESERVED)
    # ========================================================================
    
    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer caller to human agent when explicitly requested"""
        try:
            logger.info("🔄 HUMAN TRANSFER INITIATED")
            
            # Save conversation data
            if self.monitoring_agent:
                await self.monitoring_agent.save_conversation_data()
            
            # Get job context and find SIP participant
            job_ctx = get_job_context()
            sip_participant = None
            
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                    sip_participant = participant
                    break
            
            if not sip_participant:
                return "I'm sorry, I couldn't find any active participants to transfer."
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=self.transfer_sip_address,
                play_dialtone=True,
            )
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=30.0
            )
            
            logger.info("✅ Transfer completed successfully")
            return "Connecting you to a human agent now."
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "Having trouble with transfer. Let me continue helping you."

# ============================================================================
# ULTRA LOW LATENCY ENTRYPOINT
# ============================================================================

async def entrypoint(ctx: JobContext):
    """
    ULTRA LOW LATENCY ENTRYPOINT
    TARGET: Agent response within 2 seconds
    """
    logger.info("⚡ === ULTRA LOW LATENCY ROADSIDE AGENT (TARGET: <2s) ===")
    
    # INSTANT connection
    await ctx.connect()
    logger.info("✅ CONNECTED")
    
    # Create agent immediately
    agent = UltraLowLatencyAgent()
    agent.set_room_name(ctx.room.name)
    
    # Start RAG initialization in background (don't wait)
    asyncio.create_task(agent.rag_system.initialize_background())
    
    # ULTRA-OPTIMIZED SESSION with OpenAI Realtime API
    logger.info("⚡ CREATING REALTIME SESSION...")
    session = AgentSession(
        # REALTIME MODEL: Direct speech-to-speech (300ms latency)
        llm=openai.realtime.RealtimeModel(
            model="gpt-4o-realtime-preview",
            instructions=agent.instructions,
            voice="echo",  # Fast, clear voice
            temperature=0.1,  # Consistent, fast responses
            
            # OPTIMIZED turn detection for speed
            turn_detection=openai.realtime.ServerVAD(
                threshold=0.7,  # Slightly higher threshold for responsiveness
                prefix_padding_ms=200,  # Minimal padding
                silence_duration_ms=500,  # Quick silence detection
            ),
            
            # Enable interruptions for natural conversation
            modalities=["audio", "text"],
        ),
        
        # MINIMAL VAD for ultra-fast turn detection
        vad=silero.VAD.load(),
        
        # OPTIMIZED settings
        min_endpointing_delay=0.2,  # Ultra-fast endpointing
        max_endpointing_delay=1.0,  # Quick timeout
        allow_interruptions=True,
        min_interruption_duration=0.2,
    )
    
    logger.info("✅ REALTIME SESSION CREATED")
    
    # Set monitoring reference
    if agent.monitoring_agent:
        agent.monitoring_agent._current_session = session
    
    # ========================================================================
    # ULTRA-FAST EVENT HANDLERS
    # ========================================================================
    
    @session.on("user_input_transcribed")
    def on_user_transcript(event):
        """Capture user transcripts for monitoring"""
        if hasattr(event, 'is_final') and event.is_final and agent.monitoring_agent:
            agent.monitoring_agent.add_transcript("customer", event.transcript, True)
    
    @session.on("conversation_item_added")
    def on_conversation_item(event):
        """Capture agent responses for monitoring"""
        if hasattr(event, 'item') and hasattr(event.item, 'role'):
            if event.item.role == 'assistant' and hasattr(event.item, 'text_content'):
                if event.item.text_content and agent.monitoring_agent:
                    agent.monitoring_agent.add_transcript("agent", event.item.text_content, True)
    
    # Start session with agent
    logger.info("⚡ STARTING REALTIME SESSION...")
    await session.start(room=ctx.room, agent=agent)
    
    # ========================================================================
    # SYSTEM READY
    # ========================================================================
    
    logger.info("⚡ ULTRA LOW LATENCY SYSTEM READY!")
    logger.info("🎯 TARGET LATENCY: <2 seconds")
    logger.info("🔥 REALTIME API: ✅ 300ms speech-to-speech")
    logger.info("🧠 RAG: 🔄 Background loading")
    logger.info("📊 MONITORING: ✅ Ultra-fast extraction")
    logger.info("🔄 TRANSFER: ✅ Available")
    logger.info(f"📞 Room: {ctx.room.name}")
    
    # INSTANT greeting using generate_reply (Realtime API)
    await session.generate_reply(
        instructions="Give a brief greeting: 'Roadside assistance, this is Mark. How can I help with your vehicle today?' Keep it under 15 words and speak naturally."
    )
    
    logger.info("⚡ GREETING SENT - ULTRA LOW LATENCY ACTIVE")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        logger.info("⚡ STARTING ULTRA LOW LATENCY ROADSIDE SYSTEM")
        logger.info("🎯 TARGET: <2 seconds total latency")
        logger.info("🔥 TECH: OpenAI Realtime API + Background RAG")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except KeyboardInterrupt:
        logger.info("🛑 SYSTEM SHUTDOWN")
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)
    finally:
        logger.info("👋 ULTRA LOW LATENCY SYSTEM STOPPED")