# main.py - COMPLETE UPDATED VERSION WITH ALL FIXES
"""
Advanced LiveKit Monitoring System for Roadside Assistance - FINAL VERSION
✅ Agent responds properly with TTS
✅ All agent speech logged in console
✅ OpenAI API timeouts fixed
✅ Transcript files saved in JSON format
✅ Function tools work correctly
✅ Single voice output (no dual voice)
✅ Proper conversation flow
✅ Enhanced logging and monitoring
"""
import asyncio
import logging
import json
import httpx
from datetime import datetime
from typing import Dict, List, Optional, AsyncIterable
from dataclasses import dataclass, asdict
from enum import Enum

from livekit import rtc, api
from livekit.agents import (
    Agent, AgentSession, JobContext, RunContext, ModelSettings,
    function_tool, get_job_context, ChatContext, ChatMessage,
    WorkerOptions, cli, AutoSubscribe
)
from livekit.plugins import openai, deepgram, silero, elevenlabs

from dotenv import load_dotenv
load_dotenv()

# Import your existing systems
try:
    from qdrant_rag_system import qdrant_rag
    from config import config
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ RAG system not available, continuing without it")

# ============================================================================
# ENHANCED LOGGING SETUP - Shows all agent speech
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Monkey patch AgentSession.say to add speech logging
original_session_say = None
original_generate_reply = None

def patch_session_logging():
    """Patch AgentSession methods to log speech output"""
    global original_session_say, original_generate_reply
    
    from livekit.agents import AgentSession
    
    # Store original methods
    if original_session_say is None:
        original_session_say = AgentSession.say
        original_generate_reply = AgentSession.generate_reply
    
    async def logged_say(self, text, **kwargs):
        """Wrapper for say() that logs the speech"""
        logger.info(f"🎙️ AGENT SPEAKING: {text}")
        return await original_session_say(self, text, **kwargs)
    
    async def logged_generate_reply(self, **kwargs):
        """Wrapper for generate_reply() that logs when generating"""
        logger.info("🧠 AGENT GENERATING REPLY...")
        result = await original_generate_reply(self, **kwargs)
        logger.info("✅ AGENT REPLY GENERATED")
        return result
    
    # Apply patches
    AgentSession.say = logged_say
    AgentSession.generate_reply = logged_generate_reply
    
    logger.info("🔧 PATCHED AgentSession for enhanced speech logging")

# Apply the patch immediately
patch_session_logging()

# ============================================================================
# DATA STRUCTURES
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
# MAIN AGENT CLASS
# ============================================================================

class RoadsideAssistanceAgent(Agent):
    """
    COMPLETE ROADSIDE ASSISTANCE AGENT
    ✅ Responds to customers with proper TTS
    ✅ Logs all speech output
    ✅ Extracts and monitors information
    ✅ Saves complete transcripts
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance agent with intelligent monitoring capabilities.
            
            Your primary job is to help customers with towing, jump starts, tire changes, winch-outs, and other roadside services.
            You automatically extract and monitor important information during conversations.
            
            ALWAYS RESPOND TO THE CUSTOMER with helpful, professional answers. Never leave them without a response.
            
            Follow this process:
            1. Listen to the customer's problem
            2. Acknowledge their situation empathetically 
            3. Collect information in this order:
               - Customer's full name
               - Phone number
               - Exact location (full address, city, state, ZIP)
               - Vehicle year, make, and model
               - Type of service needed
               - For towing: Ask if neutral gear is functional
            4. Provide cost estimates when appropriate
            5. Confirm all details before dispatching
            
            Be professional, efficient, and empathetic. Always acknowledge what the customer says.
            Use the extraction tools automatically as you gather information, but focus on helping the customer first."""
        )
        
        # Initialize monitoring data
        self.transcripts: List[CallTranscript] = []
        self.service_request = ServiceRequest()
        self.room_name = ""
        self.start_time = datetime.now()
        self.conversation_complete = False
        self.session = None  # Will be set by session
    
    # ========================================================================
    # FUNCTION TOOLS FOR INFORMATION EXTRACTION
    # ========================================================================
    
    @function_tool()
    async def extract_customer_info(
        self, 
        ctx: RunContext,
        name: Optional[str] = None, 
        phone: Optional[str] = None,
        email: Optional[str] = None
    ) -> str:
        """
        Extract and store customer information
        
        Args:
            name: Customer's full name
            phone: Customer's phone number
            email: Customer's email address
        """
        extracted_info = []
        
        if name and not self.service_request.customer.name:
            self.service_request.customer.name = name.strip()
            logger.info(f"👤 EXTRACTED - Customer name: {name}")
            extracted_info.append(f"name: {name}")
        
        if phone and not self.service_request.customer.phone:
            # Clean phone number
            clean_phone = ''.join(filter(str.isdigit, phone))
            if len(clean_phone) >= 10:
                self.service_request.customer.phone = clean_phone
                logger.info(f"📞 EXTRACTED - Phone: {clean_phone}")
                extracted_info.append(f"phone: {clean_phone}")
        
        if email and not self.service_request.customer.email:
            self.service_request.customer.email = email.strip()
            logger.info(f"📧 EXTRACTED - Email: {email}")
            extracted_info.append(f"email: {email}")
        
        return f"Customer information extracted: {', '.join(extracted_info)}" if extracted_info else "No new customer information to extract"
    
    @function_tool()
    async def extract_vehicle_info(
        self,
        ctx: RunContext,
        year: Optional[str] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
        color: Optional[str] = None,
        neutral_functional: Optional[bool] = None
    ) -> str:
        """
        Extract and store vehicle information
        
        Args:
            year: Vehicle year
            make: Vehicle make/manufacturer
            model: Vehicle model
            color: Vehicle color
            neutral_functional: Whether neutral gear is functional
        """
        extracted_info = []
        
        if year and not self.service_request.vehicle.year:
            self.service_request.vehicle.year = year.strip()
            logger.info(f"🚗 EXTRACTED - Vehicle year: {year}")
            extracted_info.append(f"year: {year}")
        
        if make and not self.service_request.vehicle.make:
            self.service_request.vehicle.make = make.strip().title()
            logger.info(f"🚗 EXTRACTED - Vehicle make: {make}")
            extracted_info.append(f"make: {make}")
        
        if model and not self.service_request.vehicle.model:
            self.service_request.vehicle.model = model.strip().title()
            logger.info(f"🚗 EXTRACTED - Vehicle model: {model}")
            extracted_info.append(f"model: {model}")
        
        if color and not self.service_request.vehicle.color:
            self.service_request.vehicle.color = color.strip().lower()
            logger.info(f"🚗 EXTRACTED - Vehicle color: {color}")
            extracted_info.append(f"color: {color}")
        
        if neutral_functional is not None and self.service_request.vehicle.neutral_functional is None:
            self.service_request.vehicle.neutral_functional = neutral_functional
            logger.info(f"⚙️ EXTRACTED - Neutral functional: {neutral_functional}")
            extracted_info.append(f"neutral gear: {'functional' if neutral_functional else 'not functional'}")
        
        return f"Vehicle information extracted: {', '.join(extracted_info)}" if extracted_info else "No new vehicle information to extract"
    
    @function_tool()
    async def extract_location_info(
        self,
        ctx: RunContext,
        address: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        zip_code: Optional[str] = None,
        landmarks: Optional[str] = None
    ) -> str:
        """
        Extract and store location information
        
        Args:
            address: Street address
            city: City name
            state: State name
            zip_code: ZIP code
            landmarks: Nearby landmarks
        """
        extracted_info = []
        
        if address and not self.service_request.location.address:
            self.service_request.location.address = address.strip()
            logger.info(f"📍 EXTRACTED - Address: {address}")
            extracted_info.append(f"address: {address}")
        
        if city and not self.service_request.location.city:
            self.service_request.location.city = city.strip().title()
            logger.info(f"🏙️ EXTRACTED - City: {city}")
            extracted_info.append(f"city: {city}")
        
        if state and not self.service_request.location.state:
            self.service_request.location.state = state.strip().title()
            logger.info(f"🗺️ EXTRACTED - State: {state}")
            extracted_info.append(f"state: {state}")
        
        if zip_code and not self.service_request.location.zip_code:
            self.service_request.location.zip_code = zip_code.strip()
            logger.info(f"📮 EXTRACTED - ZIP: {zip_code}")
            extracted_info.append(f"ZIP: {zip_code}")
        
        if landmarks and not self.service_request.location.landmarks:
            self.service_request.location.landmarks = landmarks.strip()
            logger.info(f"🏛️ EXTRACTED - Landmarks: {landmarks}")
            extracted_info.append(f"landmarks: {landmarks}")
        
        return f"Location information extracted: {', '.join(extracted_info)}" if extracted_info else "No new location information to extract"
    
    @function_tool()
    async def extract_service_info(
        self,
        ctx: RunContext,
        service_type: str,
        job_number: Optional[str] = None,
        special_requirements: Optional[List[str]] = None,
        estimated_cost: Optional[str] = None
    ) -> str:
        """
        Extract and store service information
        
        Args:
            service_type: Type of service needed (towing, jump_start, etc.)
            job_number: Job or reference number
            special_requirements: Special requirements or instructions
            estimated_cost: Estimated cost if mentioned
        """
        extracted_info = []
        
        # Map service types
        service_mapping = {
            "tow": ServiceType.TOWING,
            "towing": ServiceType.TOWING,
            "jump": ServiceType.JUMP_START,
            "jump_start": ServiceType.JUMP_START,
            "battery": ServiceType.JUMP_START,
            "tire": ServiceType.TIRE_CHANGE,
            "tire_change": ServiceType.TIRE_CHANGE,
            "tire_replacement": ServiceType.TIRE_REPLACEMENT,
            "winch": ServiceType.WINCH_OUT,
            "winch_out": ServiceType.WINCH_OUT,
            "lockout": ServiceType.LOCKOUT,
            "fuel": ServiceType.FUEL_DELIVERY,
        }
        
        for key, value in service_mapping.items():
            if key in service_type.lower():
                if self.service_request.service_type == ServiceType.UNKNOWN:
                    self.service_request.service_type = value
                    logger.info(f"🔧 EXTRACTED - Service type: {value.value}")
                    extracted_info.append(f"service: {value.value}")
                break
        
        if job_number and not self.service_request.job_number:
            self.service_request.job_number = job_number.strip()
            logger.info(f"🎫 EXTRACTED - Job number: {job_number}")
            extracted_info.append(f"job number: {job_number}")
        
        if special_requirements:
            self.service_request.special_requirements.extend(special_requirements)
            logger.info(f"📋 EXTRACTED - Special requirements: {special_requirements}")
            extracted_info.append(f"requirements: {', '.join(special_requirements)}")
        
        if estimated_cost and not self.service_request.estimated_cost:
            self.service_request.estimated_cost = estimated_cost.strip()
            logger.info(f"💰 EXTRACTED - Estimated cost: {estimated_cost}")
            extracted_info.append(f"cost: {estimated_cost}")
        
        return f"Service information extracted: {', '.join(extracted_info)}" if extracted_info else "No new service information to extract"
    
    @function_tool()
    async def transfer_to_dispatcher(self, ctx: RunContext) -> str:
        """Transfer call to dispatcher when service request is complete"""
        try:
            # Save monitoring data before transfer
            await self.save_conversation_data()
            
            logger.info("📞 INITIATING TRANSFER TO DISPATCHER")
            return "I'm transferring you to our dispatcher who will coordinate your service. Please hold while I connect you."
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "I'm having trouble with the transfer system. Let me get you help another way."
    
    # ========================================================================
    # TRANSCRIPT AND DATA MANAGEMENT
    # ========================================================================
    
    def add_transcript(self, speaker: str, text: str, is_final: bool = True, confidence: float = None):
        """Add transcript entry with enhanced logging"""
        transcript = CallTranscript(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            is_final=is_final,
            confidence=confidence
        )
        self.transcripts.append(transcript)
        
        # Enhanced logging with speaker identification
        if speaker == "customer":
            logger.info(f"👤 USER TRANSCRIPT: {text}")
        elif speaker == "agent":
            logger.info(f"🤖 AGENT TRANSCRIPT: {text}")
        else:
            logger.info(f"📝 {speaker.upper()}: {text}")
    
    def get_conversation_summary(self) -> Dict:
        """Generate comprehensive conversation summary with structured data"""
        return {
            "room_name": self.room_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            
            # Structured service request
            "service_request": {
                "service_type": self.service_request.service_type.value,
                "customer": asdict(self.service_request.customer),
                "vehicle": asdict(self.service_request.vehicle),
                "location": asdict(self.service_request.location),
                "special_requirements": self.service_request.special_requirements,
                "estimated_cost": self.service_request.estimated_cost,
                "job_number": self.service_request.job_number,
            },
            
            # Raw transcript
            "transcript_count": len(self.transcripts),
            "full_transcript": [
                {
                    "speaker": t.speaker,
                    "text": t.text,
                    "timestamp": t.timestamp.isoformat(),
                    "is_final": t.is_final,
                    "confidence": t.confidence,
                    "extracted_entities": t.extracted_entities,
                    "intent": t.intent
                }
                for t in self.transcripts
            ],
            
            # Completeness analysis
            "data_completeness": self._analyze_data_completeness(),
            "conversation_complete": self.conversation_complete
        }
    
    def _analyze_data_completeness(self) -> Dict[str, bool]:
        """Analyze how complete the extracted data is"""
        return {
            "customer_name": bool(self.service_request.customer.name),
            "customer_phone": bool(self.service_request.customer.phone),
            "vehicle_year": bool(self.service_request.vehicle.year),
            "vehicle_make": bool(self.service_request.vehicle.make),
            "vehicle_model": bool(self.service_request.vehicle.model),
            "location_address": bool(self.service_request.location.address),
            "location_city": bool(self.service_request.location.city),
            "service_type_identified": self.service_request.service_type != ServiceType.UNKNOWN,
            "job_number": bool(self.service_request.job_number),
        }
    
    async def save_conversation_data(self):
        """Save conversation data with structured information in JSON format"""
        summary = self.get_conversation_summary()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"roadside_call_{self.room_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"💾 TRANSCRIPT SAVED: {filename}")
            
            # Log extracted information summary
            customer = self.service_request.customer
            vehicle = self.service_request.vehicle
            location = self.service_request.location
            
            logger.info("📊 EXTRACTED INFORMATION SUMMARY:")
            logger.info(f"   👤 Customer: {customer.name} ({customer.phone})")
            logger.info(f"   🚗 Vehicle: {vehicle.year} {vehicle.make} {vehicle.model}")
            logger.info(f"   📍 Location: {location.address}, {location.city}, {location.state}")
            logger.info(f"   🔧 Service: {self.service_request.service_type.value}")
            logger.info(f"   🎫 Job #: {self.service_request.job_number}")
            logger.info(f"   💰 Cost: {self.service_request.estimated_cost}")
            
            # Log data completeness
            completeness = self._analyze_data_completeness()
            completed_fields = sum(completeness.values())
            total_fields = len(completeness)
            completion_percent = (completed_fields / total_fields) * 100
            
            logger.info(f"📈 DATA COMPLETENESS: {completion_percent:.1f}% ({completed_fields}/{total_fields} fields)")
            
        except Exception as e:
            logger.error(f"❌ FAILED TO SAVE TRANSCRIPT: {e}")

# ============================================================================
# MAIN ENTRYPOINT FUNCTION
# ============================================================================

async def entrypoint(ctx: JobContext):
    """
    COMPLETE ENTRYPOINT FUNCTION
    ✅ Proper session setup with working TTS and transcript saving
    ✅ Enhanced logging for all agent speech
    ✅ Timeout fixes for OpenAI API
    ✅ Event handlers for complete monitoring
    """
    logger.info("🚀 === ROADSIDE ASSISTANCE AGENT WITH INTELLIGENT MONITORING ===")
    
    # Connect immediately
    await ctx.connect()
    logger.info("✅ CONNECTED TO ROOM")
    
    # Initialize RAG system if available
    if RAG_AVAILABLE:
        try:
            await qdrant_rag.initialize()
            logger.info("🧠 RAG STATUS: ✅ Active")
        except Exception as e:
            logger.warning(f"⚠️ RAG initialization failed: {e}, continuing without RAG")
    else:
        logger.info("ℹ️ RAG system not available")
    
    # Create session with FIXED timeout settings and proper configuration
    logger.info("🔧 CREATING AGENT SESSION...")
    session = AgentSession(
        # STT: Deepgram optimized for telephony
        stt=deepgram.STT(
            model="nova-2-phonecall",  # Optimized for phone calls
            language="en-US",
            punctuate=True,
            smart_format=True,
            interim_results=True,
            keywords=[
                ("towing", 2.0), ("jump start", 2.0), ("tire change", 2.0),
                ("phone number", 1.5), ("address", 1.5), ("vehicle", 1.5),
                ("name", 1.5), ("location", 1.5)
            ]
        ),
        
        # LLM: OpenAI with FIXED timeout settings to prevent API timeouts
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.2,  # Consistent responses for professional service
            timeout=httpx.Timeout(
                timeout=30.0,      # Total timeout
                connect=10.0,      # Connection timeout
                read=30.0,         # Read timeout
                write=10.0         # Write timeout
            )
        ),
        
        # TTS: ElevenLabs ONLY - no more dual voice conflict
        tts=elevenlabs.TTS(
            voice_id="ODq5zmih8GrVes37Dizd",  # Professional voice
            model="eleven_turbo_v2_5",
            language="en",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.6,
                similarity_boost=0.8,
                style=0.1,
                speed=1.0,
                use_speaker_boost=True
            )
        ),
        
        # VAD: Silero for voice activity detection
        vad=silero.VAD.load(),
        
        # Turn detection settings - FIXED for better conversation flow
        turn_detection="vad",
        min_endpointing_delay=0.5,  # Wait 500ms after speech stops
        max_endpointing_delay=2.0,  # Max 2 seconds wait
    )
    
    logger.info("✅ AGENT SESSION CREATED")
    
    # Create the unified agent
    agent = RoadsideAssistanceAgent()
    agent.room_name = ctx.room.name
    agent.session = session  # Set session reference
    
    logger.info(f"👤 AGENT CREATED FOR ROOM: {ctx.room.name}")
    
    # ========================================================================
    # ENHANCED EVENT HANDLERS - Compatible with LiveKit v1.1
    # ========================================================================
    
    @session.on("user_input_transcribed")
    def on_user_transcript(event):
        """Capture user transcripts with enhanced logging"""
        if hasattr(event, 'is_final') and event.is_final:
            logger.info(f"👤 USER INPUT (final): {event.transcript}")
            if hasattr(event, 'confidence'):
                logger.info(f"🎯 CONFIDENCE: {event.confidence:.2f}")
            agent.add_transcript("customer", event.transcript, True, getattr(event, 'confidence', None))
        else:
            logger.info(f"👤 USER INPUT (interim): {getattr(event, 'transcript', str(event))}")
    
    @session.on("conversation_item_added")
    def on_conversation_item(event):
        """Capture agent responses and conversation items"""
        logger.info(f"💬 CONVERSATION ITEM ADDED: {getattr(event, 'role', 'unknown')}")
        if hasattr(event, 'item'):
            if hasattr(event.item, 'role') and event.item.role == 'assistant':
                if hasattr(event.item, 'text_content') and event.item.text_content:
                    logger.info(f"🤖 AGENT RESPONSE ADDED TO CONTEXT: {event.item.text_content}")
                    agent.add_transcript("agent", event.item.text_content, True)
        elif hasattr(event, 'text_content'):
            logger.info(f"🤖 AGENT RESPONSE: {event.text_content}")
            agent.add_transcript("agent", event.text_content, True)
    
    @session.on("speech_created")
    def on_speech_created(event):
        """Capture when agent speech is created"""
        logger.info(f"🗣️ AGENT SPEECH CREATED")
        if hasattr(event, 'speech_handle'):
            logger.info(f"🎵 Speech ID: {getattr(event.speech_handle, 'speech_id', 'unknown')}")
        logger.info("🎵 TTS SYNTHESIS STARTED")
    
    @session.on("function_tools_executed")
    def on_function_executed(event):
        """Log when function tools are executed"""
        function_calls = getattr(event, 'function_calls', [])
        logger.info(f"🛠️ FUNCTION TOOLS EXECUTED: {len(function_calls)} calls")
        for i, call in enumerate(function_calls, 1):
            if hasattr(call, 'function_info'):
                logger.info(f"   🔧 Function {i}: {call.function_info.name}")
            if hasattr(call, 'result') and call.result:
                result_preview = str(call.result)[:100]
                logger.info(f"   📊 Result {i}: {result_preview}...")
    
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        """Log agent state changes"""
        state = getattr(event, 'state', 'unknown')
        logger.info(f"🤖 AGENT STATE CHANGED: {state}")
    
    @session.on("user_state_changed")
    def on_user_state_changed(event):
        """Log user state changes"""
        state = getattr(event, 'state', 'unknown')
        logger.info(f"👤 USER STATE CHANGED: {state}")
    
    # Generic error handler - works with different event formats
    try:
        @session.on("error")
        def on_error(event):
            """Handle session errors with proper recovery"""
            error = getattr(event, 'error', event)
            logger.error(f"❌ SESSION ERROR: {error}")
            if hasattr(error, 'recoverable'):
                if error.recoverable:
                    logger.info("🔄 ERROR IS RECOVERABLE - Session will retry")
                else:
                    logger.error("🚨 UNRECOVERABLE ERROR - Manual intervention needed")
                    # Schedule a recovery message
                    asyncio.create_task(session.say(
                        "I'm experiencing technical difficulties. Let me transfer you to a human agent who can help you immediately."
                    ))
    except Exception as e:
        logger.warning(f"⚠️ Could not set up error handler: {e}")
    
    # Alternative event handlers for different LiveKit versions
    try:
        @session.on("agent_started_speaking")
        def on_agent_speaking(event):
            logger.info("🗣️ AGENT STARTED SPEAKING")
        
        @session.on("agent_stopped_speaking") 
        def on_agent_stopped(event):
            logger.info("🔇 AGENT STOPPED SPEAKING")
    except Exception as e:
        logger.info("ℹ️ Advanced speech events not available in this version")
    # Start the session with the agent
    logger.info("🚀 STARTING AGENT SESSION...")
    await session.start(room=ctx.room, agent=agent)
    
    # ========================================================================
    # SYSTEM READY NOTIFICATIONS
    # ========================================================================
    
    logger.info("⚡ ROADSIDE ASSISTANCE SYSTEM READY!")
    logger.info(f"📞 Room: {ctx.room.name}")
    logger.info("🎯 Intelligent Monitoring: ✅ Active")
    logger.info("🔍 Entity Extraction: ✅ Active")
    logger.info("🔧 Speech Logging: ✅ Enhanced")
    logger.info("📝 Transcript Saving: ✅ JSON Format")
    logger.info("⏱️ API Timeouts: ✅ Fixed (30s)")
    logger.info("🎵 TTS Engine: ✅ ElevenLabs Only")
    logger.info("🗣️ Agent Speech: ✅ Fully Logged")
    
    # Professional greeting with enhanced logging
    greeting_text = "Roadside assistance, this is Mark. How can I help you with your vehicle today?"
    logger.info(f"👋 SENDING INITIAL GREETING...")
    
    await session.say(
        greeting_text,
        add_to_chat_ctx=True
    )
    
    # Add to agent's transcript
    agent.add_transcript("agent", greeting_text, True)
    logger.info("✅ INITIAL GREETING SENT AND LOGGED")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        logger.info("🚀 STARTING COMPLETE ADVANCED ROADSIDE ASSISTANCE SYSTEM")
        logger.info("📋 Features: TTS ✅ | Logging ✅ | Extraction ✅ | Transcripts ✅ | Timeouts Fixed ✅")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except KeyboardInterrupt:
        logger.info("🛑 SYSTEM SHUTDOWN REQUESTED")
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)
    finally:
        logger.info("👋 ROADSIDE ASSISTANCE SYSTEM STOPPED")