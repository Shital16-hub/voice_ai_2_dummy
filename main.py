# advanced_monitoring_system.py
"""
Advanced LiveKit Monitoring System for Roadside Assistance
Uses LiveKit's structured tools, LLM function calling, and intelligent entity extraction
Based on real call transcript analysis for towing/roadside assistance
"""
import asyncio
import logging
import json
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
from qdrant_rag_system import qdrant_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class IntelligentMonitoringAgent(Agent):
    """
    Advanced monitoring agent that uses LiveKit's structured tools
    and LLM function calling for intelligent entity extraction
    """
    
    def __init__(self) -> None:
        # Agent with intelligence to extract structured information
        super().__init__(
            instructions="""You are an intelligent call monitoring assistant for roadside assistance services.
            
            Your job is to listen to conversations and extract structured information including:
            - Customer details (name, phone, email)
            - Vehicle information (year, make, model, color)
            - Location details (address, city, state, landmarks)
            - Service type and requirements
            - Job numbers and special instructions
            
            Use the provided tools to extract and structure this information automatically.
            Be very accurate with names, addresses, and phone numbers as these are critical for service dispatch.
            
            Always extract information as you hear it, don't wait for the conversation to end."""
        )
        
        self.transcripts: List[CallTranscript] = []
        self.service_request = ServiceRequest()
        self.room_name = ""
        self.start_time = datetime.now()
        self.conversation_complete = False
        
        # LLM for intelligent extraction
        self.extraction_llm = openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1  # Low temperature for consistent extraction
        )
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        Extract information from user messages using LLM structured extraction
        """
        try:
            user_text = new_message.text_content
            if not user_text:
                return
                
            # Create transcript entry
            transcript = CallTranscript(
                speaker="customer",
                text=user_text,
                timestamp=datetime.now(),
                is_final=True,
                confidence=1.0
            )
            
            # Use LLM tools to extract structured information
            await self._extract_information_with_llm(user_text)
            
            self.transcripts.append(transcript)
            logger.info(f"📝 Customer: {user_text[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ User turn processing error: {e}")
    
    async def transcription_node(self, text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[str]:
        """
        Capture AI agent responses and extract information
        """
        async for delta in text:
            if delta.strip():
                # Create transcript for AI response
                transcript = CallTranscript(
                    speaker="ai_agent",
                    text=delta,
                    timestamp=datetime.now(),
                    is_final=True
                )
                
                # Extract information from AI responses too
                await self._extract_information_with_llm(delta)
                
                self.transcripts.append(transcript)
            
            # Pass through unchanged
            yield delta
    
    @function_tool()
    async def extract_customer_info(self, name: Optional[str] = None, phone: Optional[str] = None, 
                                  email: Optional[str] = None) -> str:
        """
        Extract and store customer information
        
        Args:
            name: Customer's full name
            phone: Customer's phone number
            email: Customer's email address
        """
        if name and not self.service_request.customer.name:
            self.service_request.customer.name = name.strip()
            logger.info(f"👤 Customer name: {name}")
        
        if phone and not self.service_request.customer.phone:
            # Clean phone number
            clean_phone = ''.join(filter(str.isdigit, phone))
            if len(clean_phone) >= 10:
                self.service_request.customer.phone = clean_phone
                logger.info(f"📞 Phone: {clean_phone}")
        
        if email and not self.service_request.customer.email:
            self.service_request.customer.email = email.strip()
            logger.info(f"📧 Email: {email}")
        
        return "Customer information extracted successfully"
    
    @function_tool()
    async def extract_vehicle_info(self, year: Optional[str] = None, make: Optional[str] = None,
                                 model: Optional[str] = None, color: Optional[str] = None,
                                 neutral_functional: Optional[bool] = None) -> str:
        """
        Extract and store vehicle information
        
        Args:
            year: Vehicle year
            make: Vehicle make/manufacturer
            model: Vehicle model
            color: Vehicle color
            neutral_functional: Whether neutral gear is functional
        """
        if year and not self.service_request.vehicle.year:
            self.service_request.vehicle.year = year.strip()
            logger.info(f"🚗 Vehicle year: {year}")
        
        if make and not self.service_request.vehicle.make:
            self.service_request.vehicle.make = make.strip().title()
            logger.info(f"🚗 Vehicle make: {make}")
        
        if model and not self.service_request.vehicle.model:
            self.service_request.vehicle.model = model.strip().title()
            logger.info(f"🚗 Vehicle model: {model}")
        
        if color and not self.service_request.vehicle.color:
            self.service_request.vehicle.color = color.strip().lower()
            logger.info(f"🚗 Vehicle color: {color}")
        
        if neutral_functional is not None and self.service_request.vehicle.neutral_functional is None:
            self.service_request.vehicle.neutral_functional = neutral_functional
            logger.info(f"⚙️ Neutral functional: {neutral_functional}")
        
        return "Vehicle information extracted successfully"
    
    @function_tool()
    async def extract_location_info(self, address: Optional[str] = None, city: Optional[str] = None,
                                  state: Optional[str] = None, zip_code: Optional[str] = None,
                                  landmarks: Optional[str] = None) -> str:
        """
        Extract and store location information
        
        Args:
            address: Street address
            city: City name
            state: State name
            zip_code: ZIP code
            landmarks: Nearby landmarks
        """
        if address and not self.service_request.location.address:
            self.service_request.location.address = address.strip()
            logger.info(f"📍 Address: {address}")
        
        if city and not self.service_request.location.city:
            self.service_request.location.city = city.strip().title()
            logger.info(f"🏙️ City: {city}")
        
        if state and not self.service_request.location.state:
            self.service_request.location.state = state.strip().title()
            logger.info(f"🗺️ State: {state}")
        
        if zip_code and not self.service_request.location.zip_code:
            self.service_request.location.zip_code = zip_code.strip()
            logger.info(f"📮 ZIP: {zip_code}")
        
        if landmarks and not self.service_request.location.landmarks:
            self.service_request.location.landmarks = landmarks.strip()
            logger.info(f"🏛️ Landmarks: {landmarks}")
        
        return "Location information extracted successfully"
    
    @function_tool()
    async def extract_service_info(self, service_type: str, job_number: Optional[str] = None,
                                 special_requirements: Optional[List[str]] = None,
                                 estimated_cost: Optional[str] = None) -> str:
        """
        Extract and store service information
        
        Args:
            service_type: Type of service needed (towing, jump_start, etc.)
            job_number: Job or reference number
            special_requirements: Special requirements or instructions
            estimated_cost: Estimated cost if mentioned
        """
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
                self.service_request.service_type = value
                logger.info(f"🔧 Service type: {value.value}")
                break
        
        if job_number and not self.service_request.job_number:
            self.service_request.job_number = job_number.strip()
            logger.info(f"🎫 Job number: {job_number}")
        
        if special_requirements:
            self.service_request.special_requirements.extend(special_requirements)
            logger.info(f"📋 Special requirements: {special_requirements}")
        
        if estimated_cost and not self.service_request.estimated_cost:
            self.service_request.estimated_cost = estimated_cost.strip()
            logger.info(f"💰 Estimated cost: {estimated_cost}")
        
        return "Service information extracted successfully"
    
    async def _extract_information_with_llm(self, text: str):
        """
        Use LLM to intelligently extract structured information from text
        """
        try:
            # Create a chat context for extraction
            extraction_prompt = f"""
            Extract structured information from this roadside assistance conversation text:
            
            Text: "{text}"
            
            Look for and extract:
            1. Customer information (names, phone numbers, email)
            2. Vehicle details (year, make, model, color, neutral gear status)
            3. Location information (addresses, cities, states, ZIP codes, landmarks)
            4. Service type and requirements
            5. Job numbers or reference numbers
            6. Cost information
            
            Use the available tools to extract this information. Only extract information that is clearly mentioned.
            Be very accurate with names, phone numbers, and addresses.
            """
            
            chat_ctx = ChatContext()
            chat_ctx.add_message(role="user", content=extraction_prompt)
            
            # Run LLM with tools
            response_stream = self.extraction_llm.chat(
                chat_ctx=chat_ctx,
                fnc_ctx=self._get_function_context()
            )
            
            # Process the response (this will trigger tool calls)
            async for chunk in response_stream:
                # Tool calls are handled automatically by the LLM
                pass
                
        except Exception as e:
            logger.error(f"❌ LLM extraction error: {e}")
    
    def _get_function_context(self):
        """Get function context with our extraction tools"""
        from livekit.agents import function_context
        
        ctx = function_context.FunctionContext()
        # Add our tools to the context
        ctx.add_function_tool(self.extract_customer_info)
        ctx.add_function_tool(self.extract_vehicle_info)
        ctx.add_function_tool(self.extract_location_info)
        ctx.add_function_tool(self.extract_service_info)
        
        return ctx
    
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
        """Save conversation data with structured information"""
        summary = self.get_conversation_summary()
        
        # Save to JSON file (replace with database in production)
        filename = f"roadside_call_{self.room_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Call data saved: {filename}")
        
        # Log extracted information summary
        customer = self.service_request.customer
        vehicle = self.service_request.vehicle
        location = self.service_request.location
        
        logger.info("📊 EXTRACTED INFORMATION SUMMARY:")
        logger.info(f"   Customer: {customer.name} ({customer.phone})")
        logger.info(f"   Vehicle: {vehicle.year} {vehicle.make} {vehicle.model}")
        logger.info(f"   Location: {location.address}, {location.city}, {location.state}")
        logger.info(f"   Service: {self.service_request.service_type.value}")
        logger.info(f"   Job #: {self.service_request.job_number}")
        logger.info(f"   Cost: {self.service_request.estimated_cost}")

# Enhanced main agent that works with monitoring
class EnhancedRoadsideAgent(Agent):
    """
    Your existing roadside assistance agent enhanced with monitoring integration
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance agent.
            
            Your job is to help customers with towing, jump starts, tire changes, winch-outs, and other roadside services.
            
            Always collect this information in this order:
            1. Customer's full name
            2. Phone number
            3. Exact location (full address, city, state, ZIP)
            4. Vehicle year, make, and model
            5. Type of service needed
            6. For towing: Ask if neutral gear is functional
            
            Be professional, efficient, and empathetic. Provide cost estimates when appropriate.
            Transfer to dispatcher when ready to complete the service request."""
        )
        self.monitoring_agent: Optional[IntelligentMonitoringAgent] = None
    
    def set_monitoring_agent(self, monitoring_agent: IntelligentMonitoringAgent):
        """Connect the monitoring agent"""
        self.monitoring_agent = monitoring_agent
    
    # Your existing function tools would go here
    @function_tool()
    async def transfer_to_dispatcher(self, ctx: RunContext) -> str:
        """Transfer call to dispatcher when service request is complete"""
        try:
            # Save monitoring data before transfer
            if self.monitoring_agent:
                await self.monitoring_agent.save_conversation_data()
            
            await ctx.session.generate_reply(
                instructions="Say: 'I'm transferring you to our dispatcher who will coordinate your service. Please hold.'"
            )
            
            return "Transfer to dispatcher completed successfully"
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "I'm having trouble with the transfer. Let me get you help another way."

async def create_enhanced_session() -> AgentSession:
    """Create session optimized for roadside assistance monitoring"""
    
    # Use Deepgram optimized for telephony
    stt_engine = deepgram.STT(
        model="nova-2-phonecall",  # Optimized for phone calls
        language="en-US",
        punctuate=True,
        smart_format=True,
        interim_results=True,
        keywords=[
            ("towing", 2.0), ("jump start", 2.0), ("tire change", 2.0),
            ("phone number", 1.5), ("address", 1.5), ("vehicle", 1.5)
        ]
    )
    
    # Use ElevenLabs for consistent voice
    tts_engine = elevenlabs.TTS(
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
    )
    
    session = AgentSession(
        stt=stt_engine,
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.2,  # Consistent responses for professional service
        ),
        tts=tts_engine,
        vad=silero.VAD.load(),
        turn_detection="stt",
        min_endpointing_delay=0.4,
        max_endpointing_delay=2.5,
        allow_interruptions=True,
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """
    Enhanced entrypoint with intelligent monitoring for roadside assistance
    """
    logger.info("=== ROADSIDE ASSISTANCE AGENT WITH INTELLIGENT MONITORING ===")
    
    # Connect immediately
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Initialize systems
    init_tasks = [
        qdrant_rag.initialize(),
        create_enhanced_session()
    ]
    
    rag_success, session = await asyncio.gather(*init_tasks)
    
    # Create agents
    main_agent = EnhancedRoadsideAgent()
    monitoring_agent = IntelligentMonitoringAgent()
    
    # Connect agents
    main_agent.set_monitoring_agent(monitoring_agent)
    monitoring_agent.room_name = ctx.room.name
    
    # Start main session
    await session.start(room=ctx.room, agent=main_agent)
    
    # Start monitoring session in background
    monitor_session = AgentSession(
        stt=deepgram.STT(model="nova-2-phonecall", language="en-US"),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.1),
        tts=openai.TTS(model="tts-1", voice="nova"),  # Minimal TTS for monitoring
        vad=silero.VAD.load(),
    )
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(
        monitor_session.start(room=ctx.room, agent=monitoring_agent)
    )
    
    # Professional greeting
    await session.generate_reply(
        instructions="Greet the caller professionally as Mark from roadside assistance and ask how you can help them today."
    )
    
    logger.info("⚡ ROADSIDE ASSISTANCE SYSTEM READY!")
    logger.info(f"📞 Room: {ctx.room.name}")
    logger.info(f"🧠 RAG Status: {'✅ Active' if rag_success else '⚠️ Fallback'}")
    logger.info("🎯 Intelligent Monitoring: ✅ Active")
    logger.info("🔍 Entity Extraction: ✅ Active")

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Advanced Roadside Assistance System")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)