# fixed_multi_agent_orchestrator.py - FIXED STT CONFIGURATION
"""
Multi-Agent Orchestrator with IMPROVED STT for better word recognition
FIXED: Better Deepgram configuration for telephony accuracy
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from dotenv import load_dotenv
from livekit import api, agents
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context
)
from livekit.plugins import deepgram, openai, elevenlabs, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from qdrant_rag_system import qdrant_rag
from config import config

# Import the fixed CallData from enhanced_conversational_agent
from enhanced_conversational_agent import CallData

load_dotenv()
logger = logging.getLogger(__name__)

class AgentType(Enum):
    DISPATCHER = "dispatcher"
    TOWING_SPECIALIST = "towing_specialist"
    BATTERY_SPECIALIST = "battery_specialist"
    TIRE_SPECIALIST = "tire_specialist"
    EMERGENCY_RESPONSE = "emergency_response"
    INSURANCE_SPECIALIST = "insurance_specialist"
    CUSTOMER_SERVICE = "customer_service"

@dataclass
class HandoffContext:
    """Context information for agent handoffs"""
    reason: str
    previous_agent: str
    conversation_summary: str
    urgency_level: str
    collected_info: Dict[str, Any]
    next_steps: List[str] = field(default_factory=list)

class DispatcherAgent(Agent):
    """Main dispatcher that routes calls to appropriate specialists"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance dispatcher. 

CONVERSATION FLOW (gather ONE piece at a time):
1. If no name: "Could you please provide your full name?"
2. If no phone: "Could you also provide a good phone number where we can reach you?"
3. If no location: "What is the exact location of your vehicle?"
4. If no vehicle info: "Could you tell me the year, make, and model of your vehicle?"
5. If no service type: "What type of service do you need today?"

RESPONSE STYLE:
- Keep under 25 words for phone clarity
- Confirm unclear information: "Just to confirm, you said..."
- Be patient and professional
- Use context clues to understand intent

USE FUNCTIONS:
- search_knowledge: For pricing, services, policies
- gather_caller_information: Store information as gathered
- route_to_specialist: For complex issues needing expertise"""
        )
        self.call_start_time = time.time()
        
    # Let the LLM handle conversation flow naturally - no hardcoded patterns needed

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search knowledge base - FIXED with better error handling"""
        try:
            logger.info(f"🔍 SEARCHING KNOWLEDGE BASE: {query}")
            
            results = await asyncio.wait_for(
                qdrant_rag.search(query, limit=2),
                timeout=2.0
            )
            
            if results and len(results) > 0:
                relevant_results = []
                for result in results:
                    if result["score"] >= 0.15:
                        relevant_results.append(result)
                
                if relevant_results:
                    response_parts = []
                    for result in relevant_results[:2]:
                        text = result["text"]
                        cleaned = text.replace("Q:", "").replace("A:", "").strip()
                        if len(cleaned) > 20:
                            response_parts.append(cleaned)
                    
                    if response_parts:
                        combined_response = " | ".join(response_parts)
                        logger.info(f"📊 Found {len(relevant_results)} relevant results")
                        return combined_response[:200]
                
            logger.warning(f"⚠️ No relevant results found for: {query}")
            return "I don't have specific information about that. Let me connect you with a specialist who can provide detailed information."
                
        except asyncio.TimeoutError:
            logger.error("❌ Knowledge search timeout")
            return "I'm having trouble accessing the information right now. Let me transfer you to someone who can help."
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return "I'm having trouble accessing the information. Let me transfer you to someone who can help."

    @function_tool()
    async def gather_caller_information(
        self, 
        context: RunContext[CallData],
        name: str = None,
        phone: str = None,
        location: str = None,
        vehicle_year: str = None,
        vehicle_make: str = None,
        vehicle_model: str = None,
        vehicle_color: str = None,
        issue: str = None,
        service_needed: str = None
    ) -> str:
        """Store caller information as it's gathered one piece at a time"""
        
        if name:
            context.userdata.caller_name = name
            context.userdata.gathered_info["name"] = True
            return "Thank you. Could you also provide a good phone number where we can reach you?"
            
        if phone:
            context.userdata.phone_number = phone
            context.userdata.gathered_info["phone"] = True
            return "Got it. What is the exact location of your vehicle?"
            
        if location:
            context.userdata.location = location
            context.userdata.gathered_info["location"] = True
            return "Perfect. Could you tell me the year, make, and model of your vehicle?"
            
        if any([vehicle_year, vehicle_make, vehicle_model, vehicle_color]):
            if vehicle_year:
                context.userdata.vehicle_year = vehicle_year
            if vehicle_make:
                context.userdata.vehicle_make = vehicle_make
            if vehicle_model:
                context.userdata.vehicle_model = vehicle_model
            if vehicle_color:
                context.userdata.vehicle_color = vehicle_color
            context.userdata.gathered_info["vehicle"] = True
            return "Great. What type of service do you need today?"
            
        if service_needed:
            context.userdata.service_type = service_needed
            context.userdata.gathered_info["service"] = True
            return "Perfect! I have all your information. Let me find the best service options for you."
        
        return "Could you provide that information again?"

    @function_tool()
    async def route_to_specialist(
        self, 
        context: RunContext[CallData],
        specialist_type: str,
        reason: str = "Specialized assistance needed"
    ) -> str:
        """Route call to appropriate specialist agent"""
        
        logger.info(f"🔄 Routing to {specialist_type}: {reason}")
        specialist_name = specialist_type.replace('_', ' ').title()
        return f"I'm connecting you with our {specialist_name} who can provide expert assistance. Please hold."

# Keep all the specialist classes the same...
class TowingSpecialistAgent(Agent):
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"Towing specialist. Context: {handoff_context.conversation_summary}")

class BatterySpecialistAgent(Agent):
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"Battery specialist. Context: {handoff_context.conversation_summary}")

class TireSpecialistAgent(Agent):
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"Tire specialist. Context: {handoff_context.conversation_summary}")

class EmergencyResponseAgent(Agent):
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"Emergency response. Context: {handoff_context.conversation_summary}")

class CustomerServiceAgent(Agent):
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"Customer service. Context: {handoff_context.conversation_summary}")

async def create_optimized_session(userdata: CallData) -> AgentSession[CallData]:
    """Create session with IMPROVED STT configuration for telephony"""
    
    session = AgentSession[CallData](
        # 🔧 PROPER STT CONFIGURATION FOR TELEPHONY
        stt=deepgram.STT(
            model="nova-2-phonecall",    # CORRECT: Telephony-optimized model
            language="en-US",
            
            # LiveKit Deepgram STT supported parameters only
            smart_format=True,           # Better formatting
            punctuate=True,             # Add punctuation
            profanity_filter=False,     # Allow natural speech
            numerals=True,              # Convert numbers properly
            interim_results=True,       # Get partial results
            endpointing_ms=300,         # 300ms silence detection for end of speech
            filler_words=True,          # Handle "um", "uh" naturally
        ),
        
        # Faster LLM settings
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
        ),
        
        # Optimized TTS for clarity
        tts=elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.7,           # More stable for phone
                similarity_boost=0.8,
                style=0.1,
                speed=0.9                # Slightly slower for clarity
            ),
            model="eleven_turbo_v2_5",
        ),
        
        # Optimized VAD for telephony
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        
        # TELEPHONY-OPTIMIZED TIMING
        allow_interruptions=True,
        min_interruption_duration=0.5,   # Allow natural interruptions
        min_endpointing_delay=0.8,       # Wait for user to finish
        max_endpointing_delay=3.0,       # Don't wait too long
        
        userdata=userdata
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """Entrypoint with IMPROVED STT for better word recognition"""
    
    logger.info("🚀 Multi-Agent System with Proper STT Starting")
    logger.info("🔧 Using nova-2-phonecall model for telephony accuracy")
    
    await ctx.connect()
    
    # Initialize knowledge base in background
    asyncio.create_task(qdrant_rag.initialize())
    
    # Create call data and session
    call_data = CallData()
    session = await create_optimized_session(call_data)
    
    # Start with dispatcher agent
    dispatcher = DispatcherAgent()
    
    await session.start(
        agent=dispatcher,
        room=ctx.room
    )
    
    # Give greeting
    await session.generate_reply(
        instructions="Give a clear, slow greeting: 'Roadside assistance, this is Mark, how can I help you?'"
    )
    
    logger.info("✅ Multi-agent system ready with proper STT")
    logger.info("🎯 Using Deepgram nova-2-phonecall for better telephony accuracy")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting Multi-Agent System with Proper STT")
        logger.info("🔧 STT: Deepgram nova-2-phonecall for telephony")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)