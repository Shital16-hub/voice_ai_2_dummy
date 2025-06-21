# final_working_multi_agent.py - BASED ON OFFICIAL LIVEKIT DOCUMENTATION
"""
Multi-Agent System with Proper on_enter() Lifecycle Hooks
SOLUTION: Based on official LiveKit Pipeline Nodes & Hooks documentation
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

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

# Import the CallData from enhanced_conversational_agent
from enhanced_conversational_agent import CallData

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class CustomerData:
    """Customer information that persists across agent handoffs"""
    name: str = ""
    phone: str = ""
    location: str = ""
    vehicle_year: str = ""
    vehicle_make: str = ""
    vehicle_model: str = ""
    service_type: str = ""
    
    def get_summary(self) -> str:
        """Create a summary for agent handoff"""
        parts = []
        if self.name: parts.append(f"Customer: {self.name}")
        if self.phone: parts.append(f"Phone: {self.phone}")
        if self.location: parts.append(f"Location: {self.location}")
        if self.vehicle_year or self.vehicle_make:
            vehicle = f"{self.vehicle_year} {self.vehicle_make} {self.vehicle_model}".strip()
            parts.append(f"Vehicle: {vehicle}")
        if self.service_type: parts.append(f"Service: {self.service_type}")
        return " | ".join(parts) if parts else "No information collected yet"

class DispatcherAgent(Agent):
    """Main dispatcher that collects info and routes to specialists"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance dispatcher.

TASK: Collect customer information step by step, then route to the appropriate specialist.

INFORMATION GATHERING SEQUENCE:
1. Full name
2. Phone number
3. Vehicle location (complete address)
4. Vehicle details (year, make, model)
5. Service type needed

ROUTING RULES:
- For towing: Use transfer_to_towing_specialist()
- For battery issues: Use transfer_to_battery_specialist()
- For tire problems: Use transfer_to_tire_specialist()

STYLE:
- Ask ONE question at a time
- Be empathetic and professional
- Keep responses under 25 words for phone clarity
- Confirm information when provided"""
        )

    async def on_enter(self):
        """Called when this agent becomes active"""
        await self.session.generate_reply(
            instructions="Give a professional greeting: 'Roadside assistance, this is Mark, how can I help you today?'"
        )

    @function_tool()
    async def collect_information(
        self,
        context: RunContext[CallData],
        name: str = None,
        phone: str = None,
        location: str = None,
        vehicle_year: str = None,
        vehicle_make: str = None,
        vehicle_model: str = None,
        service_type: str = None
    ) -> str:
        """Collect customer information step by step"""
        
        # Initialize customer data in userdata if needed
        if not hasattr(context.userdata, 'customer'):
            context.userdata.customer = CustomerData()
        
        customer = context.userdata.customer
        
        # Update with provided information
        if name: customer.name = name
        if phone: customer.phone = phone
        if location: customer.location = location
        if vehicle_year: customer.vehicle_year = vehicle_year
        if vehicle_make: customer.vehicle_make = vehicle_make
        if vehicle_model: customer.vehicle_model = vehicle_model
        if service_type: customer.service_type = service_type
        
        # Determine what to ask next
        if not customer.name:
            return "Could you please provide your full name?"
        elif not customer.phone:
            return "Thank you. Could you provide a phone number where we can reach you?"
        elif not customer.location:
            return "Got it. What is the exact location of your vehicle? Please provide the complete address."
        elif not customer.vehicle_year or not customer.vehicle_make:
            return "Perfect. What year, make, and model is your vehicle?"
        elif not customer.service_type:
            return "Great. What type of service do you need today?"
        else:
            return "Excellent! I have all your information. Let me connect you with the right specialist."

    @function_tool()
    async def transfer_to_towing_specialist(self, context: RunContext[CallData]) -> Agent:
        """Transfer to towing specialist with customer context"""
        logger.info("🔄 TRANSFERRING TO TOWING SPECIALIST")
        
        customer = getattr(context.userdata, 'customer', CustomerData())
        return TowingSpecialistAgent(customer)

    @function_tool()
    async def transfer_to_battery_specialist(self, context: RunContext[CallData]) -> Agent:
        """Transfer to battery specialist with customer context"""
        logger.info("🔄 TRANSFERRING TO BATTERY SPECIALIST")
        
        customer = getattr(context.userdata, 'customer', CustomerData())
        return BatterySpecialistAgent(customer)

    @function_tool()
    async def transfer_to_tire_specialist(self, context: RunContext[CallData]) -> Agent:
        """Transfer to tire specialist with customer context"""
        logger.info("🔄 TRANSFERRING TO TIRE SPECIALIST")
        
        customer = getattr(context.userdata, 'customer', CustomerData())
        return TireSpecialistAgent(customer)

class TowingSpecialistAgent(Agent):
    """Towing specialist with proper on_enter() greeting"""
    
    def __init__(self, customer: CustomerData):
        self.customer = customer
        
        super().__init__(
            instructions=f"""You are a TOWING SPECIALIST.

CUSTOMER INFORMATION (already collected by dispatcher):
{customer.get_summary()}

CRITICAL: DO NOT ask for name, phone, location, or vehicle info again! You already have it.

YOUR ROLE:
- Assess towing requirements and destination
- Provide distance-based pricing quotes
- Arrange towing service scheduling
- Handle special vehicle requirements (AWD, low clearance, etc.)

Use search_towing_rates for current pricing information.
Keep responses professional and under 30 words for phone clarity."""
        )

    async def on_enter(self):
        """CRITICAL: Called when this agent becomes active - greet with context"""
        vehicle_info = f"{self.customer.vehicle_year} {self.customer.vehicle_make} {self.customer.vehicle_model}".strip()
        location = self.customer.location or "your location"
        
        await self.session.generate_reply(
            instructions=f"Greet as towing specialist: 'Hi {self.customer.name}, I'm your towing specialist. I see you need towing for your {vehicle_info} at {location}. Where would you like it towed to?'"
        )

    @function_tool()
    async def search_towing_rates(self, context: RunContext[CallData], query: str) -> str:
        """Search for towing rates and policies"""
        try:
            enhanced_query = f"towing service pricing rates {query}"
            results = await asyncio.wait_for(qdrant_rag.search(enhanced_query, limit=1), timeout=0.5)
            if results and results[0]["score"] >= 0.2:
                return results[0]["text"][:120]
            return "Let me check our current towing rates and availability."
        except Exception:
            return "I'll verify those details for you."

    @function_tool()
    async def calculate_towing_quote(
        self,
        context: RunContext[CallData],
        distance_miles: float,
        destination: str = "repair shop"
    ) -> str:
        """Calculate towing quote based on distance"""
        base_rate = 75
        per_mile = 3.50
        total = base_rate + (distance_miles * per_mile)
        
        return f"Towing to {destination}: ${base_rate} hookup + ${per_mile}/mile × {distance_miles:.1f} miles = ${total:.2f} total. ETA: 30-45 minutes."

class BatterySpecialistAgent(Agent):
    """Battery specialist with proper on_enter() greeting"""
    
    def __init__(self, customer: CustomerData):
        self.customer = customer
        
        super().__init__(
            instructions=f"""You are a BATTERY SPECIALIST.

CUSTOMER INFO: {customer.get_summary()}

DO NOT re-ask for basic information! Focus on:
- Battery symptoms and diagnosis
- Jump start vs replacement options
- Service scheduling and pricing

Use search_battery_services for service information."""
        )

    async def on_enter(self):
        """Called when this agent becomes active"""
        await self.session.generate_reply(
            instructions=f"Greet as battery specialist: 'Hi {self.customer.name}, I'm your battery specialist. I have your location and vehicle info. What battery problems are you experiencing?'"
        )

    @function_tool()
    async def search_battery_services(self, context: RunContext[CallData], query: str) -> str:
        """Search battery service information"""
        try:
            enhanced_query = f"battery jumpstart replacement service {query}"
            results = await asyncio.wait_for(qdrant_rag.search(enhanced_query, limit=1), timeout=0.5)
            if results and results[0]["score"] >= 0.2:
                return results[0]["text"][:120]
            return "Let me check our battery service options."
        except Exception:
            return "I'll verify our battery services."

class TireSpecialistAgent(Agent):
    """Tire specialist with proper on_enter() greeting"""
    
    def __init__(self, customer: CustomerData):
        self.customer = customer
        
        super().__init__(
            instructions=f"""You are a TIRE SPECIALIST.

CUSTOMER INFO: {customer.get_summary()}

Focus on tire service needs:
- Type of tire damage assessment
- Spare tire availability check
- Repair vs replacement recommendations

Use search_tire_services for service information."""
        )

    async def on_enter(self):
        """Called when this agent becomes active"""
        await self.session.generate_reply(
            instructions=f"Greet as tire specialist: 'Hi {self.customer.name}, I'm your tire specialist. I have your vehicle and location info. What's the tire problem? Do you have a spare tire available?'"
        )

    @function_tool()
    async def search_tire_services(self, context: RunContext[CallData], query: str) -> str:
        """Search tire service information"""
        try:
            enhanced_query = f"tire flat repair change service {query}"
            results = await asyncio.wait_for(qdrant_rag.search(enhanced_query, limit=1), timeout=0.5)
            if results and results[0]["score"] >= 0.2:
                return results[0]["text"][:120]
            return "Let me check our tire service options."
        except Exception:
            return "I'll verify our tire services."

async def create_session_with_userdata(call_data: CallData) -> AgentSession[CallData]:
    """Create session with proper userdata support for agent handoffs"""
    
    session = AgentSession[CallData](
        stt=deepgram.STT(
            model="nova-2-general",
            language="en-US",
            smart_format=True,
            profanity_filter=False,
            numerals=True,
        ),
        
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
        ),
        
        tts=elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.6,
                similarity_boost=0.7,
                style=0.0,
                speed=1.0
            ),
            model="eleven_turbo_v2_5",
        ),
        
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        
        allow_interruptions=True,
        min_interruption_duration=0.4,
        min_endpointing_delay=0.6,
        max_endpointing_delay=2.5,
        
        userdata=call_data
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """Entrypoint with proper agent lifecycle management"""
    
    logger.info("🚀 FINAL Multi-Agent System with on_enter() Lifecycle Hooks")
    logger.info("✅ Based on official LiveKit Pipeline Nodes & Hooks documentation")
    
    await ctx.connect()
    
    # Initialize Qdrant in background
    asyncio.create_task(qdrant_rag.initialize())
    
    # Create call data and session
    call_data = CallData()
    session = await create_session_with_userdata(call_data)
    
    # Start with dispatcher agent (on_enter() will be called automatically)
    dispatcher = DispatcherAgent()
    
    await session.start(
        agent=dispatcher,
        room=ctx.room
    )
    
    logger.info("✅ Multi-agent system ready with proper lifecycle hooks")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting FINAL Multi-Agent System")
        logger.info("🔄 Agent handoffs with proper on_enter() lifecycle hooks")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)