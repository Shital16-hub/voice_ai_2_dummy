# multi_agent_orchestrator.py - FIXED VERSION
"""
Multi-Agent Orchestrator for Specialized Roadside Assistance
FIXED: Updated to use correct turn detector imports
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
# FIXED: Import the correct turn detector classes
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
            instructions="""You are Mark, the main dispatcher for roadside assistance. Your role is to:

1. Conduct initial assessment and gather basic information
2. Determine the type of service needed
3. Route complex cases to appropriate specialists
4. Handle simple inquiries directly using knowledge base

ROUTING DECISIONS:
- Complex towing situations → Towing Specialist
- Battery/electrical issues → Battery Specialist  
- Tire problems → Tire Specialist
- Emergency situations → Emergency Response
- Insurance/coverage questions → Insurance Specialist
- General inquiries → Handle directly

Always gather: name, phone, location, vehicle info, and problem description before routing.
Use search_knowledge for any service or pricing questions.
Only transfer when specialist expertise is truly needed."""
        )
        self.call_start_time = time.time()
        self.specialist_agents = {}
        
    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search knowledge base for service information"""
        try:
            results = await asyncio.wait_for(
                qdrant_rag.search(query, limit=2),
                timeout=1.0
            )
            
            if results and results[0]["score"] >= 0.25:
                response_parts = []
                for result in results[:2]:
                    if result["score"] >= 0.25:
                        formatted = self._format_for_voice(result["text"])
                        if formatted:
                            response_parts.append(formatted)
                
                if response_parts:
                    return " | ".join(response_parts)
            
            return "Let me connect you with a specialist who can provide detailed information about that."
                
        except Exception as e:
            logger.error(f"Knowledge search error: {e}")
            return "I'll transfer you to someone who can help with that specific question."

    def _format_for_voice(self, text: str) -> str:
        """Format text for voice delivery"""
        cleaned = text.replace("Q:", "").replace("A:", "").replace("•", "").strip()
        sentences = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 10]
        if sentences:
            result = sentences[0]
            return result[:120] + "..." if len(result) > 120 else result
        return cleaned[:100] if cleaned else ""

    @function_tool()
    async def route_to_specialist(
        self, 
        context: RunContext[CallData],
        specialist_type: str,
        reason: str
    ) -> Agent:
        """Route call to appropriate specialist agent"""
        
        logger.info(f"🔄 Routing to {specialist_type}: {reason}")
        
        # Create handoff context
        handoff = HandoffContext(
            reason=reason,
            previous_agent="dispatcher",
            conversation_summary=self._create_conversation_summary(context.userdata),
            urgency_level=context.userdata.urgency_level,
            collected_info=self._gather_collected_info(context.userdata)
        )
        
        # Store handoff context
        context.userdata.conversation_history.append(f"Routing to {specialist_type}: {reason}")
        
        # Create or get specialist agent
        specialist = self._get_specialist_agent(specialist_type, handoff)
        
        # Inform user about transfer
        await context.session.generate_reply(
            instructions=f"Tell the customer you're connecting them with our {specialist_type.replace('_', ' ')} specialist who can provide expert assistance with their specific situation. Keep it brief and professional."
        )
        
        return specialist

    def _create_conversation_summary(self, call_data: CallData) -> str:
        """Create summary of conversation for handoff"""
        summary_parts = []
        
        if call_data.caller_name:
            summary_parts.append(f"Customer: {call_data.caller_name}")
        if call_data.phone_number:
            summary_parts.append(f"Phone: {call_data.phone_number}")
        if call_data.location:
            summary_parts.append(f"Location: {call_data.location}")
        if call_data.vehicle_make or call_data.vehicle_model:
            vehicle = f"{call_data.vehicle_year or ''} {call_data.vehicle_make or ''} {call_data.vehicle_model or ''}".strip()
            summary_parts.append(f"Vehicle: {vehicle}")
        if call_data.issue_description:
            summary_parts.append(f"Issue: {call_data.issue_description}")
        if call_data.service_type:
            summary_parts.append(f"Service: {call_data.service_type}")
            
        return " | ".join(summary_parts)

    def _gather_collected_info(self, call_data: CallData) -> Dict[str, Any]:
        """Gather all collected information for handoff"""
        return {
            "name": call_data.caller_name,
            "phone": call_data.phone_number,
            "location": call_data.location,
            "vehicle": {
                "year": call_data.vehicle_year,
                "make": call_data.vehicle_make,
                "model": call_data.vehicle_model,
                "color": call_data.vehicle_color
            },
            "service_type": call_data.service_type,
            "issue": call_data.issue_description,
            "urgency": call_data.urgency_level,
            "call_stage": call_data.call_stage
        }

    def _get_specialist_agent(self, specialist_type: str, handoff: HandoffContext) -> Agent:
        """Get or create specialist agent"""
        if specialist_type not in self.specialist_agents:
            if specialist_type == "towing_specialist":
                self.specialist_agents[specialist_type] = TowingSpecialistAgent(handoff)
            elif specialist_type == "battery_specialist":
                self.specialist_agents[specialist_type] = BatterySpecialistAgent(handoff)
            elif specialist_type == "tire_specialist":
                self.specialist_agents[specialist_type] = TireSpecialistAgent(handoff)
            elif specialist_type == "emergency_response":
                self.specialist_agents[specialist_type] = EmergencyResponseAgent(handoff)
            elif specialist_type == "insurance_specialist":
                self.specialist_agents[specialist_type] = InsuranceSpecialistAgent(handoff)
            else:
                self.specialist_agents[specialist_type] = CustomerServiceAgent(handoff)
        
        return self.specialist_agents[specialist_type]

class TowingSpecialistAgent(Agent):
    """Specialist for complex towing situations"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are a towing specialist with expert knowledge of all towing services.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}
URGENCY: {handoff_context.urgency_level}

SPECIALTIES:
- Long-distance towing arrangements
- Heavy vehicle towing
- Accident recovery
- Complex vehicle situations
- Towing equipment requirements
- Route planning and logistics

Use search_knowledge for specific towing rates, equipment capabilities, and service areas.
Always confirm towing destination and any special requirements.
Provide accurate time estimates and coordinate with tow truck dispatch."""
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search for towing-specific information"""
        enhanced_query = f"towing {query}"
        return await self._search_specialist_knowledge(enhanced_query)

    async def _search_specialist_knowledge(self, query: str) -> str:
        """Enhanced knowledge search for specialists"""
        try:
            results = await qdrant_rag.search(query, limit=3)
            if results:
                best_results = [r for r in results if r["score"] >= 0.2]
                if best_results:
                    formatted_results = []
                    for result in best_results[:2]:
                        cleaned = result["text"].replace("Q:", "").replace("A:", "").strip()
                        sentences = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 10]
                        if sentences:
                            formatted_results.append(sentences[0])
                    return " | ".join(formatted_results) if formatted_results else "Let me get specific details on that for you."
            return "Let me check on those specific details and get back to you."
        except Exception:
            return "I'll need to verify those details for you."

class BatterySpecialistAgent(Agent):
    """Specialist for battery and electrical issues"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are a battery and electrical systems specialist.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}

SPECIALTIES:
- Battery testing and diagnosis
- Jump start services
- Electrical system troubleshooting
- Battery replacement recommendations
- Charging system issues
- Cold weather battery problems

Use search_knowledge for battery service pricing and technical specifications.
Always ask about symptoms: lights dimming, clicking sounds, age of battery, etc.
Provide preventive maintenance advice."""
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search for battery-specific information"""
        enhanced_query = f"battery jumpstart electrical {query}"
        return await self._search_specialist_knowledge(enhanced_query)

    async def _search_specialist_knowledge(self, query: str) -> str:
        """Battery specialist knowledge search"""
        try:
            results = await qdrant_rag.search(query, limit=3)
            if results and results[0]["score"] >= 0.2:
                return self._format_battery_response(results[0]["text"])
            return "Let me look up the specific battery service information for your situation."
        except Exception:
            return "I'll get the exact battery service details for you."

    def _format_battery_response(self, text: str) -> str:
        """Format battery-specific responses"""
        cleaned = text.replace("Q:", "").replace("A:", "").strip()
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
        return cleaned

class TireSpecialistAgent(Agent):
    """Specialist for tire-related services"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are a tire service specialist with expertise in all tire-related issues.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}

SPECIALTIES:
- Flat tire repair and replacement
- Tire mounting and balancing
- Tire pressure issues
- Spare tire installation
- Run-flat tire services
- Tire safety assessments

Use search_knowledge for tire service pricing and technical requirements.
Always check: tire size, spare tire availability, location safety for service.
Provide tire safety recommendations."""
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search for tire-specific information"""
        enhanced_query = f"tire flat spare wheel {query}"
        return await self._search_specialist_knowledge(enhanced_query)

    async def _search_specialist_knowledge(self, query: str) -> str:
        """Tire specialist knowledge search"""
        try:
            results = await qdrant_rag.search(query, limit=2)
            if results and results[0]["score"] >= 0.2:
                formatted = results[0]["text"].replace("Q:", "").replace("A:", "").strip()
                return formatted[:120] + "..." if len(formatted) > 120 else formatted
            return "Let me get the specific tire service information for your situation."
        except Exception:
            return "I'll look up those tire service details for you."

class EmergencyResponseAgent(Agent):
    """Specialist for emergency situations"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are an emergency response specialist for urgent roadside situations.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}
⚠️ EMERGENCY SITUATION - PRIORITY RESPONSE ⚠️

PRIORITIES:
1. Ensure immediate safety
2. Assess if 911 services needed
3. Expedite roadside assistance
4. Coordinate with emergency services if needed
5. Provide constant communication

EMERGENCY PROTOCOLS:
- Highway/unsafe locations get priority dispatch
- Weather-related emergencies require special equipment
- Medical emergencies - direct to 911 first
- Traffic hazards - coordinate with authorities

Speak with urgency but remain calm and professional.
Get precise location immediately for emergency dispatch."""
        )

    @function_tool()
    async def emergency_dispatch(
        self, 
        context: RunContext[CallData],
        situation: str,
        location: str
    ) -> str:
        """Priority emergency dispatch"""
        
        logger.warning(f"🚨 EMERGENCY DISPATCH: {situation} at {location}")
        
        context.userdata.urgency_level = "emergency"
        context.userdata.call_stage = "emergency_dispatch"
        
        # Immediate emergency response
        await context.session.generate_reply(
            instructions="Acknowledge the emergency with urgency. Tell them you're dispatching emergency roadside assistance immediately to their location. Ask if they need any immediate safety instructions while help is on the way.",
            allow_interruptions=True
        )
        
        return f"EMERGENCY: {situation} - Priority dispatch initiated to {location}"

class InsuranceSpecialistAgent(Agent):
    """Specialist for insurance and coverage questions"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are an insurance and coverage specialist.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}

SPECIALTIES:
- Coverage verification
- Membership benefits
- Policy limitations
- Claims procedures
- Coverage area verification
- Service entitlements

Use search_knowledge for coverage details, membership levels, and policy information.
Always verify membership status and coverage area.
Explain benefits clearly and any limitations."""
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search for insurance/coverage information"""
        enhanced_query = f"insurance coverage membership policy {query}"
        return await self._search_specialist_knowledge(enhanced_query)

    async def _search_specialist_knowledge(self, query: str) -> str:
        """Insurance specialist knowledge search"""
        try:
            results = await qdrant_rag.search(query, limit=3)
            if results:
                coverage_info = []
                for result in results[:2]:
                    if result["score"] >= 0.2:
                        cleaned = result["text"].replace("Q:", "").replace("A:", "").strip()
                        coverage_info.append(cleaned[:100])
                return " | ".join(coverage_info) if coverage_info else "Let me look up your specific coverage details."
            return "I'll verify your coverage and benefits for you."
        except Exception:
            return "Let me check your policy details."

class CustomerServiceAgent(Agent):
    """General customer service for non-specialized inquiries"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are a customer service specialist for general inquiries.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}

RESPONSIBILITIES:
- General questions and information
- Account management
- Service feedback
- Billing inquiries
- Service area questions
- General troubleshooting

Use search_knowledge for any service or policy questions.
Escalate complex technical issues back to appropriate specialists.
Focus on customer satisfaction and clear communication."""
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search for general service information"""
        return await self._search_specialist_knowledge(query)

    async def _search_specialist_knowledge(self, query: str) -> str:
        """General knowledge search"""
        try:
            results = await qdrant_rag.search(query, limit=2)
            if results and results[0]["score"] >= 0.25:
                return results[0]["text"].replace("Q:", "").replace("A:", "").strip()[:120]
            return "Let me get that information for you."
        except Exception:
            return "I'll look that up for you."

async def create_enhanced_session(userdata: CallData) -> AgentSession[CallData]:
    """Create optimized session for natural conversation matching transcript quality"""
    
    session = AgentSession[CallData](
        # Enhanced STT for better accuracy - matching transcript quality
        stt=deepgram.STT(
            model="nova-2-general",
            language="en-US",
            smart_format=True,  # Better punctuation and formatting
            profanity_filter=False,  # Allow natural speech
            numerals=True,  # Convert numbers properly
        ),
        
        # Optimized LLM for conversation flow
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.2,  # More consistent, professional responses
        ),
        
        # Professional TTS voice (like "Mark" in transcripts)
        tts=elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Professional male voice
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.7,      # More stable for professional calls
                similarity_boost=0.8,
                style=0.1,          # Less dramatic, more professional
                speed=0.95          # Slightly slower for clarity
            ),
            model="eleven_turbo_v2_5",  # Fastest model for low latency
        ),
        
        # Enhanced turn detection for natural conversation flow
        vad=silero.VAD.load(),
        # FIXED: Use MultilingualModel instead of EOUModel
        turn_detection=MultilingualModel(),  # Semantic end-of-utterance detection
        
        # Natural conversation timing (based on transcript analysis)
        allow_interruptions=True,
        min_interruption_duration=0.6,  # Allow natural interruptions
        min_endpointing_delay=0.8,      # Natural pause handling
        max_endpointing_delay=4.0,      # Allow time for people to think
        
        # FIXED: userdata goes in constructor, not start() method
        userdata=userdata
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """Multi-agent orchestrator entrypoint"""
    
    logger.info("🚀 Multi-Agent Roadside Assistance System Starting")
    logger.info("🎯 Agents: Dispatcher, Towing, Battery, Tire, Emergency, Insurance, Customer Service")
    
    await ctx.connect()
    
    # Initialize knowledge base
    await qdrant_rag.initialize()
    
    # Create call data and enhanced session
    call_data = CallData()
    session = await create_enhanced_session(call_data)
    
    # Start with dispatcher agent
    dispatcher = DispatcherAgent()
    
    # FIXED: start() only takes agent and room parameters
    await session.start(
        agent=dispatcher,
        room=ctx.room
    )
    
    # Professional greeting
    await session.generate_reply(
        instructions="Give the standard greeting: 'Roadside assistance, this is Mark, how can I help?'"
    )
    
    logger.info("✅ Multi-agent system ready")
    logger.info("📞 Starting with dispatcher for call routing")

if __name__ == "__main__":
    try:
        logger.info("🎙️  Starting Multi-Agent Roadside Assistance System")
        logger.info("🔄 Intelligent call routing to specialized agents")
        logger.info("📊 Dynamic knowledge base integration")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)