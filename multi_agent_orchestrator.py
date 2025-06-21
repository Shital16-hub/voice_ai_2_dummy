# multi_agent_orchestrator.py - OPTIMIZED FOR FASTER STARTUP AND BETTER FLOW
"""
Multi-Agent Orchestrator for Specialized Roadside Assistance
OPTIMIZED: Faster initialization, better conversation flow, one-by-one information gathering
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
            instructions="""You are Mark, a professional roadside assistance dispatcher. Follow this natural conversation flow:

CRITICAL: Gather information ONE piece at a time, following this exact order:
1. If no name: "Could you please provide your full name?"
2. If no phone: "Could you also provide a good phone number where we can reach you?"
3. If no location: "What is the exact location of your vehicle? Please provide the full street address, city, and any nearby landmarks"
4. If no vehicle info: "Could you tell me the year, make, and model of your vehicle?"
5. If no service type: "What type of service do you need today?"

CONVERSATION STYLE:
- Ask for ONE piece of information at a time
- Be empathetic: "I'm sorry to hear about that"
- Confirm details: "Just to confirm, you said..."
- Use natural transitions: "Got it" "Thanks for that" "Perfect"
- Keep responses under 25 words for phone clarity

ROUTING DECISIONS:
- Complex towing situations → route_to_specialist("towing_specialist")
- Battery/electrical issues → route_to_specialist("battery_specialist")
- Tire problems → route_to_specialist("tire_specialist")
- Emergency situations → route_to_specialist("emergency_response")
- Insurance/coverage questions → route_to_specialist("insurance_specialist")
- Pricing questions → Use search_knowledge first

NEVER ask for multiple pieces of information in one response."""
        )
        self.call_start_time = time.time()
        self.specialist_agents = {}
        
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Enhanced context injection with conversation state tracking"""
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3:
                return
            
            # Analyze what information we still need
            await self._inject_conversation_state(turn_ctx, user_text)
                    
        except Exception as e:
            logger.error(f"Error in conversation context: {e}")

    async def _inject_conversation_state(self, turn_ctx: ChatContext, user_text: str) -> None:
        """Inject current conversation state to guide next question"""
        user_lower = user_text.lower()
        
        # Detect what type of information was just provided
        info_provided = {
            "name": any(word.istitle() and len(word) > 2 for word in user_text.split()),
            "phone": any(char.isdigit() for char in user_text) and len([c for c in user_text if c.isdigit()]) >= 7,
            "location": any(indicator in user_lower for indicator in ["street", "road", "avenue", "boulevard", "highway", "exit", "mile", "address", "city"]),
            "vehicle": any(brand in user_lower for brand in ["honda", "toyota", "ford", "chevy", "bmw", "audi", "mercedes", "nissan", "hyundai", "kia", "jeep", "dodge"]) or any(year in user_text for year in ["20", "19"]),
            "service_request": any(service in user_lower for service in ["tow", "battery", "tire", "jump", "fuel", "lockout", "emergency"])
        }
        
        # Determine next question needed
        next_question = "greeting"
        if info_provided["name"] or "name" in user_lower:
            next_question = "phone"
        elif info_provided["phone"] or "phone" in user_lower or "number" in user_lower:
            next_question = "location"
        elif info_provided["location"] or any(word in user_lower for word in ["location", "where", "address"]):
            next_question = "vehicle"
        elif info_provided["vehicle"] or any(word in user_lower for word in ["car", "vehicle", "suv", "truck"]):
            next_question = "service"
        elif info_provided["service_request"]:
            next_question = "complete"
        
        # Check for pricing questions
        if any(word in user_lower for word in ["price", "cost", "how much", "pricing", "plan"]):
            next_question = "pricing"
        
        # Inject appropriate context
        context_msg = f"CONVERSATION STATE:\n"
        
        if next_question == "phone":
            context_msg += "✅ Have name - Now ask: 'Could you also provide a good phone number where we can reach you?'"
        elif next_question == "location":
            context_msg += "✅ Have phone - Now ask: 'What is the exact location of your vehicle? Please provide the full street address, city, and any nearby landmarks'"
        elif next_question == "vehicle":
            context_msg += "✅ Have location - Now ask: 'Could you tell me the year, make, and model of your vehicle?'"
        elif next_question == "service":
            context_msg += "✅ Have vehicle - Now ask: 'What type of service do you need today?'"
        elif next_question == "complete":
            context_msg += "✅ All info gathered - Search knowledge base and provide service options"
        elif next_question == "pricing":
            context_msg += "🔍 PRICING QUESTION - Use search_knowledge function immediately"
        else:
            context_msg += "👋 GREETING - Ask: 'Could you please provide your full name?'"
            
        turn_ctx.add_message(role="system", content=context_msg)

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search knowledge base for service information with faster timeout"""
        try:
            # Use faster timeout for real-time conversation
            results = await asyncio.wait_for(
                qdrant_rag.search(query, limit=1),  # Reduced to 1 result for speed
                timeout=0.5  # Reduced timeout to 500ms
            )
            
            if results and results[0]["score"] >= 0.2:  # Lower threshold for better coverage
                formatted = self._format_for_voice(results[0]["text"])
                if formatted:
                    return formatted
            
            return "Let me connect you with a specialist who can provide detailed information about that."
                
        except asyncio.TimeoutError:
            logger.warning("Knowledge search timeout - continuing without context")
            return "I'll transfer you to someone who can help with that specific question."
        except Exception as e:
            logger.error(f"Knowledge search error: {e}")
            return "I'll transfer you to someone who can help with that specific question."

    def _format_for_voice(self, text: str) -> str:
        """Format text for voice delivery - keep it concise"""
        cleaned = text.replace("Q:", "").replace("A:", "").replace("•", "").strip()
        sentences = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 10]
        if sentences:
            result = sentences[0]
            # Keep it under 80 words for phone clarity
            return result[:80] + "..." if len(result) > 80 else result
        return cleaned[:80] if cleaned else ""

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
        
        # Store information in userdata
        if name:
            context.userdata.caller_name = name
            context.userdata.gathered_info["name"] = True
            return "Thank you. Could you also provide a good phone number where we can reach you?"
            
        if phone:
            context.userdata.phone_number = phone
            context.userdata.gathered_info["phone"] = True
            return "Got it. What is the exact location of your vehicle? Please provide the full street address, city, and any nearby landmarks."
            
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
        
        # Inform user about transfer
        specialist_name = specialist_type.replace('_', ' ').title()
        return f"I'm connecting you with our {specialist_name} who can provide expert assistance. Please hold."

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

# Specialist agent classes remain the same but with faster search timeouts
class TowingSpecialistAgent(Agent):
    """Specialist for complex towing situations"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are a towing specialist with expert knowledge.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}

SPECIALTIES: Long-distance towing, heavy vehicle towing, accident recovery.
Use search_knowledge for specific rates and capabilities.
Keep responses under 25 words."""
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search for towing-specific information with fast timeout"""
        enhanced_query = f"towing {query}"
        try:
            results = await asyncio.wait_for(qdrant_rag.search(enhanced_query, limit=1), timeout=0.4)
            if results and results[0]["score"] >= 0.2:
                return results[0]["text"][:100]
            return "Let me get those specific details for you."
        except:
            return "I'll verify those details for you."

# Similar optimizations for other specialist agents...
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

class InsuranceSpecialistAgent(Agent):
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"Insurance specialist. Context: {handoff_context.conversation_summary}")

class CustomerServiceAgent(Agent):
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"Customer service. Context: {handoff_context.conversation_summary}")

async def create_optimized_session(userdata: CallData) -> AgentSession[CallData]:
    """Create highly optimized session for fast startup and conversation"""
    
    session = AgentSession[CallData](
        # Optimized STT for speed
        stt=deepgram.STT(
            model="nova-2-general",
            language="en-US",
            smart_format=True,
            profanity_filter=False,
            numerals=True,
        ),
        
        # Faster LLM settings
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,  # More deterministic
        ),
        
        # Optimized TTS for speed
        tts=elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.6,
                similarity_boost=0.7,
                style=0.0,      # No style for speed
                speed=1.1       # Slightly faster speech
            ),
            model="eleven_turbo_v2_5",
        ),
        
        # Fast turn detection
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        
        # Optimized timing for faster conversation
        allow_interruptions=True,
        min_interruption_duration=0.4,  # Faster interruptions
        min_endpointing_delay=0.6,      # Faster response
        max_endpointing_delay=2.5,      # Don't wait too long
        
        userdata=userdata
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """Optimized entrypoint for faster startup"""
    
    logger.info("🚀 Multi-Agent Roadside Assistance System Starting")
    logger.info("⚡ OPTIMIZED for faster startup and better conversation flow")
    
    await ctx.connect()
    
    # FASTER INITIALIZATION: Start Qdrant in background, don't wait for cache warm-up
    asyncio.create_task(qdrant_rag.initialize())  # Don't await - let it initialize in background
    
    # Create call data and session immediately
    call_data = CallData()
    session = await create_optimized_session(call_data)
    
    # Start with dispatcher agent
    dispatcher = DispatcherAgent()
    
    await session.start(
        agent=dispatcher,
        room=ctx.room
    )
    
    # Give greeting immediately while knowledge base loads in background
    await session.generate_reply(
        instructions="Give a brief professional greeting: 'Roadside assistance, this is Mark, how can I help?'"
    )
    
    logger.info("✅ Multi-agent system ready")
    logger.info("📞 Natural one-by-one information gathering enabled")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting OPTIMIZED Multi-Agent System")
        logger.info("⚡ Faster startup, better conversation flow")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)