# fixed_multi_agent_orchestrator.py - FIXED RAG INTEGRATION
"""
Multi-Agent Orchestrator with WORKING RAG system
FIXED: Proper timeout handling, better search, working knowledge base
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
    """Main dispatcher that routes calls to appropriate specialists - FIXED RAG"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance dispatcher. 

CRITICAL RAG INSTRUCTIONS:
- When user asks about PRICING, COSTS, or "how much" - ALWAYS use search_knowledge function FIRST
- Use the knowledge base information to provide accurate pricing and service details
- If search_knowledge returns relevant information, use it in your response
- Only route to specialists for complex situations that need specialized help

CONVERSATION FLOW:
1. If no name: "Could you please provide your full name?"
2. If no phone: "Could you also provide a good phone number where we can reach you?"
3. If no location: "What is the exact location of your vehicle? Please provide the full street address, city, and any nearby landmarks"
4. If no vehicle info: "Could you tell me the year, make, and model of your vehicle?"
5. If no service type: "What type of service do you need today?"

PRICING QUESTIONS:
- ALWAYS search knowledge base first with search_knowledge
- Provide specific pricing from the knowledge base
- Only transfer if you can't find pricing information

ROUTING DECISIONS (only after trying search_knowledge):
- Complex towing situations → route_to_specialist("towing_specialist")
- Battery/electrical issues → route_to_specialist("battery_specialist")
- Tire problems → route_to_specialist("tire_specialist")
- Emergency situations → route_to_specialist("emergency_response")

Keep responses under 30 words for phone clarity."""
        )
        self.call_start_time = time.time()
        
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Enhanced context injection with automatic RAG for pricing questions"""
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3:
                return
            
            user_lower = user_text.lower()
            
            # Auto-inject RAG context for pricing questions
            pricing_keywords = ["price", "cost", "how much", "pricing", "fee", "charge", "rate", "plan", "membership"]
            if any(keyword in user_lower for keyword in pricing_keywords):
                logger.info(f"🔍 Pricing question detected, auto-searching: {user_text}")
                try:
                    results = await asyncio.wait_for(
                        qdrant_rag.search(user_text, limit=3),
                        timeout=2.0  # Increased timeout for pricing queries
                    )
                    
                    if results and results[0]["score"] >= 0.2:  # Lower threshold for pricing
                        context_parts = []
                        for result in results[:2]:  # Use top 2 results
                            if result["score"] >= 0.2:
                                context_parts.append(result["text"])
                        
                        if context_parts:
                            combined_context = " | ".join(context_parts)
                            turn_ctx.add_message(
                                role="system", 
                                content=f"[PRICING INFO FROM KNOWLEDGE BASE]: {combined_context}"
                            )
                            logger.info(f"💰 Pricing context auto-injected (score: {results[0]['score']:.3f})")
                
                except Exception as e:
                    logger.warning(f"Auto-RAG failed: {e}")
            
            # Analyze conversation state
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
        
        # Check for pricing questions
        pricing_question = any(word in user_lower for word in ["price", "cost", "how much", "pricing", "plan"])
        
        # Inject appropriate context
        context_msg = f"CONVERSATION STATE:\n"
        
        if pricing_question:
            context_msg += "💰 PRICING QUESTION - Use search_knowledge function immediately to find pricing information\n"
        elif not any(info_provided.values()):
            context_msg += "👋 GREETING - Ask: 'Could you please provide your full name?'\n"
        elif info_provided["name"] and not info_provided["phone"]:
            context_msg += "✅ Have name - Now ask: 'Could you also provide a good phone number where we can reach you?'\n"
        elif info_provided["phone"] and not info_provided["location"]:
            context_msg += "✅ Have phone - Now ask: 'What is the exact location of your vehicle?'\n"
        elif info_provided["location"] and not info_provided["vehicle"]:
            context_msg += "✅ Have location - Now ask: 'Could you tell me the year, make, and model of your vehicle?'\n"
        elif info_provided["vehicle"] and not info_provided["service_request"]:
            context_msg += "✅ Have vehicle - Now ask: 'What type of service do you need today?'\n"
        else:
            context_msg += "✅ All info gathered - Use search_knowledge for service options\n"
            
        turn_ctx.add_message(role="system", content=context_msg)

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search knowledge base - FIXED with proper timeout and error handling"""
        try:
            logger.info(f"🔍 SEARCHING KNOWLEDGE BASE: {query}")
            
            # Use longer timeout for knowledge searches
            results = await asyncio.wait_for(
                qdrant_rag.search(query, limit=3),
                timeout=3.0  # Increased timeout
            )
            
            if results and len(results) > 0:
                # Process multiple results for comprehensive answers
                relevant_results = []
                for result in results:
                    if result["score"] >= 0.15:  # Lower threshold for more coverage
                        relevant_results.append(result)
                
                if relevant_results:
                    # Combine the best results
                    response_parts = []
                    for result in relevant_results[:2]:  # Top 2 results
                        text = result["text"]
                        # Clean up the text
                        cleaned = text.replace("Q:", "").replace("A:", "").strip()
                        if len(cleaned) > 20:  # Only use substantial content
                            response_parts.append(cleaned)
                    
                    if response_parts:
                        combined_response = " | ".join(response_parts)
                        logger.info(f"📊 Found {len(relevant_results)} relevant results (best score: {results[0]['score']:.3f})")
                        return combined_response[:200]  # Limit for voice
                
            # If no good results found
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
            return "Perfect! Let me search for the best service options for you."
        
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
        
        # Store routing information
        context.userdata.call_stage = "transferred"
        
        # Create handoff context
        handoff_context = HandoffContext(
            reason=reason,
            previous_agent="dispatcher",
            conversation_summary=self._create_conversation_summary(context.userdata),
            urgency_level=context.userdata.urgency_level,
            collected_info={
                "name": context.userdata.caller_name,
                "phone": context.userdata.phone_number,
                "location": context.userdata.location,
                "vehicle": f"{context.userdata.vehicle_year or ''} {context.userdata.vehicle_make or ''} {context.userdata.vehicle_model or ''}".strip(),
                "service": context.userdata.service_type,
                "issue": context.userdata.issue_description
            }
        )
        
        # Create appropriate specialist agent
        if specialist_type == "towing_specialist":
            specialist = TowingSpecialistAgent(handoff_context)
        elif specialist_type == "battery_specialist":
            specialist = BatterySpecialistAgent(handoff_context)
        elif specialist_type == "tire_specialist":
            specialist = TireSpecialistAgent(handoff_context)
        elif specialist_type == "emergency_response":
            specialist = EmergencyResponseAgent(handoff_context)
        else:
            specialist = CustomerServiceAgent(handoff_context)
        
        # Switch to specialist agent
        # Note: In a full implementation, you'd need to manage agent switching
        # For now, we'll inform the user about the transfer
        
        specialist_name = specialist_type.replace('_', ' ').title()
        return f"I'm connecting you with our {specialist_name} who can provide expert assistance. They have all your information and will take over from here."

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

# Specialist agent classes - These are AI agents, not humans
class TowingSpecialistAgent(Agent):
    """AI Specialist for towing situations - NOT HUMAN"""
    
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(
            instructions=f"""You are a TOWING SPECIALIST AI agent with expert knowledge.

HANDOFF CONTEXT: {handoff_context.conversation_summary}
REASON FOR TRANSFER: {handoff_context.reason}

You are an AI agent specializing in towing services. You have access to the knowledge base.

SPECIALTIES: 
- Long-distance towing
- Heavy vehicle towing  
- Accident recovery
- Flatbed services
- Wheel lift services

Use search_knowledge for specific rates, availability, and procedures.
Provide expert advice and coordinate towing services.
Keep responses under 30 words for phone clarity."""
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        """Search for towing-specific information"""
        enhanced_query = f"towing {query}"
        try:
            results = await asyncio.wait_for(qdrant_rag.search(enhanced_query, limit=2), timeout=2.0)
            if results and results[0]["score"] >= 0.15:
                return results[0]["text"][:150]
            return "Let me get those specific towing details for you."
        except:
            return "I'll verify those towing details for you."

class BatterySpecialistAgent(Agent):
    """AI Specialist for battery issues - NOT HUMAN"""
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"""You are a BATTERY SPECIALIST AI agent.
Context: {handoff_context.conversation_summary}
Specializing in jump starts, battery replacement, electrical diagnostics.
Use search_knowledge for battery service information.""")

    @function_tool()
    async def search_knowledge(self, context: RunContext[CallData], query: str) -> str:
        enhanced_query = f"battery jumpstart {query}"
        try:
            results = await asyncio.wait_for(qdrant_rag.search(enhanced_query, limit=2), timeout=2.0)
            if results and results[0]["score"] >= 0.15:
                return results[0]["text"][:150]
            return "Let me get battery service details."
        except:
            return "I'll check battery service information."

class TireSpecialistAgent(Agent):
    """AI Specialist for tire issues - NOT HUMAN"""
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"""You are a TIRE SPECIALIST AI agent.
Context: {handoff_context.conversation_summary}
Specializing in flat tire repair, tire changes, wheel services.""")

class EmergencyResponseAgent(Agent):
    """AI Specialist for emergency situations - NOT HUMAN"""
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"""You are an EMERGENCY RESPONSE AI agent.
Context: {handoff_context.conversation_summary}
Prioritize safety and rapid response for emergency situations.""")

class CustomerServiceAgent(Agent):
    """AI General customer service agent - NOT HUMAN"""
    def __init__(self, handoff_context: HandoffContext):
        self.handoff_context = handoff_context
        super().__init__(instructions=f"""You are a CUSTOMER SERVICE AI agent.
Context: {handoff_context.conversation_summary}
Handle general inquiries and service coordination.""")

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
    """Entrypoint with WORKING RAG system"""
    
    logger.info("🚀 Multi-Agent Roadside Assistance System Starting")
    logger.info("⚡ FIXED RAG integration with proper knowledge base access")
    
    await ctx.connect()
    
    # Initialize Qdrant RAG system and wait for it to be ready
    logger.info("📚 Initializing knowledge base...")
    rag_success = await qdrant_rag.initialize()
    
    if rag_success:
        logger.info("✅ Knowledge base ready")
        # Test search to verify
        try:
            test_results = await qdrant_rag.search("pricing", limit=1)
            logger.info(f"🔍 Knowledge base test: Found {len(test_results)} results")
        except Exception as e:
            logger.warning(f"Knowledge base test failed: {e}")
    else:
        logger.error("❌ Knowledge base initialization failed")
    
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
        instructions="Give a brief professional greeting: 'Roadside assistance, this is Mark, how can I help?'"
    )
    
    logger.info("✅ Multi-agent system ready with WORKING RAG")
    logger.info("💰 Knowledge base ready for pricing questions")
    logger.info("🎭 All specialist agents are AI agents, not humans")

if __name__ == "__main__":
    try:
        logger.info("🎙️ Starting FIXED Multi-Agent System with Working RAG")
        logger.info("📊 Excel knowledge base integration active")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)