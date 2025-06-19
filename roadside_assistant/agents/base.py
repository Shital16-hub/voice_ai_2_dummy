# agents/base_agent.py
"""
Base agent class for LiveKit 1.1 multi-agent roadside assistance system
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from livekit.agents import Agent, ChatContext, ChatMessage
from livekit.plugins import elevenlabs, deepgram, openai, silero

from models.session_data import RoadsideSessionData
from services.rag_service import RAGService
from services.monitoring_service import MonitoringService
from services.transfer_service import TransferService

logger = logging.getLogger(__name__)


class RoadsideBaseAgent(Agent):
    """
    Base agent class with common functionality for roadside assistance
    Provides shared services, monitoring, and agent coordination
    """
    
    def __init__(
        self,
        instructions: str,
        agent_name: str,
        rag_service: Optional[RAGService] = None,
        monitoring_service: Optional[MonitoringService] = None,
        transfer_service: Optional[TransferService] = None,
        chat_ctx: Optional[ChatContext] = None
    ):
        super().__init__(instructions=instructions)
        
        self.agent_name = agent_name
        self.rag_service = rag_service
        self.monitoring_service = monitoring_service
        self.transfer_service = transfer_service
        self.processing = False
        
        # Continue chat context if provided
        if chat_ctx:
            self.chat_ctx = chat_ctx
        
        logger.info(f"🤖 Initialized {agent_name} agent")
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Enhanced turn completion with comprehensive processing"""
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 2 or self.processing:
                return
            
            self.processing = True
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Get session data from context
                session_data = turn_ctx.userdata
                if not isinstance(session_data, RoadsideSessionData):
                    logger.warning("⚠️ Session data not found or invalid type")
                    return
                
                # Add user message to interaction log
                session_data.add_interaction(
                    agent_type=self.agent_name,
                    speaker="customer",
                    message=user_text
                )
                
                # Extract information using monitoring service
                extracted_data = {}
                if self.monitoring_service:
                    extracted_data = await self.monitoring_service.extract_information(
                        user_text, session_data, self.agent_name
                    )
                
                # Add extraction context if data was found
                if extracted_data:
                    context_parts = []
                    for key, value in extracted_data.items():
                        context_parts.append(f"{key}: {value}")
                    
                    if context_parts:
                        extraction_context = f"[EXTRACTED_INFO: {'; '.join(context_parts)}]"
                        turn_ctx.add_message(role="system", content=extraction_context)
                
                # Try RAG search for relevant information
                rag_context_added = False
                if self.rag_service and self.rag_service.ready:
                    try:
                        results = await asyncio.wait_for(
                            self.rag_service.search_knowledge(user_text, limit=3),
                            timeout=1.5  # 1.5 second timeout for telephony
                        )
                        
                        if results and len(results) > 0:
                            best_result = results[0]
                            if best_result["score"] >= 0.25:  # Similarity threshold
                                cleaned_content = self.rag_service.clean_content_for_voice(
                                    best_result["text"]
                                )
                                turn_ctx.add_message(
                                    role="system",
                                    content=f"[KNOWLEDGE_BASE]: {cleaned_content}"
                                )
                                rag_context_added = True
                                
                                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                                logger.info(f"⚡ RAG context added (score: {best_result['score']:.3f}, {response_time:.1f}ms)")
                                
                    except asyncio.TimeoutError:
                        logger.debug("⚡ RAG timeout - continuing without context")
                    except Exception as e:
                        logger.debug(f"⚡ RAG error: {e}")
                
                # Add missing data context for agent awareness
                missing_info = session_data.get_missing_info()
                if missing_info:
                    missing_context = f"[MISSING_DATA]: Still need {', '.join(missing_info[:3])}. Ask naturally for the most important missing item."
                    turn_ctx.add_message(role="system", content=missing_context)
                
                # Add conversation stage context
                completeness = session_data.get_completeness_status()
                if completeness["ready_for_dispatch"]:
                    stage_context = "[STAGE]: All information collected. Ready to confirm service and dispatch."
                else:
                    stage_context = f"[STAGE]: {session_data.current_stage.value}. Continue collecting information naturally."
                
                turn_ctx.add_message(role="system", content=stage_context)
                
                # Update monitoring metrics
                if self.monitoring_service:
                    response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    await self.monitoring_service.update_session_metrics(
                        session_data, self.agent_name, response_time_ms
                    )
                
            finally:
                self.processing = False
                
        except Exception as e:
            logger.error(f"❌ {self.agent_name} turn completion error: {e}")
            self.processing = False
    
    async def before_llm_completion(self, chat_ctx: ChatContext) -> None:
        """Called before LLM generates response"""
        try:
            session_data = chat_ctx.userdata
            if isinstance(session_data, RoadsideSessionData):
                # Update last activity time
                session_data.last_update = datetime.now()
                
        except Exception as e:
            logger.error(f"❌ Before LLM completion error: {e}")
    
    async def after_llm_completion(self, chat_ctx: ChatContext, message: ChatMessage) -> None:
        """Called after LLM generates response"""
        try:
            session_data = chat_ctx.userdata
            if isinstance(session_data, RoadsideSessionData):
                # Add agent response to interaction log
                agent_response = message.text_content
                if agent_response:
                    session_data.add_interaction(
                        agent_type=self.agent_name,
                        speaker="agent",
                        message=agent_response
                    )
                
        except Exception as e:
            logger.error(f"❌ After LLM completion error: {e}")
    
    def get_voice_config(self) -> dict:
        """Get voice configuration for this agent"""
        return {
            "voice_id": "TxGEqnHWrfWFTfGW9XjX",  # Josh - professional male voice
            "model": "eleven_turbo_v2_5",
            "voice_settings": elevenlabs.VoiceSettings(
                stability=0.6,
                similarity_boost=0.8,
                style=0.3,
                speed=1.0,
                use_speaker_boost=True
            )
        }
    
    def get_stt_config(self) -> dict:
        """Get STT configuration optimized for telephony"""
        return {
            "model": "nova-2-general",
            "language": "en"
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration for fast responses"""
        return {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    
    def get_vad_config(self) -> dict:
        """Get VAD configuration"""
        return {
            "model": silero.VAD.load()
        }
    
    @staticmethod
    def create_session_config(agent_instance: 'RoadsideBaseAgent'):
        """Create optimized session configuration for telephony"""
        from livekit.agents import AgentSession
        
        return AgentSession(
            # STT optimized for telephony
            stt=deepgram.STT(**agent_instance.get_stt_config()),
            
            # Fast LLM for quick responses
            llm=openai.LLM(**agent_instance.get_llm_config()),
            
            # Professional male voice
            tts=elevenlabs.TTS(**agent_instance.get_voice_config()),
            
            # Fast VAD
            vad=agent_instance.get_vad_config()["model"],
            
            # Optimized timing for telephony
            turn_detection="stt",
            min_endpointing_delay=0.3,
            max_endpointing_delay=2.0,
            allow_interruptions=True,
            min_interruption_duration=0.3,
        )
    
    async def transfer_to_agent(self, new_agent: 'RoadsideBaseAgent', message: str = "") -> tuple:
        """
        Transfer control to another agent
        Returns (new_agent_instance, transfer_message)
        """
        try:
            logger.info(f"🔄 Transferring from {self.agent_name} to {new_agent.agent_name}")
            
            # Pass services to new agent
            new_agent.rag_service = self.rag_service
            new_agent.monitoring_service = self.monitoring_service
            new_agent.transfer_service = self.transfer_service
            
            transfer_message = message or f"Let me connect you with our {new_agent.agent_name} specialist."
            
            return new_agent, transfer_message
            
        except Exception as e:
            logger.error(f"❌ Agent transfer error: {e}")
            return self, "I'll continue helping you with your request."
    
    async def get_session_summary(self, session_data: RoadsideSessionData) -> dict:
        """Get comprehensive session summary"""
        try:
            completeness = session_data.get_completeness_status()
            missing_info = session_data.get_missing_info()
            
            return {
                "agent": self.agent_name,
                "stage": session_data.current_stage.value,
                "completeness": completeness,
                "missing_info": missing_info,
                "ready_for_dispatch": completeness["ready_for_dispatch"],
                "interaction_count": len(session_data.interactions),
                "duration_minutes": (
                    datetime.now() - session_data.conversation_start
                ).total_seconds() / 60
            }
            
        except Exception as e:
            logger.error(f"❌ Session summary error: {e}")
            return {"error": str(e)}
    
    def __str__(self) -> str:
        return f"{self.agent_name}Agent"
    
    def __repr__(self) -> str:
        return f"{self.agent_name}Agent(services={bool(self.rag_service)},{bool(self.monitoring_service)},{bool(self.transfer_service)})"