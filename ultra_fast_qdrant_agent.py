# ultra_fast_qdrant_agent.py
"""
Ultra-Fast LiveKit RAG Agent with Qdrant Integration
FIXED: RAG timeout and context injection issues
"""
import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
import time

from livekit import agents, api
from livekit.agents import (
    Agent, 
    AgentSession, 
    JobContext,
    RunContext,
    function_tool,
    get_job_context,
    ChatContext,
    ChatMessage,
    WorkerOptions,
    cli
)
from livekit.plugins import openai, deepgram, silero, elevenlabs

from dotenv import load_dotenv
load_dotenv()

from qdrant_rag_system import qdrant_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraFastQdrantAgent(Agent):
    """
    Ultra-fast LiveKit agent with Qdrant RAG integration
    FIXED: Proper timeout handling and context injection
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI voice assistant for phone calls. 

CRITICAL INSTRUCTIONS:
- Keep responses very short (under 30 words) for phone clarity
- When you receive [QdrantRAG] information, use it to answer questions directly and accurately
- ONLY transfer to human when explicitly asked: "transfer me", "human agent", "speak to a person"
- For questions about details, information, or explanations - use search_knowledge
- Always base your answers on the retrieved knowledge when available
- If no relevant information is found, politely say you don't have that specific information

AVAILABLE TOOLS:
- search_knowledge: Use for ALL information requests, questions, and details
- transfer_to_human: ONLY use when explicitly requested transfer

Always search for information first before giving generic responses."""
        )
        self.processing = False
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        FIXED: RAG injection with proper timeout handling
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3 or self.processing:
                return
            
            self.processing = True
            
            try:
                # Skip RAG for explicit transfer requests
                explicit_transfer_phrases = [
                    "transfer me", "human agent", "speak to a person",
                    "talk to a human", "connect me to someone"
                ]
                
                if any(phrase in user_text.lower() for phrase in explicit_transfer_phrases):
                    logger.info(f"🔄 Explicit transfer request detected: {user_text}")
                    return
                
                # FIXED: Use proper timeout that matches actual search performance
                results = await asyncio.wait_for(
                    qdrant_rag.search(user_text, limit=3),
                    timeout=config.rag_timeout_ms / 1000.0  # Convert to seconds
                )
                
                if results and len(results) > 0:
                    # Use the best result with lowered threshold
                    best_result = results[0]
                    if best_result["score"] >= config.similarity_threshold:
                        raw_content = best_result["text"]
                        context = self._clean_content_for_voice(raw_content)
                        turn_ctx.add_message(
                            role="system",
                            content=f"[QdrantRAG]: {context}"
                        )
                        logger.info(f"⚡ RAG context injected (score: {best_result['score']:.3f})")
                    else:
                        logger.info(f"⚠️ Low relevance score: {best_result['score']:.3f} < {config.similarity_threshold}")
                        
            except asyncio.TimeoutError:
                logger.debug("⚡ RAG timeout - continuing without context")
            except Exception as e:
                logger.error(f"❌ RAG error: {e}")
            finally:
                self.processing = False
                
        except Exception as e:
            logger.error(f"❌ on_user_turn_completed error: {e}")
            self.processing = False
    
    def _clean_content_for_voice(self, content: str) -> str:
        """Clean content for voice response"""
        try:
            # Remove common formatting characters
            content = content.replace("Q: ", "").replace("A: ", "")
            content = content.replace("■", "").replace("●", "").replace("•", "")
            content = content.replace("- ", "").replace("* ", "")
            
            # Handle multi-line content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if lines:
                # Take the first substantial line that's not a header
                for line in lines:
                    if len(line) > 15 and not line.startswith(('Q:', 'A:', '#', '-', '*', '■')):
                        content = line
                        break
                else:
                    content = lines[0]
            
            # Limit length for voice
            if len(content) > 200:
                sentences = content.split('.')
                if len(sentences) > 1:
                    content = sentences[0] + "."
                else:
                    content = content[:200] + "..."
            
            return content
            
        except Exception:
            return content[:150] if len(content) > 150 else content

    @function_tool()
    async def search_knowledge(self, query: str) -> str:
        """
        Search the knowledge base for information about any topic.
        IMPROVED: Better handling of search results and scores
        
        Use this for ALL information requests including:
        - Service information and pricing
        - Company details and policies
        - Procedures and guidelines
        - Specific questions about any topic
        - General inquiries
        """
        try:
            logger.info(f"🔍 Searching knowledge base: {query}")
            
            # Search with proper timeout
            results = await asyncio.wait_for(
                qdrant_rag.search(query, limit=config.search_limit),
                timeout=config.rag_timeout_ms / 1000.0
            )
            
            if results and len(results) > 0:
                # Find the best result with reasonable score
                best_result = None
                for result in results:
                    if result["score"] >= config.similarity_threshold:
                        best_result = result
                        break
                
                if not best_result:
                    best_result = results[0]  # Use the best available even if score is low
                
                content = self._clean_content_for_voice(best_result["text"])
                
                # If score is still low, try to combine multiple results
                if best_result["score"] < 0.4 and len(results) > 1:
                    logger.info("🔄 Low score, combining multiple results")
                    combined_content = []
                    for result in results[:2]:  # Take top 2
                        cleaned = self._clean_content_for_voice(result["text"])
                        if cleaned and len(cleaned) > 10:
                            combined_content.append(cleaned)
                    
                    if combined_content:
                        content = " ".join(combined_content)[:200]  # Limit length
                
                logger.info(f"📊 Found result with score: {best_result['score']:.3f}")
                return content
            else:
                logger.warning("⚠️ No relevant information found in knowledge base")
                return "I don't have specific information about that in my knowledge base. Would you like me to transfer you to someone who can help?"
            
        except asyncio.TimeoutError:
            logger.error("❌ Knowledge search timeout")
            return "I'm having trouble accessing the information right now. Let me try to help you directly or transfer you to someone who can assist."
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return "I'm having trouble accessing the information right now. Let me transfer you to someone who can help."

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """
        Transfer the caller to a human agent.
        
        ONLY use this when the caller EXPLICITLY requests:
        - "transfer me"
        - "human agent" 
        - "speak to a person"
        - "talk to a human"
        
        DO NOT use for information requests - use search_knowledge instead.
        """
        try:
            logger.info("🔄 EXECUTING HUMAN TRANSFER - User explicitly requested")
            job_ctx = get_job_context()
            
            # Find SIP participant
            sip_participant = None
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3":
                    sip_participant = participant
                    break
            
            if not sip_participant:
                return "I'm having trouble with the transfer. Please try calling back."
            
            # Quick transfer message
            await ctx.session.generate_reply(
                instructions="Say: 'Connecting you to a human agent now. Please hold on.'"
            )
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=config.transfer_sip_address,
                play_dialtone=True,
            )
            
            await job_ctx.api.sip.transfer_sip_participant(transfer_request)
            return "Transfer to human agent completed successfully"
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "I'm having trouble with the transfer. Let me try to help you directly instead."

async def create_ultra_fast_session() -> AgentSession:
    """Create ultra-fast optimized session with stable ElevenLabs TTS"""
    
    # Configure ElevenLabs TTS with stable, tested settings
    tts_engine = elevenlabs.TTS(
        voice_id="ODq5zmih8GrVes37Dizd",  # Professional voice
        model="eleven_turbo_v2_5",  # Fastest stable model
        language="en",  # Required for turbo model
        
        # Stable voice settings (avoid aggressive optimizations that cause API errors)
        voice_settings=elevenlabs.VoiceSettings(
            stability=0.5,              # Default stable value
            similarity_boost=0.8,       # Default stable value
            style=0.2,                  # Moderate style
            speed=1.0,                  # Normal speed (avoid speed issues)
            use_speaker_boost=True      # Enable for better quality
        ),
        
        # Conservative performance settings to avoid API errors
        inactivity_timeout=300,         # Default timeout (don't make too aggressive)
        enable_ssml_parsing=False,      # Keep disabled for speed
    )
    logger.info("🎙️ Using ElevenLabs TTS with stable configuration")
    
    session = AgentSession(
        # Fast STT
        stt=deepgram.STT(
            model="nova-2-general",
            language="en",
        ),
        
        # Fast LLM
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
        ),
        
        # Stable ElevenLabs TTS
        tts=tts_engine,
        
        # Fast VAD
        vad=silero.VAD.load(),
        
        # Use STT-based turn detection
        turn_detection="stt",
        
        # Optimized timing for telephony
        min_endpointing_delay=0.3,
        max_endpointing_delay=2.0,
        allow_interruptions=True,
        min_interruption_duration=0.3,
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """
    Ultra-fast entrypoint with Qdrant RAG and ElevenLabs TTS
    FIXED: Better initialization and error handling
    """
    logger.info("=== FIXED QDRANT RAG AGENT WITH ELEVENLABS STARTING ===")
    
    # Connect immediately
    await ctx.connect()
    logger.info("✅ Connected")
    
    # Initialize Qdrant RAG and session in parallel
    init_tasks = [
        qdrant_rag.initialize(),
        create_ultra_fast_session()
    ]
    
    rag_success, session = await asyncio.gather(*init_tasks)
    
    # Create agent
    agent = UltraFastQdrantAgent()
    
    # Start session
    await session.start(room=ctx.room, agent=agent)
    
    # Generic greeting that works for any business
    await session.generate_reply(
        instructions="Greet the user professionally and ask how you can help them today."
    )
    logger.info("✅ Initial greeting sent")
    
    logger.info("⚡ FIXED QDRANT RAG AGENT READY!")
    logger.info(f"⚡ Qdrant RAG Status: {'✅ Active' if rag_success else '⚠️ Fallback'}")
    logger.info("🎙️ ElevenLabs TTS Status: ✅ Active")
    
    # Show current configuration
    logger.info(f"📊 RAG Timeout: {config.rag_timeout_ms}ms")
    logger.info(f"📊 Search Limit: {config.search_limit}")
    logger.info(f"📊 Similarity Threshold: {config.similarity_threshold}")

if __name__ == "__main__":
    try:
        logger.info("⚡ Starting FIXED Qdrant RAG Agent with ElevenLabs")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)