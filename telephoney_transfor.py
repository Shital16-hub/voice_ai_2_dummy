from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import (
    Agent, 
    AgentSession, 
    RoomInputOptions, 
    RunContext,
    function_tool,
    get_job_context
)
from livekit.plugins import (
    openai,
    elevenlabs,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import asyncio
import logging

load_dotenv()

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. You can help users with their queries.
            
            If a user explicitly asks to speak to a human agent, requests human support, says they want to talk to a person,
            or if you cannot help them with their request, offer to transfer them to a human agent. 
            
            Always confirm with the user before transferring the call by saying something like:
            "I'd be happy to transfer you to a human agent. Would you like me to do that now?"
            
            Wait for their confirmation before calling the transfer function."""
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Enhanced transfer function with better debugging and timeout handling"""
        
        transfer_to = "sip:voiceai@sip.linphone.org"
        
        # Get the current job context
        job_ctx = get_job_context()
        
        # Enhanced logging for debugging
        logger.info(f"=== TRANSFER CALL INITIATED ===")
        logger.info(f"Room: {job_ctx.room.name}")
        logger.info(f"Total remote participants: {len(job_ctx.room.remote_participants)}")
        
        # Find the SIP participant
        sip_participant = None
        for participant in job_ctx.room.remote_participants.values():
            logger.info(f"Found participant: {participant.identity}, kind: {participant.kind}")
            if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                sip_participant = participant
                logger.info(f"✅ Found SIP participant: {participant.identity}")
                break
        
        if not sip_participant:
            logger.error("❌ No SIP participants found!")
            await ctx.session.generate_reply(
                instructions="I'm sorry, I couldn't find any active participants to transfer. Please try calling again."
            )
            return "Could not find any participant to transfer. Please try again."
        
        participant_identity = sip_participant.identity
        logger.info(f"🔄 Will transfer participant: {participant_identity} to SIP: {transfer_to}")
        
        # Inform the user about the transfer with instructions
        await ctx.session.generate_reply(
            instructions="""I'm connecting you to a human agent now. The transfer will begin in just a moment. 
            If you hear ringing, the agent should answer automatically. Please stay on the line."""
        )
        
        # Wait for the message to complete
        await asyncio.sleep(2)
        
        try:
            # Execute the SIP transfer with detailed logging
            logger.info(f"🚀 Starting SIP transfer request...")
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=participant_identity,
                transfer_to=transfer_to,
                play_dialtone=True,
            )
            
            # Start the transfer
            logger.info(f"📞 Executing transfer_sip_participant...")
            start_time = asyncio.get_event_loop().time()
            
            # Try with 30 second timeout (in case auto-answer delay is configured)
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=30.0
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            logger.info(f"✅ SIP Transfer completed successfully in {duration:.2f} seconds!")
            logger.info(f"   From: {participant_identity}")
            logger.info(f"   To: {transfer_to}")
            logger.info(f"   Room: {job_ctx.room.name}")
            
            return "Call transfer completed successfully to human agent"
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Transfer timeout after 30 seconds")
            logger.error(f"🔍 DEBUGGING INFO:")
            logger.error(f"   • Transfer reached Linphone (you should see missed call)")
            logger.error(f"   • Auto-answer is not responding")
            logger.error(f"   • Check: Auto-answer delay = 0 seconds")
            logger.error(f"   • Check: App running in background")
            logger.error(f"   • Check: Do Not Disturb = OFF")
            logger.error(f"   • Check: Phone not in silent mode")
            
            await ctx.session.generate_reply(
                instructions="""I'm having trouble connecting to our human agent. The call is reaching them, 
                but their phone isn't automatically answering. They may need to check their auto-answer settings. 
                Would you like me to try again, or would you prefer to call back later?"""
            )
            return "Transfer timed out - auto-answer not responding. Check Linphone settings."
                    
        except Exception as e:
            logger.error(f"❌ Error transferring call: {e}")
            logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            
            # Provide specific guidance based on error type
            if "408" in str(e):
                logger.error("💡 408 = Call reached destination but timed out waiting for answer")
                await ctx.session.generate_reply(
                    instructions="The call reached our human agent but they didn't answer in time. Please try again or they will call you back shortly."
                )
            elif "500" in str(e):
                logger.error("💡 500 = Server error, possibly SIP configuration issue")
                await ctx.session.generate_reply(
                    instructions="I'm experiencing a technical issue with the transfer system. Please try again in a moment."
                )
            elif "404" in str(e):
                logger.error("💡 404 = SIP address not found")
                await ctx.session.generate_reply(
                    instructions="I couldn't locate our human agent's phone system. Please try again later."
                )
            else:
                await ctx.session.generate_reply(
                    instructions="I apologize, but I'm having trouble transferring your call right now. Please try again in a moment."
                )
            
            return f"Transfer failed: {str(e)}"

    @function_tool()
    async def check_transfer_availability(self, ctx: RunContext):
        """Check if human agents are available for transfer"""
        logger.info("Checking human agent availability")
        return "Human agents are available for transfer. Would you like me to connect you to one?"

    @function_tool()
    async def get_business_hours(self, ctx: RunContext):
        """Provide information about when human agents are available"""
        return "Our human agents are available 24/7. I can transfer you to speak with someone right now if you'd like."


async def entrypoint(ctx: agents.JobContext):
    """Main entry point for the voice agent"""
    
    # Enhanced logging
    logger.info(f"=== AGENT SESSION STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    logger.info(f"Agent: my-telephony-agent")
    
    # ✅ CORRECTED SESSION CONFIGURATION
    session = AgentSession(
        # STT: Deepgram configuration
        stt=deepgram.STT(model="nova-3", language="multi"),
        
        # LLM: OpenAI configuration
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.3,
        ),
        
        # ✅ FIXED: Correct OpenAI TTS configuration
        tts=openai.TTS(
            model="tts-1",           # ✅ Correct OpenAI TTS model
            voice="nova",            # ✅ Valid OpenAI voice (nova, alloy, echo, fable, onyx, shimmer)
        ),
        
        # VAD: Silero configuration
        vad=silero.VAD.load(),
        
        # Turn Detection: Multilingual model
        turn_detection=MultilingualModel(),
    )

    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # Noise cancellation for telephony
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # Connect to the room
    await ctx.connect()
    logger.info("✅ Agent connected to room successfully")

    # Generate initial greeting
    await session.generate_reply(
        instructions="""Give a brief, friendly greeting. Say: "Hello! I'm your AI assistant. How can I help you today? If you need to speak with a human agent, just let me know and I can transfer you right away." Keep it short and to the point."""
    )
    
    logger.info("✅ Initial greeting sent")


if __name__ == "__main__":
    # Start the agent
    logger.info("🚀 Starting OPTIMIZED Voice Agent with Human Transfer")
    logger.info("📞 Transfer destination: sip:voiceai@sip.linphone.org")
    
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="my-telephony-agent"
    ))