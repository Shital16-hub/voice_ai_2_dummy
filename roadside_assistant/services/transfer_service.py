# services/transfer_service.py
"""
Human transfer service for LiveKit 1.1 multi-agent system
"""
import asyncio
import logging
from typing import Optional
from livekit import api
from livekit.agents import get_job_context

from models.session_data import RoadsideSessionData

logger = logging.getLogger(__name__)


class TransferService:
    """
    Service for handling transfers to human agents
    Integrates with SIP/telephony systems
    """
    
    def __init__(
        self,
        default_transfer_address: str = "sip:voiceai@sip.linphone.org",
        transfer_timeout: int = 30,
        enable_dialtone: bool = True
    ):
        self.default_transfer_address = default_transfer_address
        self.transfer_timeout = transfer_timeout
        self.enable_dialtone = enable_dialtone
        
        # Track transfer attempts
        self.transfer_history: dict = {}
    
    async def transfer_to_human(
        self, 
        session_data: RoadsideSessionData,
        reason: Optional[str] = None,
        custom_address: Optional[str] = None
    ) -> dict:
        """
        Transfer call to human agent
        Returns transfer result with status and details
        """
        transfer_address = custom_address or self.default_transfer_address
        session_id = session_data.room_name or "unknown"
        
        logger.info(f"🔄 Initiating human transfer for session {session_id}")
        logger.info(f"   Reason: {reason or 'User requested'}")
        logger.info(f"   Target: {transfer_address}")
        
        try:
            # Update session data
            session_data.transfer_requested = True
            session_data.transfer_reason = reason
            session_data.add_interaction(
                agent_type="transfer_service",
                speaker="system",
                message=f"Transfer initiated: {reason or 'User requested'}"
            )
            
            # Get LiveKit job context
            job_ctx = get_job_context()
            
            # Find SIP participant
            sip_participant = await self._find_sip_participant(job_ctx)
            if not sip_participant:
                error_msg = "No SIP participants found for transfer"
                logger.error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_code": "NO_SIP_PARTICIPANT"
                }
            
            logger.info(f"✅ Found SIP participant: {sip_participant.identity}")
            
            # Execute transfer
            transfer_result = await self._execute_transfer(
                job_ctx, 
                sip_participant,
                transfer_address,
                session_id
            )
            
            # Log result
            if transfer_result["success"]:
                logger.info(f"✅ Transfer completed successfully")
                session_data.add_interaction(
                    agent_type="transfer_service",
                    speaker="system",
                    message="Transfer completed successfully"
                )
            else:
                logger.error(f"❌ Transfer failed: {transfer_result['error']}")
                session_data.add_interaction(
                    agent_type="transfer_service",
                    speaker="system",
                    message=f"Transfer failed: {transfer_result['error']}"
                )
            
            # Store in history
            self.transfer_history[session_id] = {
                "timestamp": session_data.last_update,
                "reason": reason,
                "target": transfer_address,
                "result": transfer_result
            }
            
            return transfer_result
            
        except Exception as e:
            error_msg = f"Transfer system error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "error_code": "SYSTEM_ERROR"
            }
    
    async def _find_sip_participant(self, job_ctx):
        """Find SIP participant in the room"""
        try:
            for participant in job_ctx.room.remote_participants.values():
                logger.info(f"📞 Checking participant: {participant.identity}, kind: {participant.kind}")
                
                # Check for SIP participant (kind 3 is SIP in LiveKit)
                if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                    return participant
            
            logger.warning("⚠️ No SIP participants found in room")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding SIP participant: {e}")
            return None
    
    async def _execute_transfer(
        self, 
        job_ctx, 
        sip_participant, 
        transfer_address: str,
        session_id: str
    ) -> dict:
        """Execute the actual SIP transfer"""
        try:
            # Create transfer request
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=transfer_address,
                play_dialtone=self.enable_dialtone,
            )
            
            logger.info(f"🚀 Executing SIP transfer...")
            logger.info(f"   Room: {job_ctx.room.name}")
            logger.info(f"   Participant: {sip_participant.identity}")
            logger.info(f"   Target: {transfer_address}")
            
            # Execute with timeout
            start_time = asyncio.get_event_loop().time()
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=self.transfer_timeout
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            logger.info(f"✅ SIP transfer completed in {duration:.2f} seconds")
            
            return {
                "success": True,
                "duration_seconds": duration,
                "participant_id": sip_participant.identity,
                "target_address": transfer_address
            }
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Transfer timeout after {self.transfer_timeout} seconds")
            return {
                "success": False,
                "error": f"Transfer timed out after {self.transfer_timeout} seconds",
                "error_code": "TIMEOUT",
                "troubleshooting": [
                    "Check if target SIP address is reachable",
                    "Verify auto-answer settings on destination",
                    "Ensure destination device is not in Do Not Disturb mode",
                    "Check network connectivity to SIP provider"
                ]
            }
            
        except Exception as e:
            error_details = str(e)
            logger.error(f"❌ SIP transfer error: {error_details}")
            
            # Provide specific error guidance
            error_code = "UNKNOWN"
            troubleshooting = []
            
            if "408" in error_details:
                error_code = "REQUEST_TIMEOUT"
                troubleshooting = [
                    "Call reached destination but timed out waiting for answer",
                    "Check auto-answer delay settings",
                    "Verify destination is available"
                ]
            elif "500" in error_details:
                error_code = "SERVER_ERROR"
                troubleshooting = [
                    "SIP server configuration issue",
                    "Check SIP provider status",
                    "Verify account credentials"
                ]
            elif "404" in error_details:
                error_code = "NOT_FOUND"
                troubleshooting = [
                    "SIP address not found",
                    "Check destination address format",
                    "Verify SIP routing configuration"
                ]
            
            return {
                "success": False,
                "error": error_details,
                "error_code": error_code,
                "troubleshooting": troubleshooting
            }
    
    async def check_transfer_capability(self) -> dict:
        """Check if transfer capability is available"""
        try:
            job_ctx = get_job_context()
            sip_participant = await self._find_sip_participant(job_ctx)
            
            return {
                "available": sip_participant is not None,
                "participant_count": len(job_ctx.room.remote_participants),
                "sip_participant_found": sip_participant is not None,
                "default_target": self.default_transfer_address
            }
            
        except Exception as e:
            logger.error(f"❌ Transfer capability check failed: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    def get_transfer_history(self, session_id: Optional[str] = None) -> dict:
        """Get transfer history for session or all sessions"""
        if session_id:
            return self.transfer_history.get(session_id, {})
        return self.transfer_history.copy()
    
    async def prepare_for_transfer(
        self, 
        session_data: RoadsideSessionData,
        message: str = "I'm connecting you to a human agent now. Please stay on the line."
    ) -> str:
        """
        Prepare session for transfer with appropriate messaging
        Returns the preparation message for the agent to speak
        """
        # Update session state
        session_data.transfer_requested = True
        session_data.add_interaction(
            agent_type="transfer_service",
            speaker="system",
            message="Preparing for human transfer"
        )
        
        # Return contextual message based on session data
        if session_data.get_completeness_status()["ready_for_dispatch"]:
            return f"{message} I have all your information and will pass it along to the agent."
        else:
            missing_info = session_data.get_missing_info()
            if missing_info:
                return f"{message} I'll let them know we still need: {', '.join(missing_info[:2])}."
            else:
                return message


# Global instance for dependency injection
transfer_service: Optional[TransferService] = None


def get_transfer_service() -> Optional[TransferService]:
    """Get the global transfer service instance"""
    return transfer_service


def initialize_transfer_service(
    default_transfer_address: str = "sip:voiceai@sip.linphone.org",
    **kwargs
) -> TransferService:
    """Initialize global transfer service"""
    global transfer_service
    transfer_service = TransferService(
        default_transfer_address=default_transfer_address,
        **kwargs
    )
    return transfer_service