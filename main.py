# main.py - ENHANCED VERSION WITH FULL RAG INTEGRATION
"""
Enhanced Roadside Assistance Agent
✅ Full RAG integration (no static responses)
✅ Male voice (Josh from ElevenLabs)
✅ Complete data collection (name, phone, address, vehicle)
✅ Detailed call transcripts for MongoDB indexing
✅ All monitoring features preserved
"""
import asyncio
import logging
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from livekit import rtc, api
from livekit.agents import (
    Agent, AgentSession, JobContext, RunContext,
    function_tool, get_job_context, ChatContext, ChatMessage,
    WorkerOptions, cli
)
from livekit.plugins import openai, silero, elevenlabs, deepgram

from dotenv import load_dotenv
load_dotenv()

# Import your working RAG system
try:
    from qdrant_rag_system import qdrant_rag
    from config import config
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("RAG system not available - using fallback responses")

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED DATA STRUCTURES
# ============================================================================

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
    """Enhanced vehicle information collection"""
    year: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    neutral_functional: Optional[bool] = None
    vehicle_type: Optional[str] = None  # SUV, sedan, truck, etc.

@dataclass
class LocationInfo:
    """Complete location information for precise dispatch"""
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    landmarks: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    
    @property
    def full_address(self) -> str:
        """Get complete formatted address"""
        parts = []
        if self.street_address:
            parts.append(self.street_address)
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        if self.zip_code:
            parts.append(self.zip_code)
        return ", ".join(parts) if parts else "Address incomplete"

@dataclass
class CustomerInfo:
    """Complete customer information"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    insurance_info: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Get complete formatted name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        return "Name not provided"

@dataclass
class ServiceRequest:
    """Enhanced service request structure"""
    service_type: ServiceType = ServiceType.UNKNOWN
    customer: CustomerInfo = None
    vehicle: VehicleInfo = None
    location: LocationInfo = None
    special_requirements: List[str] = None
    estimated_cost: Optional[str] = None
    job_number: Optional[str] = None
    timestamp: datetime = None
    priority: str = "normal"  # normal, urgent, emergency
    status: str = "in_progress"  # in_progress, completed, transferred

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
    """Enhanced transcript entry for MongoDB indexing"""
    speaker: str
    text: str
    timestamp: datetime
    is_final: bool
    confidence: Optional[float] = None
    extracted_entities: Optional[Dict] = None
    intent: Optional[str] = None
    rag_enhanced: bool = False
    response_time_ms: Optional[float] = None

# ============================================================================
# ENHANCED MONITORING AGENT
# ============================================================================

class EnhancedMonitoringAgent:
    """Enhanced monitoring agent with complete data collection"""
    
    def __init__(self, room_name: str = ""):
        self.transcripts: List[CallTranscript] = []
        self.service_request = ServiceRequest()
        self.room_name = room_name
        self.start_time = datetime.now()
        self.conversation_stage = "active"  # Let LLM determine stage naturally through context
        
        # Enhanced extraction patterns - keep these as they're data extraction, not conversation logic
        self.extraction_patterns = {
            'name': [
                re.compile(r'(?:my name is|i\'m|this is|call me)\s+([a-zA-Z\s]+)', re.IGNORECASE),
                re.compile(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', re.IGNORECASE),],}
    
    def extract_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive extraction with intelligent analysis
        Extract information but let LLM handle conversation flow
        """
        extracted = {}
        
        # Extract name with flexible patterns
        if not (self.service_request.customer.first_name and self.service_request.customer.last_name):
            for pattern in self.extraction_patterns['name']:
                match = pattern.search(text)
                if match:
                    if len(match.groups()) == 2:
                        first, last = match.group(1).strip().title(), match.group(2).strip().title()
                        if len(first) > 1 and len(last) > 1:  # Minimal validation - let LLM handle edge cases
                            self.service_request.customer.first_name = first
                            self.service_request.customer.last_name = last
                            extracted['customer_name'] = f"{first} {last}"
                            logger.info(f"⚡ EXTRACTED - Name: {first} {last}")
                            break
                    elif len(match.groups()) == 1:
                        name_parts = match.group(1).strip().title().split()
                        if len(name_parts) >= 2:
                            # Smart name parsing - let LLM ask for clarification if needed
                            self.service_request.customer.first_name = name_parts[0]
                            self.service_request.customer.last_name = " ".join(name_parts[1:])
                            extracted['customer_name'] = " ".join(name_parts)
                            logger.info(f"⚡ EXTRACTED - Name: {' '.join(name_parts)}")
                            break
                        elif len(name_parts) == 1:
                            # Only first name - let LLM ask for last name naturally
                            self.service_request.customer.first_name = name_parts[0]
                            extracted['customer_first_name_only'] = name_parts[0]
                            logger.info(f"⚡ EXTRACTED - First name only: {name_parts[0]}")
                            break
        
        # Extract phone with intelligent formatting
        if not self.service_request.customer.phone:
            phone_text = self._normalize_spoken_numbers(text)
            
            for pattern in self.extraction_patterns['phone']:
                match = pattern.search(phone_text)
                if match:
                    phone_digits = ''.join(filter(str.isdigit, ''.join(match.groups())))
                    
                    if len(phone_digits) >= 7:  # Relaxed validation - let LLM handle formatting questions
                        # Format phone number
                        phone = phone_digits[-10:] if len(phone_digits) >= 10 else phone_digits
                        if len(phone) == 10:
                            formatted_phone = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
                        else:
                            formatted_phone = phone  # Let LLM ask for missing digits
                        
                        self.service_request.customer.phone = formatted_phone
                        extracted['customer_phone'] = formatted_phone
                        logger.info(f"⚡ EXTRACTED - Phone: {formatted_phone}")
                        break
        
        # Smart address extraction
        self._extract_location_smart(text, extracted)
        
        # Intelligent vehicle extraction
        self._extract_vehicle_smart(text, extracted)
        
        # Service type detection with confidence
        self._extract_service_smart(text, extracted)
        
        return extracted
    
    def _normalize_spoken_numbers(self, text: str) -> str:
        """Convert spoken numbers to digits - this is data processing, not conversation logic"""
        phone_text = text.lower()
        spoken_to_digit = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'oh': '0'  # Handle "oh" as zero in phone numbers
        }
        for word, digit in spoken_to_digit.items():
            phone_text = phone_text.replace(word, digit)
        return phone_text
    
    def _extract_location_smart(self, text: str, extracted: Dict):
        """Data extraction for location - let LLM handle conversation about missing pieces"""
        # Extract street address
        if not self.service_request.location.street_address:
            for pattern in self.extraction_patterns['address']:
                match = pattern.search(text)
                if match:
                    if len(match.groups()) == 2:
                        address = f"{match.group(1)} {match.group(2)}"
                    else:
                        address = match.group(1)
                    
                    # Basic cleaning - let LLM handle validation in conversation
                    address = address.strip().title()
                    if len(address) > 3:  # Minimal validation
                        self.service_request.location.street_address = address
                        extracted['street_address'] = address
                        logger.info(f"⚡ EXTRACTED - Address: {address}")
                        break
        
        # Extract city and state
        for pattern in self.extraction_patterns['city_state']:
            match = pattern.search(text)
            if match:
                city = match.group(1).strip().title()
                state_raw = match.group(2).strip()
                
                # Basic normalization - let LLM handle edge cases
                if len(state_raw) == 2:
                    state = state_raw.upper()
                else:
                    state = state_raw.title()
                
                if not self.service_request.location.city and len(city) > 1:
                    self.service_request.location.city = city
                    extracted['city'] = city
                    logger.info(f"⚡ EXTRACTED - City: {city}")
                
                if not self.service_request.location.state and len(state) <= 30:
                    self.service_request.location.state = state
                    extracted['state'] = state
                    logger.info(f"⚡ EXTRACTED - State: {state}")
                break
        
        # Extract ZIP
        if not self.service_request.location.zip_code:
            for pattern in self.extraction_patterns['zip_code']:
                match = pattern.search(text)
                if match:
                    zip_code = match.group(1)
                    # Basic validation - let LLM handle complex cases
                    if zip_code.isdigit() and 4 <= len(zip_code) <= 5:
                        self.service_request.location.zip_code = zip_code
                        extracted['zip_code'] = zip_code
                        logger.info(f"⚡ EXTRACTED - ZIP: {zip_code}")
                        break
    
    def _extract_vehicle_smart(self, text: str, extracted: Dict):
        """Data extraction for vehicle - let LLM handle conversation flow"""
        if not (self.service_request.vehicle.year and self.service_request.vehicle.make):
            for pattern in self.extraction_patterns['vehicle']:
                match = pattern.search(text)
                if match:
                    groups = match.groups()
                    year, make, model = None, None, None
                    
                    # Find year in any position
                    for group in groups:
                        if group.isdigit() and 1990 <= int(group) <= 2030:
                            year = group
                            break
                    
                    # Extract make and model from remaining groups
                    remaining = [g for g in groups if g != year]
                    if len(remaining) >= 2:
                        make, model = remaining[0].title(), remaining[1].title()
                    elif len(remaining) == 1:
                        make = remaining[0].title()
                    
                    # Store what we found - let LLM ask for missing pieces
                    if year:
                        self.service_request.vehicle.year = year
                        extracted['vehicle_year'] = year
                    if make:
                        self.service_request.vehicle.make = make
                        extracted['vehicle_make'] = make
                    if model:
                        self.service_request.vehicle.model = model
                        extracted['vehicle_model'] = model
                    
                    if year or make or model:
                        vehicle_parts = [p for p in [year, make, model] if p]
                        logger.info(f"⚡ EXTRACTED - Vehicle parts: {' '.join(vehicle_parts)}")
                        break
    
    def _extract_service_smart(self, text: str, extracted: Dict):
        """Service extraction - let LLM handle clarification"""
        if self.service_request.service_type == ServiceType.UNKNOWN:
            service_matches = []
            
            for pattern, service_type in self.extraction_patterns['service']:
                if pattern.search(text):
                    service_matches.append(service_type)
            
            # Extract what we can - let LLM handle ambiguity
            if len(service_matches) == 1:
                self.service_request.service_type = service_matches[0]
                extracted['service_type'] = service_matches[0].value
                logger.info(f"⚡ EXTRACTED - Service: {service_matches[0].value}")
            elif len(service_matches) > 1:
                extracted['possible_services'] = [s.value for s in service_matches]
                logger.info(f"⚡ MULTIPLE SERVICES DETECTED: {[s.value for s in service_matches]}")
                # Let LLM ask for clarification naturally
    
    def add_transcript(self, speaker: str, text: str, is_final: bool = True, 
                      confidence: float = None, rag_enhanced: bool = False,
                      response_time_ms: float = None):
        """Add enhanced transcript entry"""
        transcript = CallTranscript(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            is_final=is_final,
            confidence=confidence,
            rag_enhanced=rag_enhanced,
            response_time_ms=response_time_ms
        )
        self.transcripts.append(transcript)
        
        if speaker == "customer":
            logger.info(f"👤 USER: {text}")
        elif speaker == "agent":
            logger.info(f"🤖 AGENT: {text}")
    
    def get_data_completeness(self) -> Dict[str, bool]:
        """Check what data we have collected"""
        return {
            "customer_first_name": bool(self.service_request.customer.first_name),
            "customer_last_name": bool(self.service_request.customer.last_name),
            "customer_phone": bool(self.service_request.customer.phone),
            "street_address": bool(self.service_request.location.street_address),
            "city": bool(self.service_request.location.city),
            "state": bool(self.service_request.location.state),
            "zip_code": bool(self.service_request.location.zip_code),
            "vehicle_year": bool(self.service_request.vehicle.year),
            "vehicle_make": bool(self.service_request.vehicle.make),
            "vehicle_model": bool(self.service_request.vehicle.model),
            "service_type": self.service_request.service_type != ServiceType.UNKNOWN,
        }
    
    def update_conversation_stage(self):
        """Let LLM determine conversation stage naturally - remove hard logic"""
        completeness = self.get_data_completeness()
        complete_count = sum(completeness.values())
        total_count = len(completeness)
        
        # Provide context to LLM instead of hard rules
        if complete_count == 0:
            self.conversation_stage = "initial_contact"
        elif complete_count < total_count // 2:
            self.conversation_stage = "gathering_basic_info"
        elif complete_count < total_count:
            self.conversation_stage = "completing_details"
        else:
            self.conversation_stage = "ready_for_service"
        """Provide context about missing data to LLM - let LLM decide what to ask"""
        completeness = self.get_data_completeness()
        missing_items = []
        
        if not completeness["customer_first_name"] or not completeness["customer_last_name"]:
            missing_items.append("customer full name (first and last)")
        
        if not completeness["customer_phone"]:
            missing_items.append("customer phone number")
        
        if not completeness["street_address"]:
            missing_items.append("exact street address")
        
        if not completeness["city"] or not completeness["state"]:
            missing_items.append("city and state")
        
        if not completeness["zip_code"]:
            missing_items.append("ZIP code")
        
        if not completeness["vehicle_year"] or not completeness["vehicle_make"] or not completeness["vehicle_model"]:
            missing_items.append("vehicle details (year, make, model)")
        
        if not completeness["service_type"]:
            missing_items.append("type of service needed")
        
        if missing_items:
            return f"MISSING_DATA: {', '.join(missing_items)}. Ask for the most important missing item naturally in conversation."
        
        return "DATA_COMPLETE: All required information collected. Proceed with service confirmation."
    
    async def save_conversation_data(self):
        """Save enhanced conversation data for MongoDB indexing"""
        summary = {
            "_id": f"call_{self.room_name}_{int(self.start_time.timestamp())}",
            "call_metadata": {
                "room_name": self.room_name,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "conversation_stage": self.conversation_stage,
                "total_exchanges": len(self.transcripts),
            },
            "service_request": {
                "service_type": self.service_request.service_type.value,
                "priority": self.service_request.priority,
                "status": self.service_request.status,
                "customer": {
                    "first_name": self.service_request.customer.first_name,
                    "last_name": self.service_request.customer.last_name,
                    "full_name": self.service_request.customer.full_name,
                    "phone": self.service_request.customer.phone,
                    "email": self.service_request.customer.email,
                },
                "vehicle": {
                    "year": self.service_request.vehicle.year,
                    "make": self.service_request.vehicle.make,
                    "model": self.service_request.vehicle.model,
                    "color": self.service_request.vehicle.color,
                    "vehicle_type": self.service_request.vehicle.vehicle_type,
                    "neutral_functional": self.service_request.vehicle.neutral_functional,
                },
                "location": {
                    "street_address": self.service_request.location.street_address,
                    "city": self.service_request.location.city,
                    "state": self.service_request.location.state,
                    "zip_code": self.service_request.location.zip_code,
                    "full_address": self.service_request.location.full_address,
                    "landmarks": self.service_request.location.landmarks,
                },
                "estimated_cost": self.service_request.estimated_cost,
                "special_requirements": self.service_request.special_requirements,
                "job_number": self.service_request.job_number,
            },
            "data_completeness": self.get_data_completeness(),
            "conversation_transcript": [
                {
                    "speaker": t.speaker,
                    "text": t.text,
                    "timestamp": t.timestamp.isoformat(),
                    "is_final": t.is_final,
                    "confidence": t.confidence,
                    "rag_enhanced": t.rag_enhanced,
                    "response_time_ms": t.response_time_ms,
                    "extracted_entities": t.extracted_entities,
                    "intent": t.intent,
                }
                for t in self.transcripts
            ],
            "analytics": {
                "rag_enhanced_responses": sum(1 for t in self.transcripts if t.rag_enhanced),
                "average_response_time_ms": sum(t.response_time_ms for t in self.transcripts if t.response_time_ms) / max(1, len([t for t in self.transcripts if t.response_time_ms])),
                "conversation_quality_score": sum(self.get_data_completeness().values()) / len(self.get_data_completeness()) * 100,
            }
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_call_{self.room_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"💾 ENHANCED TRANSCRIPT SAVED: {filename}")
        except Exception as e:
            logger.error(f"❌ FAILED TO SAVE TRANSCRIPT: {e}")

# ============================================================================
# ENHANCED ROADSIDE ASSISTANCE AGENT
# ============================================================================

class EnhancedRoadsideAgent(Agent):
    """Enhanced Roadside Assistance Agent with full RAG integration"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance agent.

CRITICAL REQUIREMENTS:
- Keep responses under 30 words for phone clarity
- ALWAYS use information from [QdrantRAG] when available - this is your knowledge base
- ALWAYS use information from [EXTRACTED_INFO] to track what you already know
- ALWAYS use information from [MISSING_DATA] to understand what you still need

NATURAL CONVERSATION FLOW:
- Ask for missing information naturally based on context
- Don't follow a rigid script - adapt based on what the customer says
- If customer provides partial info, acknowledge it and naturally ask for what's still needed
- Example: Customer says "I'm John" → "Thanks John! What's your last name and phone number?"

INFORMATION TO COLLECT (but ask naturally):
- Customer's full name (first and last)
- Phone number for callback
- Exact location (street address, city, state, ZIP code)
- Vehicle details (year, make, model)
- Type of service needed
- For towing: whether neutral gear works

RESPONSE STYLE:
- Natural and conversational: "Thanks! I have your name. Where exactly are you located?"
- Acknowledge what you received: "Got it, so you're in Chicago. What's the street address?"
- Be helpful: "I have most of your info. Just need your phone number to complete this."

TOOLS USAGE:
- search_knowledge: Use for pricing, services, policies, business hours
- transfer_to_human: ONLY when explicitly requested

Let the conversation flow naturally - don't be robotic about collecting information."""
        )
        
        self.monitoring_agent: Optional[EnhancedMonitoringAgent] = None
        self.processing = False
        self.transfer_sip_address = "sip:voiceai@sip.linphone.org"
    
    def set_room_name(self, room_name: str):
        """Initialize enhanced monitoring agent"""
        self.monitoring_agent = EnhancedMonitoringAgent(room_name)
        logger.info(f"⚡ ENHANCED MONITORING INITIALIZED: {room_name}")
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Enhanced turn completion with comprehensive data extraction"""
        try:
            if not self.monitoring_agent:
                return
            
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 2 or self.processing:
                return
            
            self.processing = True
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Comprehensive extraction
                extracted = self.monitoring_agent.extract_comprehensive(user_text)
                
                if extracted:
                    # Add extraction context
                    context_parts = []
                    for key, value in extracted.items():
                        context_parts.append(f"{key}: {value}")
                    
                    if context_parts:
                        context = f"[EXTRACTED_INFO: {'; '.join(context_parts)}]"
                        turn_ctx.add_message(role="system", content=context)
                
                # Try RAG search for relevant information
                if RAG_AVAILABLE:
                    try:
                        # Use configurable timeout
                        timeout = getattr(config, 'rag_timeout_ms', 1500) / 1000.0
                        results = await asyncio.wait_for(
                            qdrant_rag.search(user_text, limit=getattr(config, 'search_limit', 3)),
                            timeout=timeout
                        )
                        
                        if results and len(results) > 0:
                            best_result = results[0]
                            threshold = getattr(config, 'similarity_threshold', 0.25)
                            
                            if best_result["score"] >= threshold:
                                raw_content = best_result["text"]
                                context = self._clean_content_for_voice(raw_content)
                                turn_ctx.add_message(
                                    role="system",
                                    content=f"[QdrantRAG]: {context}"
                                )
                                
                                # Mark as RAG enhanced
                                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                                logger.info(f"⚡ RAG context injected (score: {best_result['score']:.3f}, {response_time:.1f}ms)")
                                
                                # Add to transcript with RAG flag
                                if self.monitoring_agent:
                                    self.monitoring_agent.add_transcript(
                                        "customer", user_text, True, rag_enhanced=True,
                                        response_time_ms=response_time
                                    )
                                return
                            else:
                                logger.info(f"⚠️ Low relevance score: {best_result['score']:.3f} < {threshold}")
                                
                    except asyncio.TimeoutError:
                        logger.debug("⚡ RAG timeout - continuing without context")
                    except Exception as e:
                        logger.debug(f"⚡ RAG error: {e}")
                
                # Add missing data context for LLM to handle naturally
                missing_context = self.monitoring_agent.get_missing_data_context()
                if "MISSING_DATA" in missing_context:
                    turn_ctx.add_message(
                        role="system",
                        content=f"[{missing_context}]"
                    )
                        
            finally:
                self.processing = False
                
        except Exception as e:
            logger.error(f"❌ on_user_turn_completed error: {e}")
            self.processing = False
    
    def _clean_content_for_voice(self, content: str) -> str:
        """Clean content for voice response"""
        try:
            # Remove formatting characters
            content = content.replace("Q: ", "").replace("A: ", "")
            content = content.replace("■", "").replace("●", "").replace("•", "")
            content = content.replace("- ", "").replace("* ", "")
            
            # Handle multi-line content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if lines:
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
    
    # ========================================================================
    # ENHANCED FUNCTION TOOLS
    # ========================================================================
    
    @function_tool()
    async def search_knowledge(self, query: str) -> str:
        """
        Search the knowledge base for real-time information.
        Use for ALL service information including pricing, business hours, policies.
        This replaces static responses with live data from your knowledge base.
        """
        try:
            logger.info(f"🔍 Searching knowledge base: {query}")
            
            if RAG_AVAILABLE:
                timeout = getattr(config, 'rag_timeout_ms', 1500) / 1000.0
                results = await asyncio.wait_for(
                    qdrant_rag.search(query, limit=3),
                    timeout=timeout
                )
                
                if results and len(results) > 0:
                    best_result = results[0]
                    content = self._clean_content_for_voice(best_result["text"])
                    logger.info(f"📊 Found result with score: {best_result['score']:.3f}")
                    
                    # Mark response as RAG enhanced
                    if self.monitoring_agent:
                        self.monitoring_agent.add_transcript(
                            "agent", f"Knowledge search: {content}", True,
                            rag_enhanced=True
                        )
                    
                    return content
                else:
                    logger.warning("⚠️ No relevant information found")
                    return "I don't have specific information about that in my current knowledge base. Let me transfer you to someone who can help with those details."
            
            return "I'm having trouble accessing the knowledge base right now. Let me transfer you to someone who can help."
            
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return "I'm having trouble accessing the information. Let me transfer you to someone who can help."

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """
        Transfer caller to human agent.
        
        The LLM should call this function when:
        - Customer explicitly requests human help
        - Customer expresses frustration with AI
        - Complex issues that need human expertise
        - Customer directly asks for transfer
        
        The LLM should confirm before transferring.
        """
        try:
            logger.info("🔄 HUMAN TRANSFER INITIATED")
            
            # Save enhanced conversation data
            if self.monitoring_agent:
                self.monitoring_agent.service_request.status = "transferred"
                await self.monitoring_agent.save_conversation_data()
            
            # Get job context and find SIP participant
            job_ctx = get_job_context()
            sip_participant = None
            
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                    sip_participant = participant
                    break
            
            if not sip_participant:
                return "I'm sorry, I couldn't find any active participants to transfer."
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=self.transfer_sip_address,
                play_dialtone=True,
            )
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=30.0
            )
            
            logger.info("✅ Transfer completed successfully")
            return "Connecting you to a human agent now."
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "Having trouble with transfer. Let me continue helping you."

# ============================================================================
# ENHANCED SESSION CREATION
# ============================================================================

async def create_enhanced_session() -> AgentSession:
    """Create enhanced session with male voice and optimized settings"""
    
    # Configure ElevenLabs TTS with male voice (Josh)
    tts_engine = elevenlabs.TTS(
        voice_id="TxGEqnHWrfWFTfGW9XjX",  # Josh - professional male voice
        model="eleven_turbo_v2_5",       # Fast stable model
        
        # Professional male voice settings
        voice_settings=elevenlabs.VoiceSettings(
            stability=0.6,              # Slightly more stable for professional sound
            similarity_boost=0.8,       # High similarity
            style=0.3,                  # Moderate style for clarity
            speed=1.0,                  # Normal speed
            use_speaker_boost=True      # Enable for better quality
        ),
        
        # Conservative settings to avoid API errors
        inactivity_timeout=300,
        enable_ssml_parsing=False,
    )
    logger.info("🎙️ ElevenLabs TTS configured with Josh (male voice)")
    
    session = AgentSession(
        # Fast STT - using proven configuration
        stt=deepgram.STT(
            model="nova-2-general",
            language="en",
        ),
        
        # Fast LLM
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
        ),
        
        # ElevenLabs TTS with male voice
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

# ============================================================================
# ENHANCED ENTRYPOINT
# ============================================================================

async def entrypoint(ctx: JobContext):
    """Enhanced entrypoint with full RAG integration"""
    logger.info("⚡ === ENHANCED ROADSIDE AGENT WITH FULL RAG ===")
    
    # Connect immediately
    await ctx.connect()
    logger.info("✅ CONNECTED")
    
    # Initialize RAG in background
    if RAG_AVAILABLE:
        asyncio.create_task(qdrant_rag.initialize())
        logger.info("🧠 RAG: Starting background initialization...")
    
    # Create enhanced session and agent
    session = await create_enhanced_session()
    agent = EnhancedRoadsideAgent()
    agent.set_room_name(ctx.room.name)
    
    # Start session
    logger.info("⚡ STARTING ENHANCED SESSION...")
    await session.start(room=ctx.room, agent=agent)
    
    # System ready
    logger.info("⚡ ENHANCED SYSTEM READY!")
    logger.info("🎯 TARGET LATENCY: <2 seconds")
    logger.info("🔥 TTS: ✅ ElevenLabs Josh (male voice)")
    logger.info("🧠 RAG: ✅ Full knowledge base integration")
    logger.info("📊 MONITORING: ✅ Complete data collection")
    logger.info("🔄 TRANSFER: ✅ Available")
    logger.info(f"📞 Room: {ctx.room.name}")
    
    # Professional greeting
    await session.generate_reply(
        instructions="Give a professional greeting: 'Roadside assistance, this is Mark. How can I help with your vehicle today?' Keep it brief and professional."
    )
    
    logger.info("⚡ GREETING SENT - ENHANCED SYSTEM ACTIVE")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        logger.info("⚡ STARTING ENHANCED ROADSIDE SYSTEM")
        logger.info("🎯 TARGET: Complete data collection + full RAG")
        logger.info("🔥 TECH: Male voice + dynamic knowledge base")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except KeyboardInterrupt:
        logger.info("🛑 SYSTEM SHUTDOWN")
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)
    finally:
        logger.info("👋 ENHANCED SYSTEM STOPPED")),
                re.compile(r'([A-Z][a-z]+)\s+([A-Z][a-z]+)', re.IGNORECASE),
            ],
            'phone': [
                re.compile(r'(\d{3})\s*[-.]?\s*(\d{3})\s*[-.]?\s*(\d{4})'),
                re.compile(r'\((\d{3})\)\s*(\d{3})\s*[-.]?\s*(\d{4})'),
                re.compile(r'(?:eight|one|two|three|four|five|six|seven|nine|zero)', re.IGNORECASE),
            ],
            'address': [
                re.compile(r'(\d+)\s+([A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd))', re.IGNORECASE),
                re.compile(r'([A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd))', re.IGNORECASE),
            ],
            'city_state': [
                re.compile(r'([A-Za-z\s]+),\s*([A-Z]{2})', re.IGNORECASE),
                re.compile(r'in\s+([A-Za-z\s]+),?\s*([A-Za-z\s]+)', re.IGNORECASE),
            ],
            'zip_code': [
                re.compile(r'\b(\d{5})\b'),
                re.compile(r'(?:zip|zip code)\s*(\d{5})', re.IGNORECASE),
            ],
            'vehicle': [
                re.compile(r'(\d{4})\s+([A-Za-z]+)\s+([A-Za-z\s]+)', re.IGNORECASE),
                re.compile(r'([A-Za-z]+)\s+([A-Za-z\s]+)\s+(\d{4})', re.IGNORECASE),
            ],
            'service': [
                (re.compile(r'\b(?:tow|towing|stuck|won\'t start|can\'t start|broke down)\b', re.IGNORECASE), ServiceType.TOWING),
                (re.compile(r'\b(?:jump\s*start|battery|dead battery|won\'t start)\b', re.IGNORECASE), ServiceType.JUMP_START),
                (re.compile(r'\b(?:tire|flat tire|tire change)\b', re.IGNORECASE), ServiceType.TIRE_CHANGE),
                (re.compile(r'\b(?:locked out|lockout|keys|key)\b', re.IGNORECASE), ServiceType.LOCKOUT),
                (re.compile(r'\b(?:winch|stuck|pull|off road)\b', re.IGNORECASE), ServiceType.WINCH_OUT),
                (re.compile(r'\b(?:fuel|gas|gasoline|out of gas)\b', re.IGNORECASE), ServiceType.FUEL_DELIVERY),
            ]
        }
    
    def extract_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive extraction with intelligent analysis
        Extract information but let LLM handle conversation flow
        """
        extracted = {}
        
        # Extract name with flexible patterns
        if not (self.service_request.customer.first_name and self.service_request.customer.last_name):
            for pattern in self.extraction_patterns['name']:
                match = pattern.search(text)
                if match:
                    if len(match.groups()) == 2:
                        first, last = match.group(1).strip().title(), match.group(2).strip().title()
                        if len(first) > 1 and len(last) > 1 and not any(char.isdigit() for char in first + last):
                            self.service_request.customer.first_name = first
                            self.service_request.customer.last_name = last
                            extracted['customer_name'] = f"{first} {last}"
                            logger.info(f"⚡ EXTRACTED - Name: {first} {last}")
                            break
                    elif len(match.groups()) == 1:
                        name_parts = match.group(1).strip().title().split()
                        if len(name_parts) >= 2:
                            # Smart name parsing - let LLM ask for clarification if needed
                            self.service_request.customer.first_name = name_parts[0]
                            self.service_request.customer.last_name = " ".join(name_parts[1:])
                            extracted['customer_name'] = " ".join(name_parts)
                            logger.info(f"⚡ EXTRACTED - Name: {' '.join(name_parts)}")
                            break
                        elif len(name_parts) == 1:
                            # Only first name - let LLM ask for last name naturally
                            self.service_request.customer.first_name = name_parts[0]
                            extracted['customer_first_name_only'] = name_parts[0]
                            logger.info(f"⚡ EXTRACTED - First name only: {name_parts[0]}")
                            break
        
        # Extract phone with intelligent formatting
        if not self.service_request.customer.phone:
            phone_text = self._normalize_spoken_numbers(text)
            
            for pattern in self.extraction_patterns['phone']:
                match = pattern.search(phone_text)
                if match:
                    phone_digits = ''.join(filter(str.isdigit, ''.join(match.groups())))
                    
                    if len(phone_digits) >= 10:
                        # Format phone number intelligently
                        phone = phone_digits[-10:]  # Last 10 digits
                        formatted_phone = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
                        self.service_request.customer.phone = formatted_phone
                        extracted['customer_phone'] = formatted_phone
                        logger.info(f"⚡ EXTRACTED - Phone: {formatted_phone}")
                        break
        
        # Smart address extraction
        self._extract_location_smart(text, extracted)
        
        # Intelligent vehicle extraction
        self._extract_vehicle_smart(text, extracted)
        
        # Service type detection with confidence
        self._extract_service_smart(text, extracted)
        
        return extracted
    
    def _normalize_spoken_numbers(self, text: str) -> str:
        """Convert spoken numbers to digits intelligently"""
        phone_text = text.lower()
        spoken_to_digit = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'oh': '0'  # Handle "oh" as zero in phone numbers
        }
        for word, digit in spoken_to_digit.items():
            phone_text = phone_text.replace(word, digit)
        return phone_text
    
    def _extract_location_smart(self, text: str, extracted: Dict):
        """Smart location extraction with context awareness"""
        # Extract street address
        if not self.service_request.location.street_address:
            for pattern in self.extraction_patterns['address']:
                match = pattern.search(text)
                if match:
                    if len(match.groups()) == 2:
                        address = f"{match.group(1)} {match.group(2)}"
                    else:
                        address = match.group(1)
                    
                    # Clean and validate address
                    address = address.strip().title()
                    if len(address) > 5:  # Basic validation
                        self.service_request.location.street_address = address
                        extracted['street_address'] = address
                        logger.info(f"⚡ EXTRACTED - Address: {address}")
                        break
        
        # Extract city and state with better parsing
        for pattern in self.extraction_patterns['city_state']:
            match = pattern.search(text)
            if match:
                city = match.group(1).strip().title()
                state_raw = match.group(2).strip()
                
                # Smart state handling (abbreviations vs full names)
                if len(state_raw) == 2:
                    state = state_raw.upper()
                else:
                    state = state_raw.title()
                
                if not self.service_request.location.city and len(city) > 2:
                    self.service_request.location.city = city
                    extracted['city'] = city
                    logger.info(f"⚡ EXTRACTED - City: {city}")
                
                if not self.service_request.location.state and len(state) <= 20:
                    self.service_request.location.state = state
                    extracted['state'] = state
                    logger.info(f"⚡ EXTRACTED - State: {state}")
                break
        
        # Extract ZIP with validation
        if not self.service_request.location.zip_code:
            for pattern in self.extraction_patterns['zip_code']:
                match = pattern.search(text)
                if match:
                    zip_code = match.group(1)
                    # Basic ZIP validation
                    if zip_code.isdigit() and len(zip_code) == 5:
                        self.service_request.location.zip_code = zip_code
                        extracted['zip_code'] = zip_code
                        logger.info(f"⚡ EXTRACTED - ZIP: {zip_code}")
                        break
    
    def _extract_vehicle_smart(self, text: str, extracted: Dict):
        """Smart vehicle extraction with flexible patterns"""
        if not (self.service_request.vehicle.year and self.service_request.vehicle.make):
            for pattern in self.extraction_patterns['vehicle']:
                match = pattern.search(text)
                if match:
                    groups = match.groups()
                    year, make, model = None, None, None
                    
                    # Intelligently identify which group is the year
                    for group in groups:
                        if group.isdigit() and 1990 <= int(group) <= 2030:
                            year = group
                            break
                    
                    # Extract make and model from remaining groups
                    remaining = [g for g in groups if g != year]
                    if len(remaining) >= 2:
                        make, model = remaining[0].title(), remaining[1].title()
                    elif len(remaining) == 1:
                        make = remaining[0].title()
                    
                    # Store extracted vehicle info
                    if year:
                        self.service_request.vehicle.year = year
                        extracted['vehicle_year'] = year
                    if make:
                        self.service_request.vehicle.make = make
                        extracted['vehicle_make'] = make
                    if model:
                        self.service_request.vehicle.model = model
                        extracted['vehicle_model'] = model
                    
                    if year and make:
                        vehicle_info = f"{year} {make}"
                        if model:
                            vehicle_info += f" {model}"
                        logger.info(f"⚡ EXTRACTED - Vehicle: {vehicle_info}")
                        break
    
    def _extract_service_smart(self, text: str, extracted: Dict):
        """Smart service type extraction with confidence scoring"""
        if self.service_request.service_type == ServiceType.UNKNOWN:
            service_matches = []
            
            for pattern, service_type in self.extraction_patterns['service']:
                if pattern.search(text):
                    service_matches.append(service_type)
            
            # If multiple matches, let LLM clarify
            if len(service_matches) == 1:
                self.service_request.service_type = service_matches[0]
                extracted['service_type'] = service_matches[0].value
                logger.info(f"⚡ EXTRACTED - Service: {service_matches[0].value}")
            elif len(service_matches) > 1:
                extracted['possible_services'] = [s.value for s in service_matches]
                logger.info(f"⚡ MULTIPLE SERVICES DETECTED: {[s.value for s in service_matches]}")
                # Let LLM ask for clarification
    
    def add_transcript(self, speaker: str, text: str, is_final: bool = True, 
                      confidence: float = None, rag_enhanced: bool = False,
                      response_time_ms: float = None):
        """Add enhanced transcript entry"""
        transcript = CallTranscript(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            is_final=is_final,
            confidence=confidence,
            rag_enhanced=rag_enhanced,
            response_time_ms=response_time_ms
        )
        self.transcripts.append(transcript)
        
        if speaker == "customer":
            logger.info(f"👤 USER: {text}")
        elif speaker == "agent":
            logger.info(f"🤖 AGENT: {text}")
    
    def get_data_completeness(self) -> Dict[str, bool]:
        """Check what data we have collected"""
        return {
            "customer_first_name": bool(self.service_request.customer.first_name),
            "customer_last_name": bool(self.service_request.customer.last_name),
            "customer_phone": bool(self.service_request.customer.phone),
            "street_address": bool(self.service_request.location.street_address),
            "city": bool(self.service_request.location.city),
            "state": bool(self.service_request.location.state),
            "zip_code": bool(self.service_request.location.zip_code),
            "vehicle_year": bool(self.service_request.vehicle.year),
            "vehicle_make": bool(self.service_request.vehicle.make),
            "vehicle_model": bool(self.service_request.vehicle.model),
            "service_type": self.service_request.service_type != ServiceType.UNKNOWN,
        }
    
    def update_conversation_stage(self):
        """Let LLM determine conversation stage naturally - remove hard logic"""
        completeness = self.get_data_completeness()
        complete_count = sum(completeness.values())
        total_count = len(completeness)
        
        # Provide context to LLM instead of hard rules
        if complete_count == 0:
            self.conversation_stage = "initial_contact"
        elif complete_count < total_count // 2:
            self.conversation_stage = "gathering_basic_info"
        elif complete_count < total_count:
            self.conversation_stage = "completing_details"
        else:
            self.conversation_stage = "ready_for_service"
        """Provide context about missing data to LLM - let LLM decide what to ask"""
        completeness = self.get_data_completeness()
        missing_items = []
        
        if not completeness["customer_first_name"] or not completeness["customer_last_name"]:
            missing_items.append("customer full name (first and last)")
        
        if not completeness["customer_phone"]:
            missing_items.append("customer phone number")
        
        if not completeness["street_address"]:
            missing_items.append("exact street address")
        
        if not completeness["city"] or not completeness["state"]:
            missing_items.append("city and state")
        
        if not completeness["zip_code"]:
            missing_items.append("ZIP code")
        
        if not completeness["vehicle_year"] or not completeness["vehicle_make"] or not completeness["vehicle_model"]:
            missing_items.append("vehicle details (year, make, model)")
        
        if not completeness["service_type"]:
            missing_items.append("type of service needed")
        
        if missing_items:
            return f"MISSING_DATA: {', '.join(missing_items)}. Ask for the most important missing item naturally in conversation."
        
        return "DATA_COMPLETE: All required information collected. Proceed with service confirmation."
    
    async def save_conversation_data(self):
        """Save enhanced conversation data for MongoDB indexing"""
        summary = {
            "_id": f"call_{self.room_name}_{int(self.start_time.timestamp())}",
            "call_metadata": {
                "room_name": self.room_name,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "conversation_stage": self.conversation_stage,
                "total_exchanges": len(self.transcripts),
            },
            "service_request": {
                "service_type": self.service_request.service_type.value,
                "priority": self.service_request.priority,
                "status": self.service_request.status,
                "customer": {
                    "first_name": self.service_request.customer.first_name,
                    "last_name": self.service_request.customer.last_name,
                    "full_name": self.service_request.customer.full_name,
                    "phone": self.service_request.customer.phone,
                    "email": self.service_request.customer.email,
                },
                "vehicle": {
                    "year": self.service_request.vehicle.year,
                    "make": self.service_request.vehicle.make,
                    "model": self.service_request.vehicle.model,
                    "color": self.service_request.vehicle.color,
                    "vehicle_type": self.service_request.vehicle.vehicle_type,
                    "neutral_functional": self.service_request.vehicle.neutral_functional,
                },
                "location": {
                    "street_address": self.service_request.location.street_address,
                    "city": self.service_request.location.city,
                    "state": self.service_request.location.state,
                    "zip_code": self.service_request.location.zip_code,
                    "full_address": self.service_request.location.full_address,
                    "landmarks": self.service_request.location.landmarks,
                },
                "estimated_cost": self.service_request.estimated_cost,
                "special_requirements": self.service_request.special_requirements,
                "job_number": self.service_request.job_number,
            },
            "data_completeness": self.get_data_completeness(),
            "conversation_transcript": [
                {
                    "speaker": t.speaker,
                    "text": t.text,
                    "timestamp": t.timestamp.isoformat(),
                    "is_final": t.is_final,
                    "confidence": t.confidence,
                    "rag_enhanced": t.rag_enhanced,
                    "response_time_ms": t.response_time_ms,
                    "extracted_entities": t.extracted_entities,
                    "intent": t.intent,
                }
                for t in self.transcripts
            ],
            "analytics": {
                "rag_enhanced_responses": sum(1 for t in self.transcripts if t.rag_enhanced),
                "average_response_time_ms": sum(t.response_time_ms for t in self.transcripts if t.response_time_ms) / max(1, len([t for t in self.transcripts if t.response_time_ms])),
                "conversation_quality_score": sum(self.get_data_completeness().values()) / len(self.get_data_completeness()) * 100,
            }
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_call_{self.room_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"💾 ENHANCED TRANSCRIPT SAVED: {filename}")
        except Exception as e:
            logger.error(f"❌ FAILED TO SAVE TRANSCRIPT: {e}")

# ============================================================================
# ENHANCED ROADSIDE ASSISTANCE AGENT
# ============================================================================

class EnhancedRoadsideAgent(Agent):
    """Enhanced Roadside Assistance Agent with full RAG integration"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Mark, a professional roadside assistance agent.

CRITICAL REQUIREMENTS:
- Keep responses under 30 words for phone clarity
- ALWAYS use information from [QdrantRAG] when available - this is your knowledge base
- ALWAYS use information from [EXTRACTED_INFO] to track what you already know
- ALWAYS use information from [MISSING_DATA] to understand what you still need

NATURAL CONVERSATION FLOW:
- Ask for missing information naturally based on context
- Don't follow a rigid script - adapt based on what the customer says
- If customer provides partial info, acknowledge it and naturally ask for what's still needed
- Example: Customer says "I'm John" → "Thanks John! What's your last name and phone number?"

INFORMATION TO COLLECT (but ask naturally):
- Customer's full name (first and last)
- Phone number for callback
- Exact location (street address, city, state, ZIP code)
- Vehicle details (year, make, model)
- Type of service needed
- For towing: whether neutral gear works

RESPONSE STYLE:
- Natural and conversational: "Thanks! I have your name. Where exactly are you located?"
- Acknowledge what you received: "Got it, so you're in Chicago. What's the street address?"
- Be helpful: "I have most of your info. Just need your phone number to complete this."

TOOLS USAGE:
- search_knowledge: Use for pricing, services, policies, business hours
- transfer_to_human: ONLY when explicitly requested

Let the conversation flow naturally - don't be robotic about collecting information."""
        )
        
        self.monitoring_agent: Optional[EnhancedMonitoringAgent] = None
        self.processing = False
        self.transfer_sip_address = "sip:voiceai@sip.linphone.org"
    
    def set_room_name(self, room_name: str):
        """Initialize enhanced monitoring agent"""
        self.monitoring_agent = EnhancedMonitoringAgent(room_name)
        logger.info(f"⚡ ENHANCED MONITORING INITIALIZED: {room_name}")
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Enhanced turn completion with comprehensive data extraction"""
        try:
            if not self.monitoring_agent:
                return
            
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 2 or self.processing:
                return
            
            self.processing = True
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Comprehensive extraction
                extracted = self.monitoring_agent.extract_comprehensive(user_text)
                
                if extracted:
                    # Add extraction context
                    context_parts = []
                    for key, value in extracted.items():
                        context_parts.append(f"{key}: {value}")
                    
                    if context_parts:
                        context = f"[EXTRACTED_INFO: {'; '.join(context_parts)}]"
                        turn_ctx.add_message(role="system", content=context)
                
                # Skip RAG for transfer requests - let LLM handle this naturally
                transfer_indicators = ["transfer", "human", "person", "agent", "someone else"]
                has_transfer_intent = any(indicator in user_text.lower() for indicator in transfer_indicators)
                
                if has_transfer_intent:
                    # Add context for LLM to decide
                    turn_ctx.add_message(
                        role="system",
                        content="[TRANSFER_REQUEST_DETECTED]: Customer may want human transfer. Ask for confirmation before transferring."
                    )
                    logger.info(f"🔄 Possible transfer request detected - letting LLM decide")
                    return
                
                # Try RAG search for relevant information
                if RAG_AVAILABLE:
                    try:
                        timeout = getattr(config, 'rag_timeout_ms', 1500) / 1000.0
                        results = await asyncio.wait_for(
                            qdrant_rag.search(user_text, limit=3),
                            timeout=timeout
                        )
                        
                        if results and len(results) > 0:
                            best_result = results[0]
                            threshold = getattr(config, 'similarity_threshold', 0.25)
                            
                            if best_result["score"] >= threshold:
                                raw_content = best_result["text"]
                                context = self._clean_content_for_voice(raw_content)
                                turn_ctx.add_message(
                                    role="system",
                                    content=f"[QdrantRAG]: {context}"
                                )
                                
                                # Mark as RAG enhanced
                                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                                logger.info(f"⚡ RAG context injected (score: {best_result['score']:.3f}, {response_time:.1f}ms)")
                                
                                # Add to transcript with RAG flag
                                self.monitoring_agent.add_transcript(
                                    "customer", user_text, True, rag_enhanced=True,
                                    response_time_ms=response_time
                                )
                                return
                            else:
                                logger.info(f"⚠️ Low relevance score: {best_result['score']:.3f} < {threshold}")
                                
                    except asyncio.TimeoutError:
                        logger.debug("⚡ RAG timeout - continuing without context")
                    except Exception as e:
                        logger.debug(f"⚡ RAG error: {e}")
                
                # Add missing data context for LLM to handle naturally
                missing_context = self.monitoring_agent.get_missing_data_context()
                if "MISSING_DATA" in missing_context:
                    turn_ctx.add_message(
                        role="system",
                        content=f"[{missing_context}]"
                    )
                        
            finally:
                self.processing = False
                
        except Exception as e:
            logger.error(f"❌ on_user_turn_completed error: {e}")
            self.processing = False
    
    def _clean_content_for_voice(self, content: str) -> str:
        """Clean content for voice response"""
        try:
            # Remove formatting characters
            content = content.replace("Q: ", "").replace("A: ", "")
            content = content.replace("■", "").replace("●", "").replace("•", "")
            content = content.replace("- ", "").replace("* ", "")
            
            # Handle multi-line content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if lines:
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
    
    # ========================================================================
    # ENHANCED FUNCTION TOOLS
    # ========================================================================
    
    @function_tool()
    async def search_knowledge(self, query: str) -> str:
        """
        Search the knowledge base for real-time information.
        Use for ALL service information including pricing, business hours, policies.
        This replaces static responses with live data from your knowledge base.
        """
        try:
            logger.info(f"🔍 Searching knowledge base: {query}")
            
            if RAG_AVAILABLE:
                timeout = getattr(config, 'rag_timeout_ms', 1500) / 1000.0
                results = await asyncio.wait_for(
                    qdrant_rag.search(query, limit=3),
                    timeout=timeout
                )
                
                if results and len(results) > 0:
                    best_result = results[0]
                    content = self._clean_content_for_voice(best_result["text"])
                    logger.info(f"📊 Found result with score: {best_result['score']:.3f}")
                    
                    # Mark response as RAG enhanced
                    if self.monitoring_agent:
                        self.monitoring_agent.add_transcript(
                            "agent", f"Knowledge search: {content}", True,
                            rag_enhanced=True
                        )
                    
                    return content
                else:
                    logger.warning("⚠️ No relevant information found")
                    return "I don't have specific information about that in my current knowledge base. Let me transfer you to someone who can help with those details."
            
            return "I'm having trouble accessing the knowledge base right now. Let me transfer you to someone who can help."
            
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return "I'm having trouble accessing the information. Let me transfer you to someone who can help."

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """
        Transfer caller to human agent.
        
        The LLM should call this function when:
        - Customer explicitly requests human help
        - Customer expresses frustration with AI
        - Complex issues that need human expertise
        - Customer directly asks for transfer
        
        The LLM should confirm before transferring.
        """
        try:
            logger.info("🔄 HUMAN TRANSFER INITIATED")
            
            # Save enhanced conversation data
            if self.monitoring_agent:
                self.monitoring_agent.service_request.status = "transferred"
                await self.monitoring_agent.save_conversation_data()
            
            # Get job context and find SIP participant
            job_ctx = get_job_context()
            sip_participant = None
            
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                    sip_participant = participant
                    break
            
            if not sip_participant:
                return "I'm sorry, I couldn't find any active participants to transfer."
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=self.transfer_sip_address,
                play_dialtone=True,
            )
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=30.0
            )
            
            logger.info("✅ Transfer completed successfully")
            return "Connecting you to a human agent now."
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "Having trouble with transfer. Let me continue helping you."

# ============================================================================
# ENHANCED SESSION CREATION
# ============================================================================

async def create_enhanced_session() -> AgentSession:
    """Create enhanced session with male voice and optimized settings"""
    
    # Configure ElevenLabs TTS with male voice (Josh)
    tts_engine = elevenlabs.TTS(
        voice_id="TxGEqnHWrfWFTfGW9XjX",  # Josh - professional male voice
        model="eleven_turbo_v2_5",       # Fast stable model
        
        # Professional male voice settings
        voice_settings=elevenlabs.VoiceSettings(
            stability=0.6,              # Slightly more stable for professional sound
            similarity_boost=0.8,       # High similarity
            style=0.3,                  # Moderate style for clarity
            speed=1.0,                  # Normal speed
            use_speaker_boost=True      # Enable for better quality
        ),
        
        # Conservative settings to avoid API errors
        inactivity_timeout=300,
        enable_ssml_parsing=False,
    )
    logger.info("🎙️ ElevenLabs TTS configured with Josh (male voice)")
    
    session = AgentSession(
        # Fast STT - using proven configuration
        stt=deepgram.STT(
            model="nova-2-general",
            language="en",
        ),
        
        # Fast LLM
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
        ),
        
        # ElevenLabs TTS with male voice
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

# ============================================================================
# ENHANCED ENTRYPOINT
# ============================================================================

async def entrypoint(ctx: JobContext):
    """Enhanced entrypoint with full RAG integration"""
    logger.info("⚡ === ENHANCED ROADSIDE AGENT WITH FULL RAG ===")
    
    # Connect immediately
    await ctx.connect()
    logger.info("✅ CONNECTED")
    
    # Initialize RAG in background
    if RAG_AVAILABLE:
        asyncio.create_task(qdrant_rag.initialize())
        logger.info("🧠 RAG: Starting background initialization...")
    
    # Create enhanced session and agent
    session = await create_enhanced_session()
    agent = EnhancedRoadsideAgent()
    agent.set_room_name(ctx.room.name)
    
    # Start session
    logger.info("⚡ STARTING ENHANCED SESSION...")
    await session.start(room=ctx.room, agent=agent)
    
    # System ready
    logger.info("⚡ ENHANCED SYSTEM READY!")
    logger.info("🎯 TARGET LATENCY: <2 seconds")
    logger.info("🔥 TTS: ✅ ElevenLabs Josh (male voice)")
    logger.info("🧠 RAG: ✅ Full knowledge base integration")
    logger.info("📊 MONITORING: ✅ Complete data collection")
    logger.info("🔄 TRANSFER: ✅ Available")
    logger.info(f"📞 Room: {ctx.room.name}")
    
    # Professional greeting
    await session.generate_reply(
        instructions="Give a professional greeting: 'Roadside assistance, this is Mark. How can I help with your vehicle today?' Keep it brief and professional."
    )
    
    logger.info("⚡ GREETING SENT - ENHANCED SYSTEM ACTIVE")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        logger.info("⚡ STARTING ENHANCED ROADSIDE SYSTEM")
        logger.info("🎯 TARGET: Complete data collection + full RAG")
        logger.info("🔥 TECH: Male voice + dynamic knowledge base")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except KeyboardInterrupt:
        logger.info("🛑 SYSTEM SHUTDOWN")
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)
    finally:
        logger.info("👋 ENHANCED SYSTEM STOPPED")