# services/monitoring_service.py
"""
Monitoring and data extraction service for LiveKit 1.1 multi-agent system
"""
import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from models.session_data import RoadsideSessionData, ServiceType, CallStage

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Comprehensive monitoring and data extraction service
    Handles real-time extraction, analytics, and persistence
    """
    
    def __init__(
        self,
        save_transcripts: bool = True,
        save_extracted_data: bool = True,
        output_dir: str = "call_recordings",
        real_time_analysis: bool = True
    ):
        self.save_transcripts = save_transcripts
        self.save_extracted_data = save_extracted_data
        self.output_dir = Path(output_dir)
        self.real_time_analysis = real_time_analysis
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Extraction patterns
        self.extraction_patterns = self._build_extraction_patterns()
        
        # Metrics tracking
        self.session_metrics: Dict[str, Any] = {}
    
    def _build_extraction_patterns(self) -> Dict[str, Any]:
        """Build regex patterns for data extraction"""
        return {
            'name': [
                re.compile(r'(?:my name is|i\'m|this is|call me)\s+([a-zA-Z\s]+)', re.IGNORECASE),
                re.compile(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', re.IGNORECASE),
            ],
            'phone': [
                re.compile(r'(\d{3})\s*[-.]?\s*(\d{3})\s*[-.]?\s*(\d{4})'),
                re.compile(r'\((\d{3})\)\s*(\d{3})\s*[-.]?\s*(\d{4})'),
                re.compile(r'(?:seven|eight|one|two|three|four|five|six|nine|zero)', re.IGNORECASE),
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
    
    async def extract_information(
        self, 
        text: str, 
        session_data: RoadsideSessionData,
        agent_type: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Extract structured information from conversation text
        Returns extracted data and updates session_data in place
        """
        extracted = {}
        
        try:
            # Extract customer information
            customer_extracted = await self._extract_customer_info(text, session_data)
            if customer_extracted:
                extracted.update(customer_extracted)
            
            # Extract location information
            location_extracted = await self._extract_location_info(text, session_data)
            if location_extracted:
                extracted.update(location_extracted)
            
            # Extract vehicle information
            vehicle_extracted = await self._extract_vehicle_info(text, session_data)
            if vehicle_extracted:
                extracted.update(vehicle_extracted)
            
            # Extract service information
            service_extracted = await self._extract_service_info(text, session_data)
            if service_extracted:
                extracted.update(service_extracted)
            
            # Log extraction if data found
            if extracted:
                logger.info(f"📊 Extracted data: {extracted}")
                
                # Add to session interaction log
                session_data.add_interaction(
                    agent_type=agent_type,
                    speaker="extraction_service",
                    message=f"Extracted: {', '.join(extracted.keys())}",
                    extracted_data=extracted
                )
            
            return extracted
            
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {}
    
    async def _extract_customer_info(self, text: str, session_data: RoadsideSessionData) -> Dict[str, Any]:
        """Extract customer information"""
        extracted = {}
        
        # Extract name
        if not (session_data.customer.first_name and session_data.customer.last_name):
            for pattern in self.extraction_patterns['name']:
                match = pattern.search(text)
                if match:
                    if len(match.groups()) == 2:
                        first, last = match.group(1).strip().title(), match.group(2).strip().title()
                        if len(first) > 1 and len(last) > 1 and not any(char.isdigit() for char in first + last):
                            session_data.customer.first_name = first
                            session_data.customer.last_name = last
                            extracted['customer_name'] = f"{first} {last}"
                            logger.info(f"⚡ Extracted name: {first} {last}")
                            break
                    elif len(match.groups()) == 1:
                        name_parts = match.group(1).strip().title().split()
                        if len(name_parts) >= 2:
                            session_data.customer.first_name = name_parts[0]
                            session_data.customer.last_name = " ".join(name_parts[1:])
                            extracted['customer_name'] = " ".join(name_parts)
                            logger.info(f"⚡ Extracted name: {' '.join(name_parts)}")
                            break
        
        # Extract phone
        if not session_data.customer.phone:
            phone_text = self._normalize_spoken_numbers(text)
            for pattern in self.extraction_patterns['phone']:
                match = pattern.search(phone_text)
                if match:
                    phone_digits = ''.join(filter(str.isdigit, ''.join(match.groups())))
                    if len(phone_digits) >= 10:
                        phone = phone_digits[-10:]  # Last 10 digits
                        formatted_phone = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
                        session_data.customer.phone = formatted_phone
                        extracted['customer_phone'] = formatted_phone
                        logger.info(f"⚡ Extracted phone: {formatted_phone}")
                        break
        
        return extracted
    
    async def _extract_location_info(self, text: str, session_data: RoadsideSessionData) -> Dict[str, Any]:
        """Extract location information"""
        extracted = {}
        
        # Extract street address
        if not session_data.location.street_address:
            for pattern in self.extraction_patterns['address']:
                match = pattern.search(text)
                if match:
                    if len(match.groups()) == 2:
                        address = f"{match.group(1)} {match.group(2)}"
                    else:
                        address = match.group(1)
                    
                    address = address.strip().title()
                    if len(address) > 5:
                        session_data.location.street_address = address
                        extracted['street_address'] = address
                        logger.info(f"⚡ Extracted address: {address}")
                        break
        
        # Extract city and state
        for pattern in self.extraction_patterns['city_state']:
            match = pattern.search(text)
            if match:
                city = match.group(1).strip().title()
                state_raw = match.group(2).strip()
                
                if len(state_raw) == 2:
                    state = state_raw.upper()
                else:
                    state = state_raw.title()
                
                if not session_data.location.city and len(city) > 2:
                    session_data.location.city = city
                    extracted['city'] = city
                    logger.info(f"⚡ Extracted city: {city}")
                
                if not session_data.location.state and len(state) <= 20:
                    session_data.location.state = state
                    extracted['state'] = state
                    logger.info(f"⚡ Extracted state: {state}")
                break
        
        # Extract ZIP
        if not session_data.location.zip_code:
            for pattern in self.extraction_patterns['zip_code']:
                match = pattern.search(text)
                if match:
                    zip_code = match.group(1)
                    if zip_code.isdigit() and len(zip_code) == 5:
                        session_data.location.zip_code = zip_code
                        extracted['zip_code'] = zip_code
                        logger.info(f"⚡ Extracted ZIP: {zip_code}")
                        break
        
        return extracted
    
    async def _extract_vehicle_info(self, text: str, session_data: RoadsideSessionData) -> Dict[str, Any]:
        """Extract vehicle information"""
        extracted = {}
        
        if not (session_data.vehicle.year and session_data.vehicle.make):
            for pattern in self.extraction_patterns['vehicle']:
                match = pattern.search(text)
                if match:
                    groups = match.groups()
                    year, make, model = None, None, None
                    
                    # Find year
                    for group in groups:
                        if group.isdigit() and 1990 <= int(group) <= 2030:
                            year = group
                            break
                    
                    # Extract make and model
                    remaining = [g for g in groups if g != year]
                    if len(remaining) >= 2:
                        make, model = remaining[0].title(), remaining[1].title()
                    elif len(remaining) == 1:
                        make = remaining[0].title()
                    
                    # Store extracted data
                    if year:
                        session_data.vehicle.year = year
                        extracted['vehicle_year'] = year
                    if make:
                        session_data.vehicle.make = make
                        extracted['vehicle_make'] = make
                    if model:
                        session_data.vehicle.model = model
                        extracted['vehicle_model'] = model
                    
                    if year and make:
                        vehicle_info = f"{year} {make}"
                        if model:
                            vehicle_info += f" {model}"
                        logger.info(f"⚡ Extracted vehicle: {vehicle_info}")
                        break
        
        return extracted
    
    async def _extract_service_info(self, text: str, session_data: RoadsideSessionData) -> Dict[str, Any]:
        """Extract service information"""
        extracted = {}
        
        if session_data.service.service_type == ServiceType.UNKNOWN:
            service_matches = []
            
            for pattern, service_type in self.extraction_patterns['service']:
                if pattern.search(text):
                    service_matches.append(service_type)
            
            if len(service_matches) == 1:
                session_data.service.service_type = service_matches[0]
                extracted['service_type'] = service_matches[0].value
                logger.info(f"⚡ Extracted service: {service_matches[0].value}")
            elif len(service_matches) > 1:
                extracted['possible_services'] = [s.value for s in service_matches]
                logger.info(f"⚡ Multiple services detected: {[s.value for s in service_matches]}")
        
        return extracted
    
    def _normalize_spoken_numbers(self, text: str) -> str:
        """Convert spoken numbers to digits"""
        phone_text = text.lower()
        spoken_to_digit = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'oh': '0'
        }
        for word, digit in spoken_to_digit.items():
            phone_text = phone_text.replace(word, digit)
        return phone_text
    
    async def update_session_metrics(
        self, 
        session_data: RoadsideSessionData,
        agent_type: str,
        response_time_ms: Optional[float] = None
    ):
        """Update metrics for the session"""
        session_id = session_data.room_name or "unknown"
        
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = {
                "start_time": session_data.conversation_start,
                "agent_transitions": [],
                "response_times": [],
                "data_completeness_over_time": [],
                "extraction_events": []
            }
        
        metrics = self.session_metrics[session_id]
        
        # Track agent transitions
        if agent_type not in [t["agent"] for t in metrics["agent_transitions"]]:
            metrics["agent_transitions"].append({
                "agent": agent_type,
                "timestamp": datetime.now(),
                "stage": session_data.current_stage.value
            })
        
        # Track response times
        if response_time_ms:
            metrics["response_times"].append(response_time_ms)
        
        # Track data completeness
        completeness = session_data.get_completeness_status()
        metrics["data_completeness_over_time"].append({
            "timestamp": datetime.now(),
            "completeness": completeness,
            "complete_percentage": sum(completeness.values()) / len(completeness) * 100
        })
    
    async def save_session_summary(self, session_data: RoadsideSessionData) -> str:
        """Save comprehensive session summary"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_id = session_data.room_name or "unknown"
            filename = f"session_{session_id}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Build comprehensive summary
            summary = {
                "session_metadata": {
                    "session_id": session_id,
                    "start_time": session_data.conversation_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": session_data.call_duration_seconds,
                    "final_stage": session_data.current_stage.value
                },
                "session_data": session_data.to_summary_dict(),
                "conversation_log": [
                    {
                        "timestamp": interaction.timestamp.isoformat(),
                        "agent_type": interaction.agent_type,
                        "speaker": interaction.speaker,
                        "message": interaction.message,
                        "extracted_data": interaction.extracted_data,
                        "rag_enhanced": interaction.rag_enhanced,
                        "response_time_ms": interaction.response_time_ms
                    }
                    for interaction in session_data.interactions
                ],
                "metrics": self.session_metrics.get(session_id, {}),
                "analytics": {
                    "total_interactions": len(session_data.interactions),
                    "rag_enhanced_responses": sum(1 for i in session_data.interactions if i.rag_enhanced),
                    "extraction_events": len([i for i in session_data.interactions if i.extracted_data]),
                    "data_completeness_final": session_data.get_completeness_status(),
                    "conversation_quality_score": self._calculate_quality_score(session_data)
                }
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"💾 Session summary saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Failed to save session summary: {e}")
            return ""
    
    def _calculate_quality_score(self, session_data: RoadsideSessionData) -> float:
        """Calculate conversation quality score (0-100)"""
        try:
            completeness = session_data.get_completeness_status()
            completeness_score = sum(completeness.values()) / len(completeness) * 100
            
            # Factor in interaction efficiency
            interaction_count = len(session_data.interactions)
            efficiency_score = min(100, max(0, 100 - (interaction_count - 10) * 2))  # Penalty for long conversations
            
            # Factor in extraction success
            extraction_count = len([i for i in session_data.interactions if i.extracted_data])
            extraction_score = min(100, extraction_count * 10)
            
            # Weighted average
            quality_score = (
                completeness_score * 0.6 +
                efficiency_score * 0.2 +
                extraction_score * 0.2
            )
            
            return round(quality_score, 2)
            
        except Exception:
            return 0.0


# Global instance for dependency injection
monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> Optional[MonitoringService]:
    """Get the global monitoring service instance"""
    return monitoring_service


def initialize_monitoring_service(**kwargs) -> MonitoringService:
    """Initialize global monitoring service"""
    global monitoring_service
    monitoring_service = MonitoringService(**kwargs)
    return monitoring_service