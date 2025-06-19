# models/session_data.py
"""
Unified session data models for LiveKit 1.1 multi-agent roadside assistance
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ServiceType(Enum):
    """Service types for roadside assistance"""
    TOWING = "towing"
    JUMP_START = "jump_start"
    TIRE_CHANGE = "tire_change"
    TIRE_REPLACEMENT = "tire_replacement"
    WINCH_OUT = "winch_out"
    LOCKOUT = "lockout"
    FUEL_DELIVERY = "fuel_delivery"
    UNKNOWN = "unknown"


class CallStage(Enum):
    """Current stage of the conversation"""
    INITIAL = "initial"
    CUSTOMER_INTAKE = "customer_intake"
    VEHICLE_DIAGNOSTICS = "vehicle_diagnostics"
    SERVICE_DISPATCH = "service_dispatch"
    CONFIRMATION = "confirmation"
    TRANSFERRED = "transferred"
    COMPLETED = "completed"


@dataclass
class CustomerInfo:
    """Customer information structure"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    insurance_info: Optional[str] = None
    customer_id: Optional[str] = None
    
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
    
    @property
    def is_complete(self) -> bool:
        """Check if customer info is complete"""
        return bool(self.first_name and self.last_name and self.phone)


@dataclass
class VehicleInfo:
    """Vehicle information structure"""
    year: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    neutral_functional: Optional[bool] = None
    vehicle_type: Optional[str] = None  # SUV, sedan, truck, etc.
    vin: Optional[str] = None
    
    @property
    def description(self) -> str:
        """Get vehicle description"""
        parts = []
        if self.year:
            parts.append(self.year)
        if self.make:
            parts.append(self.make)
        if self.model:
            parts.append(self.model)
        if self.color:
            parts.append(f"({self.color})")
        return " ".join(parts) if parts else "Vehicle details not provided"
    
    @property
    def is_complete(self) -> bool:
        """Check if vehicle info is complete"""
        return bool(self.year and self.make and self.model)


@dataclass
class LocationInfo:
    """Location information structure"""
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    landmarks: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    gps_accuracy: Optional[float] = None
    
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
    
    @property
    def is_complete(self) -> bool:
        """Check if location info is complete"""
        return bool(self.street_address and self.city and self.state)


@dataclass
class ServiceRequest:
    """Service request information"""
    service_type: ServiceType = ServiceType.UNKNOWN
    estimated_cost: Optional[str] = None
    job_number: Optional[str] = None
    priority: str = "normal"  # normal, urgent, emergency
    special_requirements: List[str] = field(default_factory=list)
    eta_minutes: Optional[int] = None
    technician_id: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if service request is complete"""
        return self.service_type != ServiceType.UNKNOWN


@dataclass
class InteractionLog:
    """Single interaction in the conversation"""
    timestamp: datetime
    agent_type: str
    speaker: str  # "customer" or "agent"
    message: str
    extracted_data: Optional[Dict[str, Any]] = None
    rag_enhanced: bool = False
    response_time_ms: Optional[float] = None


@dataclass
class RoadsideSessionData:
    """
    Main session data structure shared across all agents
    This is the core state that gets passed between agents in LiveKit 1.1
    """
    # Core information
    customer: CustomerInfo = field(default_factory=CustomerInfo)
    vehicle: VehicleInfo = field(default_factory=VehicleInfo)
    location: LocationInfo = field(default_factory=LocationInfo)
    service: ServiceRequest = field(default_factory=ServiceRequest)
    
    # Conversation state
    current_stage: CallStage = CallStage.INITIAL
    conversation_start: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    # Interaction history
    interactions: List[InteractionLog] = field(default_factory=list)
    
    # RAG and monitoring context
    rag_context: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    
    # Call metadata
    room_name: Optional[str] = None
    call_duration_seconds: Optional[float] = None
    transfer_requested: bool = False
    transfer_reason: Optional[str] = None
    
    def add_interaction(self, agent_type: str, speaker: str, message: str, 
                       extracted_data: Optional[Dict] = None, 
                       rag_enhanced: bool = False,
                       response_time_ms: Optional[float] = None):
        """Add an interaction to the log"""
        interaction = InteractionLog(
            timestamp=datetime.now(),
            agent_type=agent_type,
            speaker=speaker,
            message=message,
            extracted_data=extracted_data,
            rag_enhanced=rag_enhanced,
            response_time_ms=response_time_ms
        )
        self.interactions.append(interaction)
        self.last_update = datetime.now()
    
    def update_stage(self, new_stage: CallStage):
        """Update conversation stage"""
        self.current_stage = new_stage
        self.last_update = datetime.now()
    
    def get_completeness_status(self) -> Dict[str, bool]:
        """Get overall data completeness"""
        return {
            "customer_complete": self.customer.is_complete,
            "vehicle_complete": self.vehicle.is_complete,
            "location_complete": self.location.is_complete,
            "service_complete": self.service.is_complete,
            "ready_for_dispatch": (
                self.customer.is_complete and 
                self.vehicle.is_complete and 
                self.location.is_complete and 
                self.service.is_complete
            )
        }
    
    def get_missing_info(self) -> List[str]:
        """Get list of missing information"""
        missing = []
        
        if not self.customer.first_name or not self.customer.last_name:
            missing.append("customer full name")
        if not self.customer.phone:
            missing.append("phone number")
        
        if not self.location.street_address:
            missing.append("street address")
        if not self.location.city or not self.location.state:
            missing.append("city and state")
        
        if not self.vehicle.year or not self.vehicle.make or not self.vehicle.model:
            missing.append("vehicle details (year, make, model)")
        
        if self.service.service_type == ServiceType.UNKNOWN:
            missing.append("type of service needed")
        
        return missing
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for monitoring/export"""
        return {
            "session_id": self.room_name,
            "start_time": self.conversation_start.isoformat(),
            "duration_seconds": self.call_duration_seconds,
            "current_stage": self.current_stage.value,
            "customer": {
                "name": self.customer.full_name,
                "phone": self.customer.phone,
                "complete": self.customer.is_complete
            },
            "vehicle": {
                "description": self.vehicle.description,
                "complete": self.vehicle.is_complete
            },
            "location": {
                "address": self.location.full_address,
                "complete": self.location.is_complete
            },
            "service": {
                "type": self.service.service_type.value,
                "cost": self.service.estimated_cost,
                "job_number": self.service.job_number,
                "complete": self.service.is_complete
            },
            "completeness": self.get_completeness_status(),
            "interaction_count": len(self.interactions),
            "rag_enhanced_count": sum(1 for i in self.interactions if i.rag_enhanced),
            "transferred": self.transfer_requested
        }