# tools/service_tools.py
"""
Service management tools for LiveKit 1.1 multi-agent system
"""
import logging
from typing import Optional
from livekit.agents import function_tool, RunContext

from models.session_data import RoadsideSessionData, ServiceType
from services.monitoring_service import get_monitoring_service
from services.rag_service import get_rag_service

logger = logging.getLogger(__name__)


@function_tool
async def identify_service_type(
    context: RunContext[RoadsideSessionData],
    service_type: str,
    description: Optional[str] = None
) -> str:
    """
    Identify the type of roadside service needed.
    
    Args:
        service_type: Type of service (towing, jump_start, tire_change, etc.)
        description: Additional description of the problem
    
    Returns:
        Confirmation and relevant service information
    """
    try:
        session_data = context.userdata
        service = session_data.service
        
        # Map service type string to enum
        service_mapping = {
            "towing": ServiceType.TOWING,
            "tow": ServiceType.TOWING,
            "jump_start": ServiceType.JUMP_START,
            "jump": ServiceType.JUMP_START,
            "battery": ServiceType.JUMP_START,
            "tire_change": ServiceType.TIRE_CHANGE,
            "tire": ServiceType.TIRE_CHANGE,
            "flat_tire": ServiceType.TIRE_CHANGE,
            "tire_replacement": ServiceType.TIRE_REPLACEMENT,
            "winch_out": ServiceType.WINCH_OUT,
            "winch": ServiceType.WINCH_OUT,
            "stuck": ServiceType.WINCH_OUT,
            "lockout": ServiceType.LOCKOUT,
            "locked_out": ServiceType.LOCKOUT,
            "keys": ServiceType.LOCKOUT,
            "fuel_delivery": ServiceType.FUEL_DELIVERY,
            "fuel": ServiceType.FUEL_DELIVERY,
            "gas": ServiceType.FUEL_DELIVERY
        }
        
        service_key = service_type.lower().replace(" ", "_")
        if service_key in service_mapping:
            service.service_type = service_mapping[service_key]
            
            if description:
                service.special_requirements.append(description)
            
            logger.info(f"🔧 Identified service: {service.service_type.value}")
            
            session_data.add_interaction(
                agent_type="service_tools",
                speaker="system",
                message=f"Service identified: {service.service_type.value}",
                extracted_data={"service_type": service_type, "description": description}
            )
            
            # Get relevant context from RAG
            rag_service = get_rag_service()
            context_info = ""
            if rag_service:
                context_result = await rag_service.get_relevant_context(service.service_type.value, description)
                if context_result:
                    context_info = f" {context_result}"
            
            # Return service-specific response
            if service.service_type == ServiceType.TOWING:
                return f"I understand you need towing service.{context_info} Is your vehicle able to be put in neutral gear?"
            elif service.service_type == ServiceType.JUMP_START:
                return f"Got it, you need a jump start for your battery.{context_info}"
            elif service.service_type == ServiceType.TIRE_CHANGE:
                return f"I see you need tire service.{context_info} Do you have a spare tire?"
            elif service.service_type == ServiceType.LOCKOUT:
                return f"You're locked out of your vehicle.{context_info} Are the keys visible inside?"
            elif service.service_type == ServiceType.WINCH_OUT:
                return f"You need winch-out service to get unstuck.{context_info} How far off the road are you?"
            elif service.service_type == ServiceType.FUEL_DELIVERY:
                return f"I'll arrange fuel delivery for you.{context_info} What type of fuel does your vehicle use?"
            else:
                return f"I've recorded that you need {service.service_type.value} service.{context_info}"
        else:
            return f"I'm not sure about '{service_type}'. Could you describe what's wrong with your vehicle? For example: won't start, flat tire, locked out, stuck, or need a tow."
            
    except Exception as e:
        logger.error(f"❌ Error identifying service type: {e}")
        return "What type of service do you need? For example: towing, jump start, tire change, or lockout assistance."


@function_tool
async def get_service_pricing(
    context: RunContext[RoadsideSessionData],
    service_type: Optional[str] = None
) -> str:
    """
    Get pricing information for the requested service.
    
    Args:
        service_type: Specific service type to get pricing for (optional)
    
    Returns:
        Pricing information for the service
    """
    try:
        session_data = context.userdata
        service = session_data.service
        
        # Use provided service type or current session service type
        target_service = service_type or service.service_type.value
        
        # Try to get pricing from RAG
        rag_service = get_rag_service()
        if rag_service:
            pricing_info = await rag_service.get_pricing_info(target_service)
            if pricing_info:
                session_data.add_interaction(
                    agent_type="service_tools",
                    speaker="system",
                    message=f"Provided pricing for {target_service}",
                    rag_enhanced=True
                )
                return pricing_info
        
        # Fallback pricing information
        pricing_map = {
            "towing": "Towing service starts at $169 plus $8-10 per mile, depending on distance.",
            "jump_start": "Jump start service is typically $150-200, depending on your location and time of day.",
            "tire_change": "Tire change service is usually $75-100 if you have a spare tire.",
            "tire_replacement": "Tire replacement varies based on tire type and size, typically $150-300 plus the cost of the tire.",
            "lockout": "Lockout service is generally $75-150, depending on vehicle type and complexity.",
            "winch_out": "Winch-out service starts at $200, with additional charges based on difficulty and equipment needed.",
            "fuel_delivery": "Fuel delivery is typically $75-100 plus the cost of fuel."
        }
        
        service_key = target_service.lower().replace(" ", "_")
        if service_key in pricing_map:
            pricing = pricing_map[service_key]
            service.estimated_cost = pricing
            
            logger.info(f"💰 Provided pricing for {target_service}")
            
            session_data.add_interaction(
                agent_type="service_tools",
                speaker="system",
                message=f"Provided pricing for {target_service}"
            )
            
            return pricing
        else:
            return "I'll have our dispatcher provide you with exact pricing when we confirm your service details."
            
    except Exception as e:
        logger.error(f"❌ Error getting service pricing: {e}")
        return "Let me check on pricing for your service and get back to you."


@function_tool
async def add_special_requirements(
    context: RunContext[RoadsideSessionData],
    requirements: str
) -> str:
    """
    Add special requirements or notes for the service.
    
    Args:
        requirements: Special requirements or additional information
    
    Returns:
        Confirmation of added requirements
    """
    try:
        session_data = context.userdata
        session_data.service.special_requirements.append(requirements.strip())
        
        logger.info(f"📝 Added special requirements: {requirements}")
        
        session_data.add_interaction(
            agent_type="service_tools",
            speaker="system",
            message=f"Added requirements: {requirements}",
            extracted_data={"special_requirements": requirements}
        )
        
        return f"I've noted: {requirements}. This will be passed along to our technician."
        
    except Exception as e:
        logger.error(f"❌ Error adding special requirements: {e}")
        return "What special requirements or additional information should I note for the technician?"


@function_tool
async def set_service_priority(
    context: RunContext[RoadsideSessionData],
    priority: str,
    reason: Optional[str] = None
) -> str:
    """
    Set the priority level for the service request.
    
    Args:
        priority: Priority level (normal, urgent, emergency)
        reason: Reason for the priority level
    
    Returns:
        Confirmation of priority setting
    """
    try:
        session_data = context.userdata
        service = session_data.service
        
        priority_lower = priority.lower()
        if priority_lower in ["normal", "urgent", "emergency"]:
            service.priority = priority_lower
            
            if reason:
                service.special_requirements.append(f"Priority: {priority} - {reason}")
            
            logger.info(f"⚡ Set service priority: {priority}")
            
            session_data.add_interaction(
                agent_type="service_tools",
                speaker="system",
                message=f"Set priority: {priority}",
                extracted_data={"priority": priority, "reason": reason}
            )
            
            if priority_lower == "emergency":
                return f"I've marked this as an emergency service. We'll prioritize getting someone to you as quickly as possible."
            elif priority_lower == "urgent":
                return f"This has been marked as urgent. We'll expedite your service request."
            else:
                return f"Your service request is set to normal priority."
        else:
            return "Priority should be normal, urgent, or emergency. What priority level is appropriate?"
            
    except Exception as e:
        logger.error(f"❌ Error setting service priority: {e}")
        return "What priority level should I set for your service request?"


@function_tool
async def estimate_arrival_time(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Provide estimated arrival time for the service technician.
    
    Returns:
        Estimated arrival time information
    """
    try:
        session_data = context.userdata
        service = session_data.service
        
        # Get ETA from RAG or provide standard estimates
        rag_service = get_rag_service()
        if rag_service:
            eta_info = await rag_service.search_knowledge("arrival time ETA technician", limit=1)
            if eta_info and len(eta_info) > 0:
                response = rag_service.clean_content_for_voice(eta_info[0]["text"])
                if response:
                    return response
        
        # Fallback ETA based on priority and service type
        if service.priority == "emergency":
            eta_minutes = 30
            eta_text = "30-45 minutes"
        elif service.priority == "urgent":
            eta_minutes = 60
            eta_text = "45-60 minutes"
        else:
            eta_minutes = 90
            eta_text = "60-90 minutes"
        
        service.eta_minutes = eta_minutes
        
        logger.info(f"⏰ Estimated arrival: {eta_text}")
        
        session_data.add_interaction(
            agent_type="service_tools",
            speaker="system",
            message=f"Provided ETA: {eta_text}"
        )
        
        return f"Based on your location and current demand, our estimated arrival time is {eta_text}. You'll receive a call when the technician is on the way."
        
    except Exception as e:
        logger.error(f"❌ Error estimating arrival time: {e}")
        return "I'll have dispatch provide you with an estimated arrival time once we confirm your service details."


@function_tool
async def generate_job_number(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Generate a job number for the service request.
    
    Returns:
        Job number confirmation
    """
    try:
        import time
        import random
        
        session_data = context.userdata
        
        # Generate job number: RS + timestamp + random digits
        timestamp = int(time.time()) % 100000  # Last 5 digits of timestamp
        random_num = random.randint(100, 999)
        job_number = f"RS{timestamp}{random_num}"
        
        session_data.service.job_number = job_number
        
        logger.info(f"🎫 Generated job number: {job_number}")
        
        session_data.add_interaction(
            agent_type="service_tools",
            speaker="system",
            message=f"Generated job number: {job_number}"
        )
        
        return f"Your service request number is {job_number}. Please keep this for your records."
        
    except Exception as e:
        logger.error(f"❌ Error generating job number: {e}")
        return "I'll provide you with a service request number once we finalize your details."


@function_tool
async def verify_service_details(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Verify and summarize all service details.
    
    Returns:
        Complete summary of service request for verification
    """
    try:
        session_data = context.userdata
        service = session_data.service
        customer = session_data.customer
        vehicle = session_data.vehicle
        location = session_data.location
        
        summary_parts = []
        
        # Service type
        if service.service_type != ServiceType.UNKNOWN:
            summary_parts.append(f"Service: {service.service_type.value}")
        
        # Customer
        if customer.full_name and customer.phone:
            summary_parts.append(f"Customer: {customer.full_name} at {customer.phone}")
        
        # Vehicle
        if vehicle.is_complete:
            summary_parts.append(f"Vehicle: {vehicle.description}")
        
        # Location
        if location.is_complete:
            summary_parts.append(f"Location: {location.full_address}")
        
        # Special requirements
        if service.special_requirements:
            summary_parts.append(f"Notes: {'; '.join(service.special_requirements)}")
        
        # Pricing
        if service.estimated_cost:
            summary_parts.append(f"Estimated cost: {service.estimated_cost}")
        
        # Job number
        if service.job_number:
            summary_parts.append(f"Job number: {service.job_number}")
        
        if summary_parts:
            summary = "Let me verify your service request: " + ". ".join(summary_parts)
            
            # Check completeness
            completeness = session_data.get_completeness_status()
            if completeness["ready_for_dispatch"]:
                summary += ". Is everything correct and ready to dispatch?"
            else:
                missing = session_data.get_missing_info()
                if missing:
                    summary += f". I still need: {', '.join(missing[:3])}."
            
            return summary
        else:
            return "I don't have complete service information yet. What type of service do you need?"
            
    except Exception as e:
        logger.error(f"❌ Error verifying service details: {e}")
        return "Let me make sure I have all your service details correct..."


@function_tool
async def get_service_status(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Get the current status of the service request.
    
    Returns:
        Status message about service request completeness
    """
    try:
        session_data = context.userdata
        service = session_data.service
        completeness = session_data.get_completeness_status()
        
        if completeness["ready_for_dispatch"]:
            return f"Service request is complete and ready for dispatch. Service type: {service.service_type.value}."
        elif service.service_type != ServiceType.UNKNOWN:
            missing = session_data.get_missing_info()
            return f"Service type identified: {service.service_type.value}. Still need: {', '.join(missing)}."
        else:
            return "Service type not yet identified. What type of roadside assistance do you need?"
            
    except Exception as e:
        logger.error(f"❌ Error getting service status: {e}")
        return "Let me check the status of your service request..."