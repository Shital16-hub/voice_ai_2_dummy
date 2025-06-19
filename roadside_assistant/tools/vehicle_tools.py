# tools/vehicle_tools.py
"""
Vehicle information management tools for LiveKit 1.1 multi-agent system
"""
import logging
from typing import Optional
from livekit.agents import function_tool, RunContext

from models.session_data import RoadsideSessionData
from services.monitoring_service import get_monitoring_service

logger = logging.getLogger(__name__)


@function_tool
async def collect_vehicle_info(
    context: RunContext[RoadsideSessionData],
    year: str,
    make: str,
    model: str,
    color: Optional[str] = None
) -> str:
    """
    Collect complete vehicle information.
    
    Args:
        year: Vehicle year (e.g., "2020")
        make: Vehicle manufacturer (e.g., "Honda")
        model: Vehicle model (e.g., "Civic")
        color: Vehicle color (optional)
    
    Returns:
        Confirmation message about collected vehicle information
    """
    try:
        session_data = context.userdata
        vehicle = session_data.vehicle
        
        # Validate and store vehicle information
        if year.isdigit() and 1990 <= int(year) <= 2030:
            vehicle.year = year
        else:
            return f"The year '{year}' doesn't seem right. Could you provide the vehicle year between 1990 and 2030?"
        
        vehicle.make = make.strip().title()
        vehicle.model = model.strip().title()
        
        if color:
            vehicle.color = color.strip().title()
        
        logger.info(f"🚗 Collected vehicle info: {vehicle.description}")
        
        # Add to interaction log
        session_data.add_interaction(
            agent_type="vehicle_tools",
            speaker="system",
            message=f"Collected vehicle: {vehicle.description}",
            extracted_data={
                "year": year,
                "make": make,
                "model": model,
                "color": color
            }
        )
        
        # Update monitoring
        monitoring_service = get_monitoring_service()
        if monitoring_service:
            await monitoring_service.update_session_metrics(session_data, "vehicle_tools")
        
        # Return confirmation
        if color:
            return f"Got it! Your {color} {year} {make} {model}. Is that correct?"
        else:
            return f"Perfect! I have your {year} {make} {model} recorded."
            
    except Exception as e:
        logger.error(f"❌ Error collecting vehicle info: {e}")
        return "I had trouble recording your vehicle information. Could you tell me the year, make, and model again?"


@function_tool
async def collect_vehicle_year(
    context: RunContext[RoadsideSessionData],
    year: str
) -> str:
    """
    Collect vehicle year.
    
    Args:
        year: Vehicle year
    
    Returns:
        Confirmation message
    """
    try:
        session_data = context.userdata
        
        if year.isdigit() and 1990 <= int(year) <= 2030:
            session_data.vehicle.year = year
            logger.info(f"🗓️ Collected vehicle year: {year}")
            
            session_data.add_interaction(
                agent_type="vehicle_tools",
                speaker="system",
                message=f"Collected vehicle year: {year}",
                extracted_data={"year": year}
            )
            
            return f"Thank you, {year}. What's the make and model?"
        else:
            return f"The year '{year}' doesn't seem right. What year was your vehicle made?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting vehicle year: {e}")
        return "Could you tell me what year your vehicle is?"


@function_tool
async def collect_vehicle_make_model(
    context: RunContext[RoadsideSessionData],
    make: str,
    model: str
) -> str:
    """
    Collect vehicle make and model.
    
    Args:
        make: Vehicle manufacturer
        model: Vehicle model
    
    Returns:
        Confirmation message
    """
    try:
        session_data = context.userdata
        vehicle = session_data.vehicle
        
        vehicle.make = make.strip().title()
        vehicle.model = model.strip().title()
        
        logger.info(f"🚗 Collected vehicle make/model: {make} {model}")
        
        session_data.add_interaction(
            agent_type="vehicle_tools",
            speaker="system",
            message=f"Collected vehicle: {make} {model}",
            extracted_data={"make": make, "model": model}
        )
        
        # Check if we have complete basic info
        if vehicle.year:
            return f"Perfect! So you have a {vehicle.year} {make} {model}."
        else:
            return f"Got it, {make} {model}. What year is it?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting vehicle make/model: {e}")
        return "Could you tell me the make and model of your vehicle?"


@function_tool
async def collect_vehicle_color(
    context: RunContext[RoadsideSessionData],
    color: str
) -> str:
    """
    Collect vehicle color.
    
    Args:
        color: Vehicle color
    
    Returns:
        Confirmation message
    """
    try:
        session_data = context.userdata
        session_data.vehicle.color = color.strip().title()
        
        logger.info(f"🎨 Collected vehicle color: {color}")
        
        session_data.add_interaction(
            agent_type="vehicle_tools",
            speaker="system",
            message=f"Collected vehicle color: {color}",
            extracted_data={"color": color}
        )
        
        return f"Thank you, I've noted your vehicle is {color}."
        
    except Exception as e:
        logger.error(f"❌ Error collecting vehicle color: {e}")
        return "What color is your vehicle?"


@function_tool
async def check_neutral_gear(
    context: RunContext[RoadsideSessionData],
    neutral_works: bool
) -> str:
    """
    Check if vehicle's neutral gear is functional for towing.
    
    Args:
        neutral_works: Whether neutral gear is functional
    
    Returns:
        Information about towing implications
    """
    try:
        session_data = context.userdata
        session_data.vehicle.neutral_functional = neutral_works
        
        logger.info(f"⚙️ Vehicle neutral gear functional: {neutral_works}")
        
        session_data.add_interaction(
            agent_type="vehicle_tools",
            speaker="system",
            message=f"Neutral gear functional: {neutral_works}",
            extracted_data={"neutral_functional": neutral_works}
        )
        
        if neutral_works:
            return "Great! Since your neutral gear works, we can do a standard tow."
        else:
            return "No problem. We'll need to use a flatbed or dolly for the tow since neutral isn't working."
            
    except Exception as e:
        logger.error(f"❌ Error checking neutral gear: {e}")
        return "Is your vehicle able to be put in neutral gear?"


@function_tool
async def identify_vehicle_type(
    context: RunContext[RoadsideSessionData],
    vehicle_type: str
) -> str:
    """
    Identify the type/category of vehicle.
    
    Args:
        vehicle_type: Type of vehicle (sedan, SUV, truck, etc.)
    
    Returns:
        Confirmation and any relevant service implications
    """
    try:
        session_data = context.userdata
        session_data.vehicle.vehicle_type = vehicle_type.strip().title()
        
        logger.info(f"🚙 Vehicle type identified: {vehicle_type}")
        
        session_data.add_interaction(
            agent_type="vehicle_tools",
            speaker="system",
            message=f"Vehicle type: {vehicle_type}",
            extracted_data={"vehicle_type": vehicle_type}
        )
        
        # Provide relevant information based on vehicle type
        if vehicle_type.lower() in ["truck", "pickup", "pickup truck"]:
            return "Got it, a truck. We'll make sure to send appropriate equipment for the size and weight."
        elif vehicle_type.lower() in ["suv", "suv truck"]:
            return "Perfect, an SUV. We'll account for the higher ground clearance."
        elif vehicle_type.lower() in ["motorcycle", "bike"]:
            return "A motorcycle - we'll send specialized equipment for bike transport."
        else:
            return f"Thank you, I've noted it's a {vehicle_type}."
            
    except Exception as e:
        logger.error(f"❌ Error identifying vehicle type: {e}")
        return "What type of vehicle is it? For example, sedan, SUV, truck, etc."


@function_tool
async def verify_vehicle_information(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Verify and summarize collected vehicle information.
    
    Returns:
        Summary of vehicle information for verification
    """
    try:
        session_data = context.userdata
        vehicle = session_data.vehicle
        
        if vehicle.is_complete:
            summary = f"Let me verify your vehicle: {vehicle.description}"
            
            if vehicle.color:
                summary = f"Let me verify your vehicle: {vehicle.color} {vehicle.year} {vehicle.make} {vehicle.model}"
            
            if vehicle.vehicle_type:
                summary += f" ({vehicle.vehicle_type})"
            
            if vehicle.neutral_functional is not None:
                if vehicle.neutral_functional:
                    summary += ". Neutral gear is working."
                else:
                    summary += ". Neutral gear is not working."
            
            summary += " Is this correct?"
            return summary
        else:
            missing = []
            if not vehicle.year:
                missing.append("year")
            if not vehicle.make:
                missing.append("make")
            if not vehicle.model:
                missing.append("model")
            
            return f"I still need your vehicle's {', '.join(missing)}."
            
    except Exception as e:
        logger.error(f"❌ Error verifying vehicle information: {e}")
        return "Let me make sure I have your vehicle information correct. What year, make, and model is your vehicle?"


@function_tool
async def get_vehicle_status(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Get the current status of vehicle information collection.
    
    Returns:
        Status message about vehicle information completeness
    """
    try:
        session_data = context.userdata
        vehicle = session_data.vehicle
        
        if vehicle.is_complete:
            return f"I have complete vehicle information: {vehicle.description}."
        else:
            missing = []
            if not vehicle.year:
                missing.append("year")
            if not vehicle.make:
                missing.append("make") 
            if not vehicle.model:
                missing.append("model")
            
            if missing:
                return f"I still need your vehicle's {', '.join(missing)}."
            else:
                return "Vehicle information is complete."
                
    except Exception as e:
        logger.error(f"❌ Error getting vehicle status: {e}")
        return "Let me check what vehicle information I have..."


@function_tool
async def update_vehicle_information(
    context: RunContext[RoadsideSessionData],
    field: str,
    value: str
) -> str:
    """
    Update specific vehicle information field.
    
    Args:
        field: Field to update (year, make, model, color, vehicle_type)
        value: New value for the field
    
    Returns:
        Confirmation message about the update
    """
    try:
        session_data = context.userdata
        vehicle = session_data.vehicle
        
        if field == "year":
            if value.isdigit() and 1990 <= int(value) <= 2030:
                vehicle.year = value
                return f"Updated your vehicle year to {value}."
            else:
                return "That doesn't seem like a valid year. Please provide a year between 1990 and 2030."
                
        elif field == "make":
            vehicle.make = value.strip().title()
            return f"Updated your vehicle make to {vehicle.make}."
            
        elif field == "model":
            vehicle.model = value.strip().title()
            return f"Updated your vehicle model to {vehicle.model}."
            
        elif field == "color":
            vehicle.color = value.strip().title()
            return f"Updated your vehicle color to {vehicle.color}."
            
        elif field == "vehicle_type":
            vehicle.vehicle_type = value.strip().title()
            return f"Updated your vehicle type to {vehicle.vehicle_type}."
            
        else:
            return f"I can't update the field '{field}'. I can update year, make, model, color, or vehicle_type."
            
    except Exception as e:
        logger.error(f"❌ Error updating vehicle information: {e}")
        return "I had trouble updating that vehicle information. Could you try again?"