# tools/location_tools.py
"""
Location information management tools for LiveKit 1.1 multi-agent system
"""
import logging
from typing import Optional
from livekit.agents import function_tool, RunContext

from models.session_data import RoadsideSessionData
from services.monitoring_service import get_monitoring_service

logger = logging.getLogger(__name__)


@function_tool
async def collect_street_address(
    context: RunContext[RoadsideSessionData],
    street_address: str
) -> str:
    """
    Collect street address for the service location.
    
    Args:
        street_address: Street address where service is needed
    
    Returns:
        Confirmation message about collected address
    """
    try:
        session_data = context.userdata
        location = session_data.location
        
        # Clean and store address
        clean_address = street_address.strip().title()
        location.street_address = clean_address
        
        logger.info(f"📍 Collected street address: {clean_address}")
        
        session_data.add_interaction(
            agent_type="location_tools",
            speaker="system",
            message=f"Collected address: {clean_address}",
            extracted_data={"street_address": clean_address}
        )
        
        # Update monitoring
        monitoring_service = get_monitoring_service()
        if monitoring_service:
            await monitoring_service.update_session_metrics(session_data, "location_tools")
        
        return f"Got it, {clean_address}. What city and state is that in?"
        
    except Exception as e:
        logger.error(f"❌ Error collecting street address: {e}")
        return "I had trouble recording that address. Could you repeat the street address slowly?"


@function_tool
async def collect_city_state(
    context: RunContext[RoadsideSessionData],
    city: str,
    state: str
) -> str:
    """
    Collect city and state information.
    
    Args:
        city: City name
        state: State name or abbreviation
    
    Returns:
        Confirmation message about collected city/state
    """
    try:
        session_data = context.userdata
        location = session_data.location
        
        # Clean and store city/state
        location.city = city.strip().title()
        
        # Handle state abbreviations
        state_clean = state.strip().upper()
        if len(state_clean) == 2:
            location.state = state_clean
        else:
            location.state = state.strip().title()
        
        logger.info(f"🏙️ Collected city/state: {location.city}, {location.state}")
        
        session_data.add_interaction(
            agent_type="location_tools",
            speaker="system",
            message=f"Collected city/state: {location.city}, {location.state}",
            extracted_data={"city": city, "state": state}
        )
        
        # Check if we have street address too
        if location.street_address:
            return f"Perfect! So you're at {location.street_address}, {location.city}, {location.state}."
        else:
            return f"Thank you, {location.city}, {location.state}. What's the street address?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting city/state: {e}")
        return "Could you tell me the city and state where you need service?"


@function_tool
async def collect_zip_code(
    context: RunContext[RoadsideSessionData],
    zip_code: str
) -> str:
    """
    Collect ZIP code for the service location.
    
    Args:
        zip_code: ZIP code for the location
    
    Returns:
        Confirmation message about collected ZIP code
    """
    try:
        session_data = context.userdata
        
        # Validate ZIP code
        clean_zip = ''.join(filter(str.isdigit, zip_code))
        if len(clean_zip) == 5:
            session_data.location.zip_code = clean_zip
            
            logger.info(f"📮 Collected ZIP code: {clean_zip}")
            
            session_data.add_interaction(
                agent_type="location_tools",
                speaker="system",
                message=f"Collected ZIP: {clean_zip}",
                extracted_data={"zip_code": clean_zip}
            )
            
            return f"Thank you, ZIP code {clean_zip} is recorded."
        else:
            return "That doesn't seem like a valid ZIP code. Could you provide the 5-digit ZIP code?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting ZIP code: {e}")
        return "What's the ZIP code for your location?"


@function_tool
async def collect_complete_address(
    context: RunContext[RoadsideSessionData],
    street_address: str,
    city: str,
    state: str,
    zip_code: Optional[str] = None
) -> str:
    """
    Collect complete address information at once.
    
    Args:
        street_address: Street address
        city: City name
        state: State name or abbreviation
        zip_code: ZIP code (optional)
    
    Returns:
        Confirmation message about collected address
    """
    try:
        session_data = context.userdata
        location = session_data.location
        
        # Store all address components
        location.street_address = street_address.strip().title()
        location.city = city.strip().title()
        
        # Handle state abbreviations
        state_clean = state.strip().upper() if len(state.strip()) == 2 else state.strip().title()
        location.state = state_clean
        
        if zip_code:
            clean_zip = ''.join(filter(str.isdigit, zip_code))
            if len(clean_zip) == 5:
                location.zip_code = clean_zip
        
        logger.info(f"📍 Collected complete address: {location.full_address}")
        
        session_data.add_interaction(
            agent_type="location_tools",
            speaker="system",
            message=f"Collected complete address: {location.full_address}",
            extracted_data={
                "street_address": street_address,
                "city": city,
                "state": state,
                "zip_code": zip_code
            }
        )
        
        return f"Perfect! I have your location as {location.full_address}."
        
    except Exception as e:
        logger.error(f"❌ Error collecting complete address: {e}")
        return "I had trouble recording that address. Could you provide the street address, city, and state?"


@function_tool
async def add_landmarks(
    context: RunContext[RoadsideSessionData],
    landmarks: str
) -> str:
    """
    Add landmark information to help locate the customer.
    
    Args:
        landmarks: Nearby landmarks or additional location details
    
    Returns:
        Confirmation message about landmarks
    """
    try:
        session_data = context.userdata
        session_data.location.landmarks = landmarks.strip()
        
        logger.info(f"🗺️ Added landmarks: {landmarks}")
        
        session_data.add_interaction(
            agent_type="location_tools",
            speaker="system",
            message=f"Added landmarks: {landmarks}",
            extracted_data={"landmarks": landmarks}
        )
        
        return f"Great! I've noted the landmarks: {landmarks}. This will help our technician find you."
        
    except Exception as e:
        logger.error(f"❌ Error adding landmarks: {e}")
        return "Could you describe any nearby landmarks to help our technician find you?"


@function_tool
async def collect_gps_coordinates(
    context: RunContext[RoadsideSessionData],
    latitude: float,
    longitude: float,
    accuracy: Optional[float] = None
) -> str:
    """
    Collect GPS coordinates for precise location.
    
    Args:
        latitude: GPS latitude
        longitude: GPS longitude
        accuracy: GPS accuracy in meters (optional)
    
    Returns:
        Confirmation message about GPS coordinates
    """
    try:
        session_data = context.userdata
        location = session_data.location
        
        # Validate coordinates
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            location.coordinates = {
                "latitude": latitude,
                "longitude": longitude
            }
            
            if accuracy is not None:
                location.gps_accuracy = accuracy
            
            logger.info(f"🛰️ Collected GPS: {latitude}, {longitude}")
            
            session_data.add_interaction(
                agent_type="location_tools",
                speaker="system",
                message=f"Collected GPS coordinates",
                extracted_data={
                    "latitude": latitude,
                    "longitude": longitude,
                    "accuracy": accuracy
                }
            )
            
            if accuracy and accuracy < 10:
                return "Excellent! I have your precise GPS location."
            else:
                return "Thank you, I have your GPS coordinates recorded."
        else:
            return "Those GPS coordinates don't seem valid. Could you check them?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting GPS coordinates: {e}")
        return "I had trouble recording those GPS coordinates."


@function_tool
async def verify_location_information(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Verify and summarize collected location information.
    
    Returns:
        Summary of location information for verification
    """
    try:
        session_data = context.userdata
        location = session_data.location
        
        if location.is_complete:
            summary = f"Let me verify your location: {location.full_address}"
            
            if location.landmarks:
                summary += f". Near: {location.landmarks}"
            
            if location.coordinates:
                summary += ". I also have your GPS coordinates"
            
            summary += ". Is this correct?"
            return summary
        else:
            missing = []
            if not location.street_address:
                missing.append("street address")
            if not location.city:
                missing.append("city")
            if not location.state:
                missing.append("state")
            
            return f"I still need your {', '.join(missing)}."
            
    except Exception as e:
        logger.error(f"❌ Error verifying location information: {e}")
        return "Let me make sure I have your correct location. What's your street address, city, and state?"


@function_tool
async def get_location_status(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Get the current status of location information collection.
    
    Returns:
        Status message about location information completeness
    """
    try:
        session_data = context.userdata
        location = session_data.location
        
        if location.is_complete:
            return f"I have complete location information: {location.full_address}."
        else:
            missing = []
            if not location.street_address:
                missing.append("street address")
            if not location.city:
                missing.append("city")
            if not location.state:
                missing.append("state")
            
            if missing:
                return f"I still need your {', '.join(missing)}."
            else:
                return "Location information is complete."
                
    except Exception as e:
        logger.error(f"❌ Error getting location status: {e}")
        return "Let me check what location information I have..."


@function_tool
async def update_location_information(
    context: RunContext[RoadsideSessionData],
    field: str,
    value: str
) -> str:
    """
    Update specific location information field.
    
    Args:
        field: Field to update (street_address, city, state, zip_code, landmarks)
        value: New value for the field
    
    Returns:
        Confirmation message about the update
    """
    try:
        session_data = context.userdata
        location = session_data.location
        
        if field == "street_address":
            location.street_address = value.strip().title()
            return f"Updated your street address to {location.street_address}."
            
        elif field == "city":
            location.city = value.strip().title()
            return f"Updated your city to {location.city}."
            
        elif field == "state":
            state_clean = value.strip().upper() if len(value.strip()) == 2 else value.strip().title()
            location.state = state_clean
            return f"Updated your state to {location.state}."
            
        elif field == "zip_code":
            clean_zip = ''.join(filter(str.isdigit, value))
            if len(clean_zip) == 5:
                location.zip_code = clean_zip
                return f"Updated your ZIP code to {clean_zip}."
            else:
                return "That doesn't seem like a valid ZIP code. Please provide 5 digits."
                
        elif field == "landmarks":
            location.landmarks = value.strip()
            return f"Updated landmarks to: {location.landmarks}."
            
        else:
            return f"I can't update the field '{field}'. I can update street_address, city, state, zip_code, or landmarks."
            
    except Exception as e:
        logger.error(f"❌ Error updating location information: {e}")
        return "I had trouble updating that location information. Could you try again?"