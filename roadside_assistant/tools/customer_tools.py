# tools/customer_tools.py
"""
Customer information management tools for LiveKit 1.1 multi-agent system
"""
import logging
from typing import Optional
from livekit.agents import function_tool, RunContext

from models.session_data import RoadsideSessionData
from services.monitoring_service import get_monitoring_service

logger = logging.getLogger(__name__)


@function_tool
async def collect_customer_name(
    context: RunContext[RoadsideSessionData],
    first_name: str,
    last_name: Optional[str] = None
) -> str:
    """
    Collect customer's name information.
    
    Args:
        first_name: Customer's first name
        last_name: Customer's last name (optional if only first name provided)
    
    Returns:
        Confirmation message about collected information
    """
    try:
        session_data = context.userdata
        
        # Update customer information
        session_data.customer.first_name = first_name.strip().title()
        if last_name:
            session_data.customer.last_name = last_name.strip().title()
        
        # Log the collection
        name_collected = session_data.customer.full_name
        logger.info(f"📝 Collected customer name: {name_collected}")
        
        # Add to interaction log
        session_data.add_interaction(
            agent_type="customer_tools",
            speaker="system",
            message=f"Collected name: {name_collected}",
            extracted_data={"first_name": first_name, "last_name": last_name}
        )
        
        # Update monitoring
        monitoring_service = get_monitoring_service()
        if monitoring_service:
            await monitoring_service.update_session_metrics(session_data, "customer_tools")
        
        # Return appropriate response
        if last_name:
            return f"Thank you, {first_name}. I have your full name as {name_collected}."
        else:
            return f"Thank you, {first_name}. Could you also provide your last name?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting customer name: {e}")
        return "I had trouble recording your name. Could you please repeat it?"


@function_tool
async def collect_customer_phone(
    context: RunContext[RoadsideSessionData],
    phone_number: str
) -> str:
    """
    Collect customer's phone number.
    
    Args:
        phone_number: Customer's phone number in any format
    
    Returns:
        Confirmation message about collected phone number
    """
    try:
        session_data = context.userdata
        
        # Clean and format phone number
        clean_phone = ''.join(filter(str.isdigit, phone_number))
        
        if len(clean_phone) >= 10:
            # Format as (XXX) XXX-XXXX
            formatted_phone = f"({clean_phone[-10:-7]}) {clean_phone[-7:-4]}-{clean_phone[-4:]}"
            session_data.customer.phone = formatted_phone
            
            logger.info(f"📞 Collected customer phone: {formatted_phone}")
            
            # Add to interaction log
            session_data.add_interaction(
                agent_type="customer_tools",
                speaker="system",
                message=f"Collected phone: {formatted_phone}",
                extracted_data={"phone": formatted_phone}
            )
            
            # Update monitoring
            monitoring_service = get_monitoring_service()
            if monitoring_service:
                await monitoring_service.update_session_metrics(session_data, "customer_tools")
            
            return f"Perfect! I have your callback number as {formatted_phone}."
            
        else:
            logger.warning(f"⚠️ Invalid phone number format: {phone_number}")
            return "I couldn't quite get that phone number. Could you repeat it slowly with all 10 digits?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting customer phone: {e}")
        return "I had trouble recording your phone number. Could you please repeat it?"


@function_tool
async def collect_customer_email(
    context: RunContext[RoadsideSessionData],
    email: str
) -> str:
    """
    Collect customer's email address (optional).
    
    Args:
        email: Customer's email address
    
    Returns:
        Confirmation message about collected email
    """
    try:
        session_data = context.userdata
        
        # Basic email validation
        if "@" in email and "." in email:
            session_data.customer.email = email.lower().strip()
            
            logger.info(f"📧 Collected customer email: {email}")
            
            # Add to interaction log
            session_data.add_interaction(
                agent_type="customer_tools",
                speaker="system",
                message=f"Collected email: {email}",
                extracted_data={"email": email}
            )
            
            return f"Thank you, I've recorded your email as {email}."
        else:
            return "That doesn't seem like a valid email address. Could you spell it out for me?"
            
    except Exception as e:
        logger.error(f"❌ Error collecting customer email: {e}")
        return "I had trouble recording your email. Could you please repeat it?"


@function_tool
async def verify_customer_information(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Verify and summarize collected customer information.
    
    Returns:
        Summary of customer information for verification
    """
    try:
        session_data = context.userdata
        customer = session_data.customer
        
        # Build verification summary
        summary_parts = []
        
        if customer.first_name and customer.last_name:
            summary_parts.append(f"Name: {customer.full_name}")
        elif customer.first_name:
            summary_parts.append(f"First name: {customer.first_name}")
        
        if customer.phone:
            summary_parts.append(f"Phone: {customer.phone}")
        
        if customer.email:
            summary_parts.append(f"Email: {customer.email}")
        
        if summary_parts:
            summary = "Let me verify your information: " + ", ".join(summary_parts)
            
            # Check completeness
            if customer.is_complete:
                summary += ". Is this information correct?"
            else:
                missing = []
                if not customer.first_name or not customer.last_name:
                    missing.append("full name")
                if not customer.phone:
                    missing.append("phone number")
                
                if missing:
                    summary += f". I still need your {' and '.join(missing)}."
            
            return summary
        else:
            return "I don't have any customer information yet. Could you please provide your name and phone number?"
            
    except Exception as e:
        logger.error(f"❌ Error verifying customer information: {e}")
        return "Let me make sure I have your correct information. Could you repeat your name and phone number?"


@function_tool
async def update_customer_information(
    context: RunContext[RoadsideSessionData],
    field: str,
    value: str
) -> str:
    """
    Update specific customer information field.
    
    Args:
        field: Field to update (first_name, last_name, phone, email)
        value: New value for the field
    
    Returns:
        Confirmation message about the update
    """
    try:
        session_data = context.userdata
        customer = session_data.customer
        
        if field == "first_name":
            customer.first_name = value.strip().title()
            return f"Updated your first name to {customer.first_name}."
            
        elif field == "last_name":
            customer.last_name = value.strip().title()
            return f"Updated your last name to {customer.last_name}."
            
        elif field == "phone":
            clean_phone = ''.join(filter(str.isdigit, value))
            if len(clean_phone) >= 10:
                formatted_phone = f"({clean_phone[-10:-7]}) {clean_phone[-7:-4]}-{clean_phone[-4:]}"
                customer.phone = formatted_phone
                return f"Updated your phone number to {formatted_phone}."
            else:
                return "That doesn't seem like a valid phone number. Could you provide all 10 digits?"
                
        elif field == "email":
            if "@" in value and "." in value:
                customer.email = value.lower().strip()
                return f"Updated your email to {customer.email}."
            else:
                return "That doesn't seem like a valid email address."
        else:
            return f"I can't update the field '{field}'. I can update first_name, last_name, phone, or email."
            
    except Exception as e:
        logger.error(f"❌ Error updating customer information: {e}")
        return "I had trouble updating that information. Could you try again?"


@function_tool
async def get_customer_status(
    context: RunContext[RoadsideSessionData]
) -> str:
    """
    Get the current status of customer information collection.
    
    Returns:
        Status message about customer information completeness
    """
    try:
        session_data = context.userdata
        customer = session_data.customer
        
        if customer.is_complete:
            return "I have all required customer information: name and phone number."
        else:
            missing = []
            if not customer.first_name:
                missing.append("first name")
            if not customer.last_name:
                missing.append("last name")
            if not customer.phone:
                missing.append("phone number")
            
            if missing:
                return f"I still need: {', '.join(missing)}."
            else:
                return "Customer information is complete."
                
    except Exception as e:
        logger.error(f"❌ Error getting customer status: {e}")
        return "Let me check what customer information I have..."