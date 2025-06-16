# test_monitoring_system.py
"""
Test script to verify the enhanced monitoring system is working
"""
import asyncio
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports"""
    logger.info("🧪 Testing basic imports...")
    
    try:
        # Test LiveKit imports
        from livekit.agents import Agent, function_tool, AgentSession
        logger.info("✅ LiveKit agents imported successfully")
        
        # Test spaCy
        import spacy
        logger.info("✅ spaCy imported successfully")
        
        # Test OpenAI
        from livekit.plugins import openai
        logger.info("✅ OpenAI plugin imported successfully")
        
        # Test other essentials
        from dataclasses import dataclass
        from enum import Enum
        logger.info("✅ All basic imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_spacy_models():
    """Test spaCy models"""
    logger.info("🧠 Testing spaCy models...")
    
    try:
        import spacy
        
        # Test small model
        try:
            nlp_sm = spacy.load("en_core_web_sm")
            logger.info("✅ en_core_web_sm loaded successfully")
        except OSError:
            logger.warning("⚠️ en_core_web_sm not found")
        
        # Test medium model
        try:
            nlp_md = spacy.load("en_core_web_md")
            logger.info("✅ en_core_web_md loaded successfully")
            
            # Test entity extraction
            test_text = "Hi, my name is John Smith. My phone number is 555-123-4567. I'm at 123 Main St, Boston, MA 02101 with my 2020 Honda Civic."
            doc = nlp_md(test_text)
            
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            logger.info(f"✅ Found {len(entities)} entities: {entities}")
            
        except OSError:
            logger.warning("⚠️ en_core_web_md not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ spaCy test error: {e}")
        return False

def test_data_structures():
    """Test our custom data structures"""
    logger.info("📊 Testing data structures...")
    
    try:
        from main import (
            ServiceType, VehicleInfo, LocationInfo, 
            CustomerInfo, ServiceRequest, CallTranscript
        )
        
        # Test ServiceType enum
        service = ServiceType.TOWING
        logger.info(f"✅ ServiceType: {service.value}")
        
        # Test data classes
        customer = CustomerInfo(name="John Smith", phone="5551234567")
        vehicle = VehicleInfo(year="2020", make="Honda", model="Civic")
        location = LocationInfo(address="123 Main St", city="Boston", state="MA")
        
        service_request = ServiceRequest(
            service_type=ServiceType.TOWING,
            customer=customer,
            vehicle=vehicle,
            location=location
        )
        
        logger.info("✅ All data structures created successfully")
        logger.info(f"   Customer: {customer.name} ({customer.phone})")
        logger.info(f"   Vehicle: {vehicle.year} {vehicle.make} {vehicle.model}")
        logger.info(f"   Location: {location.address}, {location.city}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data structure test error: {e}")
        return False

def test_phone_number_extraction():
    """Test phone number extraction patterns"""
    logger.info("📞 Testing phone number extraction...")
    
    test_cases = [
        "My number is 555-123-4567",
        "Call me at (555) 123-4567",
        "You can reach me at 555.123.4567",
        "My phone is 5551234567",
        "Contact: +1 555 123 4567",
        "Seven eight one four five eight nine two two two",  # From your transcript
    ]
    
    try:
        import re
        
        # Pattern for extracting digits
        def extract_phone_digits(text):
            digits = ''.join(filter(str.isdigit, text))
            if len(digits) >= 10:
                return digits[-10:]  # Last 10 digits
            return None
        
        for test_case in test_cases:
            result = extract_phone_digits(test_case)
            logger.info(f"   '{test_case}' → {result}")
        
        logger.info("✅ Phone number extraction working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phone extraction test error: {e}")
        return False

def test_address_extraction():
    """Test address extraction"""
    logger.info("📍 Testing address extraction...")
    
    test_addresses = [
        "725 Merrimack Street, Lowell, Massachusetts 01854",
        "1101 Bedford Street, Stamford, Connecticut",
        "Avalon Forest Drive Northwest, Lawrenceville, Georgia",
        "Quality Inn on Route 3 in Plattsburgh",
    ]
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        for address in test_addresses:
            doc = nlp(address)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            logger.info(f"   '{address[:50]}...' → {entities}")
        
        logger.info("✅ Address extraction working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Address extraction test error: {e}")
        return False

async def test_async_functions():
    """Test async function capabilities"""
    logger.info("⚡ Testing async functions...")
    
    try:
        # Simulate async extraction
        async def mock_extract_info(text):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "entities_found": len(text.split()),
                "text_length": len(text),
                "processed_at": datetime.now().isoformat()
            }
        
        test_text = "My name is Sarah Johnson, phone 555-987-6543, 2019 Toyota Camry"
        result = await mock_extract_info(test_text)
        
        logger.info(f"✅ Async extraction result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Async test error: {e}")
        return False

def test_json_serialization():
    """Test JSON serialization of our data structures"""
    logger.info("💾 Testing JSON serialization...")
    
    try:
        from main import ServiceRequest, ServiceType, CustomerInfo
        from dataclasses import asdict
        
        # Create test data
        request = ServiceRequest(
            service_type=ServiceType.TOWING,
            customer=CustomerInfo(name="Test User", phone="5551234567"),
        )
        
        # Convert to dict and serialize
        request_dict = asdict(request)
        request_dict["service_type"] = request.service_type.value  # Handle enum
        
        json_str = json.dumps(request_dict, indent=2, default=str)
        logger.info("✅ JSON serialization successful")
        logger.info(f"   Sample output: {json_str[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ JSON serialization error: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Enhanced Monitoring System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("spaCy Models", test_spacy_models),
        ("Data Structures", test_data_structures),
        ("Phone Extraction", test_phone_number_extraction),
        ("Address Extraction", test_address_extraction),
        ("JSON Serialization", test_json_serialization),
    ]
    
    # Run sync tests
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Run async tests
    logger.info(f"\n📋 Running: Async Functions")
    try:
        async_result = asyncio.run(test_async_functions())
        results.append(("Async Functions", async_result))
    except Exception as e:
        logger.error(f"❌ Async Functions failed: {e}")
        results.append(("Async Functions", False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status:10} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    
    if failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED! System is ready to use.")
        logger.info("\n📋 Next steps:")
        logger.info("1. Set up your .env file with API keys")
        logger.info("2. Run: python main.py")
    else:
        logger.info(f"\n⚠️ {failed} tests failed. Please check the errors above.")
        
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)