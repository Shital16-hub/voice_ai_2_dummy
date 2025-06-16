# test_fixed_agent.py
"""
Test script to verify the fixed agent implementation works
"""
import asyncio
import logging
from datetime import datetime

# Test imports
def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        # Test LiveKit imports
        from livekit.agents import Agent, AgentSession, function_tool
        print("✅ LiveKit agents imported successfully")
        
        # Test our fixed classes
        from main import (
            RoadsideAssistanceAgent, IntelligentMonitoringAgent,
            ServiceType, ServiceRequest, CustomerInfo, VehicleInfo, LocationInfo
        )
        print("✅ Fixed agent classes imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_agent_creation():
    """Test that we can create the agent without errors"""
    print("\n🧪 Testing agent creation...")
    
    try:
        from main import RoadsideAssistanceAgent
        
        # This should not raise the "can't set attribute 'session'" error
        agent = RoadsideAssistanceAgent()
        print("✅ Agent created successfully")
        
        # Test setting room name
        agent.set_room_name("test-room")
        print("✅ Room name set successfully")
        
        # Check monitoring agent was initialized
        if agent.monitoring_agent:
            print("✅ Monitoring agent initialized")
        else:
            print("❌ Monitoring agent not initialized")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return False

def test_data_structures():
    """Test that our data structures work correctly"""
    print("\n🧪 Testing data structures...")
    
    try:
        from main import ServiceType, ServiceRequest, CustomerInfo, VehicleInfo, LocationInfo
        
        # Test enum
        service = ServiceType.TOWING
        print(f"✅ ServiceType enum: {service.value}")
        
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
        
        print(f"✅ Service request created: {service_request.service_type.value}")
        print(f"   Customer: {customer.name}")
        print(f"   Vehicle: {vehicle.year} {vehicle.make} {vehicle.model}")
        print(f"   Location: {location.city}, {location.state}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure error: {e}")
        return False

async def test_monitoring_agent():
    """Test the monitoring agent functionality"""
    print("\n🧪 Testing monitoring agent...")
    
    try:
        from main import IntelligentMonitoringAgent
        
        monitor = IntelligentMonitoringAgent("test-room")
        
        # Test information extraction
        result = await monitor.extract_customer_info(name="John Smith", phone="555-123-4567")
        print(f"✅ Customer extraction: {result}")
        
        result = await monitor.extract_vehicle_info(year="2020", make="Honda", model="Civic")
        print(f"✅ Vehicle extraction: {result}")
        
        result = await monitor.extract_location_info(city="Boston", state="MA")
        print(f"✅ Location extraction: {result}")
        
        result = await monitor.extract_service_info("towing", estimated_cost="$150")
        print(f"✅ Service extraction: {result}")
        
        # Test transcript functionality
        monitor.add_transcript("customer", "I need a tow truck", True, 0.95)
        monitor.add_transcript("agent", "I can help you with that", True)
        
        print(f"✅ Transcripts added: {len(monitor.transcripts)} entries")
        
        # Test summary generation
        summary = monitor.get_conversation_summary()
        print(f"✅ Summary generated with {summary['transcript_count']} transcripts")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring agent error: {e}")
        return False

async def test_function_tools():
    """Test that function tools can be called"""
    print("\n🧪 Testing function tools...")
    
    try:
        from main import RoadsideAssistanceAgent
        from livekit.agents import RunContext
        
        agent = RoadsideAssistanceAgent()
        agent.set_room_name("test-room")
        
        # Create a mock RunContext (in real usage, this comes from LiveKit)
        class MockRunContext:
            pass
        
        mock_ctx = MockRunContext()
        
        # Test function tools
        result = await agent.extract_customer_info(mock_ctx, name="Test User", phone="5551234567")
        print(f"✅ Customer info tool: {result}")
        
        result = await agent.extract_vehicle_info(mock_ctx, year="2021", make="Ford", model="F150")
        print(f"✅ Vehicle info tool: {result}")
        
        result = await agent.extract_location_info(mock_ctx, city="New York", state="NY")
        print(f"✅ Location info tool: {result}")
        
        result = await agent.extract_service_info(mock_ctx, service_type="towing")
        print(f"✅ Service info tool: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Function tools error: {e}")
        return False

async def test_json_serialization():
    """Test JSON serialization of data"""
    print("\n🧪 Testing JSON serialization...")
    
    try:
        from main import IntelligentMonitoringAgent
        import json
        
        monitor = IntelligentMonitoringAgent("test-room")
        
        # Add some test data using await (not asyncio.run)
        await monitor.extract_customer_info(name="Test User", phone="5551234567")
        await monitor.extract_vehicle_info(year="2020", make="Honda", model="Civic")
        
        # Generate summary and serialize
        summary = monitor.get_conversation_summary()
        json_str = json.dumps(summary, indent=2, default=str)
        
        print(f"✅ JSON serialization successful")
        print(f"   JSON length: {len(json_str)} characters")
        
        # Test that we can parse it back
        parsed = json.loads(json_str)
        print(f"✅ JSON parsing successful")
        print(f"   Service type: {parsed['service_request']['service_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Fixed LiveKit Agent Implementation")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Agent Creation", test_agent_creation),
        ("Data Structures", test_data_structures),
    ]
    
    # Async tests
    async_tests = [
        ("JSON Serialization", test_json_serialization),
        ("Monitoring Agent", test_monitoring_agent),
        ("Function Tools", test_function_tools),
    ]
    
    # Run synchronous tests
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Run asynchronous tests
    for test_name, test_func in async_tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:12} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Your fixed agent should work correctly now:")
        print("1. The 'session' attribute conflict is resolved")
        print("2. Agent creation works without errors")
        print("3. All function tools are properly defined")
        print("4. Data extraction and monitoring work")
        print("5. JSON serialization works for transcripts")
        print("\n🚀 Ready to run: python main.py dev")
    else:
        print(f"\n⚠️ {failed} tests failed. Check the errors above.")
        
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)