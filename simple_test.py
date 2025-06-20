# simple_test.py - Test Qdrant connection (FIXED)
import requests
import time

def test_qdrant():
    """Simple test for Qdrant connection"""
    print("🔍 Testing Qdrant connection...")
    
    # Wait for container to start
    for attempt in range(10):  # Reduced attempts since it's already running
        try:
            # Try the root endpoint instead of /health
            response = requests.get("http://localhost:6333/", timeout=2)
            if response.status_code == 200:
                print("✅ Qdrant is running!")
                print(f"Response: {response.text[:200]}...")  # Show first 200 chars
                
                # Test collections endpoint
                collections = requests.get("http://localhost:6333/collections", timeout=2)
                print(f"Collections: {collections.json()}")
                return True
                
        except requests.exceptions.ConnectionError:
            print(f"⏳ Attempt {attempt + 1}/10: Waiting for Qdrant...")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)
    
    print("❌ Qdrant failed to start")
    return False

if __name__ == "__main__":
    test_qdrant()