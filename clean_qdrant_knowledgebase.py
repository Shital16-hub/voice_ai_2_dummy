# clean_qdrant_knowledgebase.py
"""
Comprehensive script to clean Qdrant knowledge base with multiple methods
"""
import asyncio
import logging
from qdrant_rag_system import qdrant_rag
from config import config
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_collection_status():
    """Check current collection status"""
    try:
        await qdrant_rag.initialize()
        
        # Get collection info
        collection_info = await asyncio.to_thread(
            qdrant_rag.client.get_collection,
            config.qdrant_collection_name
        )
        
        logger.info(f"📊 Collection '{config.qdrant_collection_name}' status:")
        logger.info(f"   Points: {collection_info.points_count}")
        logger.info(f"   Segments: {collection_info.segments_count}")
        logger.info(f"   Indexed vectors: {collection_info.indexed_vectors_count}")
        
        await qdrant_rag.close()
        return collection_info.points_count
        
    except Exception as e:
        logger.info(f"📋 Collection doesn't exist or is empty: {e}")
        return 0

async def method_1_delete_collection():
    """Method 1: Delete entire collection"""
    try:
        logger.info("🗑️ Method 1: Deleting entire collection...")
        
        await qdrant_rag.initialize()
        
        # Delete the collection
        await asyncio.to_thread(
            qdrant_rag.client.delete_collection,
            collection_name=config.qdrant_collection_name
        )
        
        logger.info("✅ Collection deleted successfully")
        await qdrant_rag.close()
        return True
        
    except Exception as e:
        logger.info(f"📋 Collection deletion: {e}")
        return False

async def method_2_clear_all_points():
    """Method 2: Clear all points but keep collection structure"""
    try:
        logger.info("🧹 Method 2: Clearing all points...")
        
        await qdrant_rag.initialize()
        
        # Get all point IDs first
        points, _ = await asyncio.to_thread(
            qdrant_rag.client.scroll,
            collection_name=config.qdrant_collection_name,
            limit=10000,  # Get all points
            with_payload=False,  # Don't need payload, just IDs
            with_vectors=False
        )
        
        if points:
            point_ids = [point.id for point in points]
            logger.info(f"🔍 Found {len(point_ids)} points to delete")
            
            # Delete all points
            await asyncio.to_thread(
                qdrant_rag.client.delete,
                collection_name=config.qdrant_collection_name,
                points_selector=point_ids
            )
            
            logger.info(f"✅ Deleted {len(point_ids)} points")
        else:
            logger.info("📋 No points found to delete")
        
        await qdrant_rag.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to clear points: {e}")
        return False

def method_3_docker_reset():
    """Method 3: Reset via Docker (nuclear option)"""
    try:
        logger.info("🐳 Method 3: Docker reset (nuclear option)...")
        
        import subprocess
        import shutil
        from pathlib import Path
        
        # Stop Docker Compose
        logger.info("🛑 Stopping Qdrant Docker container...")
        subprocess.run(["docker-compose", "down"], capture_output=True)
        
        # Remove storage directory
        storage_dir = Path("qdrant_storage")
        if storage_dir.exists():
            logger.info("🗑️ Removing storage directory...")
            shutil.rmtree(storage_dir)
            logger.info("✅ Storage directory removed")
        
        # Recreate empty directory
        storage_dir.mkdir(exist_ok=True)
        
        # Start Docker Compose
        logger.info("🚀 Starting fresh Qdrant container...")
        result = subprocess.run(["docker-compose", "up", "-d"], capture_output=True)
        
        if result.returncode == 0:
            logger.info("✅ Fresh Qdrant container started")
            
            # Wait for container to be ready
            import time
            logger.info("⏳ Waiting for Qdrant to be ready...")
            for i in range(30):
                try:
                    response = requests.get("http://localhost:6333/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Qdrant is ready!")
                        return True
                except:
                    pass
                time.sleep(1)
            
            logger.warning("⚠️ Qdrant may not be fully ready yet, but container is running")
            return True
        else:
            logger.error(f"❌ Failed to start Docker container: {result.stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Docker reset failed: {e}")
        return False

def method_4_direct_api():
    """Method 4: Direct API call to delete collection"""
    try:
        logger.info("🌐 Method 4: Direct API call...")
        
        # Direct HTTP call to delete collection
        url = f"{config.qdrant_url}/collections/{config.qdrant_collection_name}"
        
        response = requests.delete(url, timeout=10)
        
        if response.status_code in [200, 404]:  # 404 means already deleted
            logger.info("✅ Collection deleted via direct API")
            return True
        else:
            logger.error(f"❌ API call failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Direct API call failed: {e}")
        return False

async def verify_cleanup():
    """Verify that cleanup was successful"""
    try:
        logger.info("🔍 Verifying cleanup...")
        
        # Check if collection exists
        url = f"{config.qdrant_url}/collections/{config.qdrant_collection_name}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 404:
            logger.info("✅ Collection successfully deleted")
            return True
        elif response.status_code == 200:
            data = response.json()
            points_count = data.get("result", {}).get("points_count", 0)
            if points_count == 0:
                logger.info("✅ Collection exists but is empty")
                return True
            else:
                logger.warning(f"⚠️ Collection still has {points_count} points")
                return False
        else:
            logger.error(f"❌ Unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

async def main():
    """Main cleanup function with multiple fallback methods"""
    logger.info("🧹 QDRANT KNOWLEDGE BASE CLEANUP")
    logger.info("=" * 50)
    
    # Step 1: Check current status
    logger.info("\n=== STEP 1: Current Status ===")
    point_count = await check_collection_status()
    
    if point_count == 0:
        logger.info("✅ Knowledge base is already empty!")
        return
    
    logger.info(f"📊 Found {point_count} points to clean")
    
    # Try methods in order of preference
    methods = [
        ("Delete Collection", method_1_delete_collection),
        ("Clear All Points", method_2_clear_all_points),
        ("Direct API Call", lambda: method_4_direct_api()),
        ("Docker Reset", lambda: method_3_docker_reset())
    ]
    
    for method_name, method_func in methods:
        logger.info(f"\n=== Trying: {method_name} ===")
        
        if asyncio.iscoroutinefunction(method_func):
            success = await method_func()
        else:
            success = method_func()
        
        if success:
            # Verify cleanup worked
            if await verify_cleanup():
                logger.info(f"🎉 Cleanup successful using: {method_name}")
                break
            else:
                logger.warning(f"⚠️ {method_name} reported success but verification failed")
        else:
            logger.warning(f"⚠️ {method_name} failed, trying next method...")
    else:
        logger.error("❌ All cleanup methods failed!")
        return
    
    logger.info("\n🎉 CLEANUP COMPLETED SUCCESSFULLY!")
    logger.info("💡 You can now re-index your Excel file with:")
    logger.info("   python qdrant_data_ingestion.py --file data/roadside_services.xlsx")

if __name__ == "__main__":
    asyncio.run(main())
    