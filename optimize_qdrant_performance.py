# fixed_qdrant_optimization.py
"""
Fixed optimization script compatible with your Qdrant version
"""
import asyncio
import logging
import time
from qdrant_rag_system import qdrant_rag
from config import config
from qdrant_client.http.models import OptimizersConfigDiff, HnswConfigDiff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def optimize_collection_compatible():
    """Optimize collection using compatible methods"""
    try:
        await qdrant_rag.initialize()
        
        # Get current collection info
        collection_info = await asyncio.to_thread(
            qdrant_rag.client.get_collection,
            config.qdrant_collection_name
        )
        
        logger.info(f"📊 Current collection stats:")
        logger.info(f"   Points: {collection_info.points_count}")
        logger.info(f"   Segments: {collection_info.segments_count}")
        logger.info(f"   Indexed vectors: {collection_info.indexed_vectors_count}")
        
        # Use update_collection instead of optimize_collection
        logger.info("🔧 Updating collection configuration for better performance...")
        
        # Update collection with better HNSW parameters
        await asyncio.to_thread(
            qdrant_rag.client.update_collection,
            collection_name=config.qdrant_collection_name,
            optimizer_config=OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=100,
                default_segment_number=1,  # Single segment for better performance
                max_segment_size=None,
                memmap_threshold=20000,  # Use memory mapping for better speed
                indexing_threshold=10000,
                flush_interval_sec=5,
                max_optimization_threads=1
            ),
            hnsw_config=HnswConfigDiff(
                m=16,  # Better connectivity
                ef_construct=200,  # Better index quality
                full_scan_threshold=5000,
                max_indexing_threads=0,
                on_disk=False  # Keep in memory for speed
            )
        )
        
        logger.info("✅ Collection configuration updated")
        
        # Create a collection snapshot for backup
        logger.info("📸 Creating collection snapshot...")
        snapshot_info = await asyncio.to_thread(
            qdrant_rag.client.create_snapshot,
            collection_name=config.qdrant_collection_name
        )
        logger.info(f"✅ Snapshot created: {snapshot_info.name}")
        
        await qdrant_rag.close()
        logger.info("✅ Collection optimization completed")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")

async def reindex_with_better_chunking():
    """Reindex the data with improved chunking strategy"""
    try:
        logger.info("🔄 Starting data reindexing with improved chunking...")
        
        # First, clear existing data
        await qdrant_rag.initialize()
        
        # Get all current points to understand the data better
        points, _ = await asyncio.to_thread(
            qdrant_rag.client.scroll,
            collection_name=config.qdrant_collection_name,
            limit=100,
            with_payload=True
        )
        
        logger.info(f"📊 Found {len(points)} existing points")
        
        # Group related information for better chunking
        improved_documents = []
        service_groups = {}
        
        for point in points:
            text = point.payload.get("text", "")
            
            # Extract service information and group by type
            if "Service Type:" in text:
                # Parse the service information
                parts = text.split(";")
                service_type = ""
                service_name = ""
                description = ""
                cost = ""
                
                for part in parts:
                    part = part.strip()
                    if part.startswith("Service Type:"):
                        service_type = part.replace("Service Type:", "").strip()
                    elif part.startswith("Service Name:"):
                        service_name = part.replace("Service Name:", "").strip()
                    elif part.startswith("Description:"):
                        description = part.replace("Description:", "").strip()
                    elif part.startswith("Base Cost:"):
                        cost = part.replace("Base Cost:", "").strip()
                
                if service_type not in service_groups:
                    service_groups[service_type] = []
                
                # Create a more readable, search-friendly text
                improved_text = f"{service_type} Service - {service_name}: {description}"
                if cost:
                    improved_text += f" Cost: {cost}"
                
                service_groups[service_type].append({
                    "text": improved_text,
                    "service_type": service_type,
                    "service_name": service_name,
                    "description": description,
                    "cost": cost
                })
            
            elif "Plan Name:" in text:
                # Handle membership plans
                improved_documents.append({
                    "id": f"membership_{len(improved_documents)}",
                    "text": text.replace(";", ". "),  # Make it more readable
                    "metadata": {
                        "category": "membership",
                        "type": "plan_details",
                        "source": "roadside_services.xlsx"
                    }
                })
            
            elif "Category: Company" in text:
                # Handle company information
                improved_documents.append({
                    "id": f"company_{len(improved_documents)}",
                    "text": text.replace(";", ". "),
                    "metadata": {
                        "category": "company_info",
                        "type": "details",
                        "source": "roadside_services.xlsx"
                    }
                })
        
        # Create comprehensive service summaries
        for service_type, services in service_groups.items():
            # Create a comprehensive service description
            service_names = [s["service_name"] for s in services]
            descriptions = [s["description"] for s in services]
            
            comprehensive_text = f"{service_type} Services Available: "
            comprehensive_text += f"We offer {', '.join(service_names)}. "
            comprehensive_text += f"Services include: {'. '.join(descriptions)}."
            
            improved_documents.append({
                "id": f"{service_type.lower()}_services",
                "text": comprehensive_text,
                "metadata": {
                    "category": f"{service_type.lower()}_services",
                    "type": "service_group",
                    "source": "roadside_services.xlsx"
                }
            })
            
            # Also keep individual services for specific queries
            for i, service in enumerate(services):
                improved_documents.append({
                    "id": f"{service_type.lower()}_{i}",
                    "text": service["text"],
                    "metadata": {
                        "category": f"{service_type.lower()}_service",
                        "type": "individual_service",
                        "service_name": service["service_name"],
                        "source": "roadside_services.xlsx"
                    }
                })
        
        logger.info(f"✅ Created {len(improved_documents)} improved documents")
        
        # Clear existing collection
        await asyncio.to_thread(
            qdrant_rag.client.delete_collection,
            collection_name=config.qdrant_collection_name
        )
        logger.info("🗑️ Cleared existing collection")
        
        # Reinitialize to create collection with better config
        await qdrant_rag.close()
        await qdrant_rag.initialize()
        
        # Add improved documents
        success = await qdrant_rag.add_documents(improved_documents)
        
        if success:
            logger.info(f"✅ Successfully reindexed {len(improved_documents)} improved documents")
        else:
            logger.error("❌ Failed to reindex documents")
        
        await qdrant_rag.close()
        
    except Exception as e:
        logger.error(f"❌ Reindexing failed: {e}")

async def comprehensive_performance_test():
    """Comprehensive performance test with roadside assistance scenarios"""
    try:
        await qdrant_rag.initialize()
        
        # Realistic customer queries for roadside assistance
        realistic_queries = [
            # Direct service requests
            "I need towing service",
            "My battery is dead, can you help?",
            "I have a flat tire",
            "I'm locked out of my car",
            "I ran out of gas",
            
            # Information requests
            "What towing services do you offer?",
            "How much does battery service cost?",
            "What membership plans are available?",
            "What are your service hours?",
            "How fast do you respond?",
            
            # Scenario-based queries
            "My car broke down on the highway",
            "I need emergency roadside assistance",
            "What services are included in membership?",
            "Do you provide 24/7 service?",
            "Can you tow large vehicles?"
        ]
        
        logger.info("🧪 Testing realistic customer scenarios...")
        
        total_time = 0
        successful_searches = 0
        high_quality_results = 0
        
        for i, query in enumerate(realistic_queries, 1):
            start_time = time.time()
            results = await qdrant_rag.search(query, limit=3)
            end_time = time.time()
            search_time = (end_time - start_time) * 1000
            total_time += search_time
            
            if results:
                successful_searches += 1
                best_score = max(r["score"] for r in results)
                
                if best_score > 0.5:
                    high_quality_results += 1
                    logger.info(f"✅ Query {i}: '{query}' - {search_time:.0f}ms, score: {best_score:.3f}")
                    logger.info(f"   📄 Result: {results[0]['text'][:100]}...")
                else:
                    logger.warning(f"⚠️ Query {i}: '{query}' - {search_time:.0f}ms, low score: {best_score:.3f}")
            else:
                logger.error(f"❌ Query {i}: '{query}' - {search_time:.0f}ms, no results")
        
        # Performance summary
        avg_time = total_time / len(realistic_queries)
        success_rate = (successful_searches / len(realistic_queries)) * 100
        quality_rate = (high_quality_results / len(realistic_queries)) * 100
        
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"   Average search time: {avg_time:.0f}ms")
        logger.info(f"   Success rate: {success_rate:.1f}% ({successful_searches}/{len(realistic_queries)})")
        logger.info(f"   High-quality results: {quality_rate:.1f}% ({high_quality_results}/{len(realistic_queries)})")
        logger.info(f"   Target performance: <500ms, >80% success, >60% high-quality")
        
        # Performance recommendations
        if avg_time > 500:
            logger.warning("⚠️ Average search time too high - consider reducing search_limit")
        if success_rate < 80:
            logger.warning("⚠️ Success rate too low - consider reindexing with better chunking")
        if quality_rate < 60:
            logger.warning("⚠️ Quality rate too low - consider lowering similarity_threshold")
        
        await qdrant_rag.close()
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")

if __name__ == "__main__":
    async def main():
        logger.info("🚀 Starting FIXED Qdrant Performance Optimization")
        
        print("\nChoose optimization strategy:")
        print("1. Quick optimization (update configuration only)")
        print("2. Full reindexing (better chunking + optimization)")
        print("3. Performance test only")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            await optimize_collection_compatible()
            await comprehensive_performance_test()
        elif choice == "2":
            await reindex_with_better_chunking()
            await optimize_collection_compatible()
            await comprehensive_performance_test()
        elif choice == "3":
            await comprehensive_performance_test()
        else:
            logger.info("Running full optimization by default...")
            await optimize_collection_compatible()
            await comprehensive_performance_test()
        
        logger.info("🎉 Optimization completed!")
    
    asyncio.run(main())