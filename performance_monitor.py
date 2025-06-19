# performance_monitor.py - Monitor and optimize Qdrant performance
"""
Performance monitoring and optimization script for local Qdrant Docker
"""
import asyncio
import time
import statistics
import logging
from typing import List, Dict
import psutil
import requests

from qdrant_rag_system import qdrant_rag
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantPerformanceMonitor:
    """Monitor and optimize Qdrant performance"""
    
    def __init__(self):
        self.search_times = []
        self.embedding_times = []
        
    async def benchmark_search_performance(self, num_queries: int = 50) -> Dict:
        """Benchmark search performance"""
        logger.info(f"🏃 Running {num_queries} search benchmarks...")
        
        # Initialize if not ready
        if not qdrant_rag.ready:
            await qdrant_rag.initialize()
        
        # Test queries for telephony scenarios
        test_queries = [
            "business hours",
            "contact information", 
            "pricing",
            "support help",
            "account information",
            "services offered",
            "location address",
            "phone number",
            "email contact",
            "appointment booking"
        ]
        
        search_times = []
        embedding_times = []
        
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            
            # Measure embedding time
            embed_start = time.time()
            try:
                await qdrant_rag._create_embedding(query)
                embed_time = (time.time() - embed_start) * 1000
                embedding_times.append(embed_time)
            except Exception as e:
                logger.warning(f"Embedding failed for query {i}: {e}")
                continue
            
            # Measure search time
            search_start = time.time()
            try:
                results = await qdrant_rag.search(query, limit=2)
                search_time = (time.time() - search_start) * 1000
                search_times.append(search_time)
                
                if i % 10 == 0:
                    logger.info(f"Query {i}: {search_time:.1f}ms, {len(results)} results")
                    
            except Exception as e:
                logger.warning(f"Search failed for query {i}: {e}")
                continue
        
        # Calculate statistics
        if search_times:
            stats = {
                "total_queries": len(search_times),
                "avg_search_time_ms": statistics.mean(search_times),
                "median_search_time_ms": statistics.median(search_times),
                "p95_search_time_ms": self._percentile(search_times, 95),
                "p99_search_time_ms": self._percentile(search_times, 99),
                "max_search_time_ms": max(search_times),
                "min_search_time_ms": min(search_times),
                "avg_embedding_time_ms": statistics.mean(embedding_times) if embedding_times else 0,
                "cache_hits": getattr(qdrant_rag, 'cache_hits', 0),
                "target_latency_met": statistics.mean(search_times) < config.rag_timeout_ms
            }
            
            logger.info("📊 Performance Benchmark Results:")
            logger.info(f"   Average search time: {stats['avg_search_time_ms']:.1f}ms")
            logger.info(f"   Median search time: {stats['median_search_time_ms']:.1f}ms")
            logger.info(f"   95th percentile: {stats['p95_search_time_ms']:.1f}ms")  
            logger.info(f"   99th percentile: {stats['p99_search_time_ms']:.1f}ms")
            logger.info(f"   Target met ({config.rag_timeout_ms}ms): {stats['target_latency_met']}")
            
            return stats
        else:
            logger.error("❌ No successful searches completed")
            return {}
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        values_sorted = sorted(values)
        index = int(len(values_sorted) * percentile / 100)
        return values_sorted[min(index, len(values_sorted) - 1)]
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage for Qdrant storage
            storage_path = config.qdrant_storage_dir
            if storage_path.exists():
                disk = psutil.disk_usage(str(storage_path))
            else:
                disk = psutil.disk_usage('/')
            
            stats = {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100
            }
            
            logger.info("🖥️  System Resources:")
            logger.info(f"   CPU: {stats['cpu_percent']:.1f}%")
            logger.info(f"   Memory: {stats['memory_used_gb']:.1f}GB / {stats['memory_total_gb']:.1f}GB ({stats['memory_percent']:.1f}%)")
            logger.info(f"   Disk: {stats['disk_used_gb']:.1f}GB / {stats['disk_total_gb']:.1f}GB ({stats['disk_percent']:.1f}%)")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to check system resources: {e}")
            return {}
    
    def check_qdrant_status(self) -> Dict:
        """Check Qdrant container status"""
        try:
            # Health check
            health_response = requests.get(f"{config.qdrant_url}/health", timeout=2)
            health_ok = health_response.status_code == 200
            
            # Collection info
            collections_response = requests.get(f"{config.qdrant_url}/collections", timeout=2)
            collections = collections_response.json() if collections_response.status_code == 200 else {}
            
            # Specific collection stats
            collection_info = {}
            if health_ok:
                try:
                    collection_response = requests.get(
                        f"{config.qdrant_url}/collections/{config.qdrant_collection_name}",
                        timeout=2
                    )
                    if collection_response.status_code == 200:
                        collection_info = collection_response.json()["result"]
                except:
                    pass
            
            stats = {
                "health_ok": health_ok,
                "collections_count": len(collections.get("result", {}).get("collections", [])),
                "collection_exists": config.qdrant_collection_name in [
                    c["name"] for c in collections.get("result", {}).get("collections", [])
                ],
                "points_count": collection_info.get("points_count", 0),
                "segments_count": collection_info.get("segments_count", 0),
                "indexed_vectors_count": collection_info.get("indexed_vectors_count", 0)
            }
            
            logger.info("🔍 Qdrant Status:")
            logger.info(f"   Health: {'✅ OK' if stats['health_ok'] else '❌ Failed'}")
            logger.info(f"   Collection exists: {'✅ Yes' if stats['collection_exists'] else '❌ No'}")
            logger.info(f"   Points: {stats['points_count']}")
            logger.info(f"   Segments: {stats['segments_count']}")
            logger.info(f"   Indexed vectors: {stats['indexed_vectors_count']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to check Qdrant status: {e}")
            return {"health_ok": False}
    
    async def optimize_performance(self) -> Dict:
        """Provide performance optimization suggestions"""
        suggestions = []
        
        # Benchmark current performance
        perf_stats = await self.benchmark_search_performance(20)
        system_stats = self.check_system_resources()
        qdrant_stats = self.check_qdrant_status()
        
        if not perf_stats:
            return {"error": "Could not benchmark performance"}
        
        avg_time = perf_stats.get("avg_search_time_ms", 0)
        target_time = config.rag_timeout_ms
        
        # Performance suggestions
        if avg_time > target_time:
            suggestions.append(f"🚨 Average search time ({avg_time:.1f}ms) exceeds target ({target_time}ms)")
            
            if avg_time > 150:
                suggestions.append("💡 Consider reducing search_limit from 3 to 2")
                suggestions.append("💡 Consider increasing similarity_threshold from 0.25 to 0.3")
            
            if system_stats.get("memory_percent", 0) > 80:
                suggestions.append("💡 High memory usage detected - consider enabling mmap in config")
            
            if system_stats.get("cpu_percent", 0) > 80:
                suggestions.append("💡 High CPU usage - consider reducing max_search_threads")
        else:
            suggestions.append(f"✅ Performance target met! Average: {avg_time:.1f}ms < {target_time}ms")
        
        # Data suggestions
        points_count = qdrant_stats.get("points_count", 0)
        if points_count == 0:
            suggestions.append("📄 No data found - run data ingestion first")
        elif points_count < 100:
            suggestions.append("💡 Small dataset detected - consider disabling indexing for faster search")
        
        return {
            "performance_stats": perf_stats,
            "system_stats": system_stats,
            "qdrant_stats": qdrant_stats,
            "suggestions": suggestions
        }

async def main():
    """Main monitoring function"""
    monitor = QdrantPerformanceMonitor()
    
    logger.info("🔧 Starting Qdrant Performance Analysis...")
    
    # Run comprehensive performance check
    results = await monitor.optimize_performance()
    
    if "error" in results:
        logger.error(f"❌ {results['error']}")
        return
    
    logger.info("\n📋 Performance Optimization Suggestions:")
    for suggestion in results["suggestions"]:
        logger.info(f"   {suggestion}")
    
    logger.info("\n🎯 Performance Summary:")
    perf = results["performance_stats"]
    logger.info(f"   Target latency: {config.rag_timeout_ms}ms")
    logger.info(f"   Average achieved: {perf.get('avg_search_time_ms', 0):.1f}ms")
    logger.info(f"   Success rate: {(perf.get('total_queries', 0) / 50) * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())