"""
Memory Cleaner Service - Automatic memory management for Trading Education AI
Runs in background and cleans up memory periodically
"""

import psutil
import gc
import time
import os
import logging
from datetime import datetime
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/memory-cleaner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryCleaner:
    """Automatic memory cleanup service"""
    
    def __init__(self):
        self.cleanup_interval = int(os.getenv("CLEANUP_INTERVAL", "600"))
        self.memory_threshold = int(os.getenv("MEMORY_THRESHOLD_MB", "1500"))
        self.last_cleanup = datetime.now()
        
        logger.info(f"Memory Cleaner initialized")
        logger.info(f"Cleanup interval: {self.cleanup_interval} seconds")
        logger.info(f"Memory threshold: {self.memory_threshold} MB")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_system_memory_usage(self) -> Tuple[float, float]:
        """Get system memory usage (used, available) in GB"""
        memory = psutil.virtual_memory()
        return memory.used / (1024 ** 3), memory.available / (1024 ** 3)
    
    def cleanup(self) -> Dict[str, float]:
        """Perform memory cleanup"""
        logger.info("=" * 60)
        logger.info("MEMORY CLEANUP STARTED")
        logger.info("=" * 60)
        
        # Get memory before
        mem_before = self.get_memory_usage()
        sys_used, sys_available = self.get_system_memory_usage()
        
        logger.info(f"Process memory before: {mem_before:.1f} MB")
        logger.info(f"System: {sys_used:.1f}GB used, {sys_available:.1f}GB available")
        
        try:
            # Python garbage collection
            logger.info("Running Python garbage collection (3 passes)...")
            for i in range(3):
                collected = gc.collect()
                logger.info(f"  Pass {i+1}: {collected} objects collected")
            
            # Get memory after
            time.sleep(0.5)  # Wait for cleanup to complete
            mem_after = self.get_memory_usage()
            freed = mem_before - mem_after
            
            sys_used_after, sys_available_after = self.get_system_memory_usage()
            
            logger.info("=" * 60)
            logger.info("MEMORY CLEANUP COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Process memory after: {mem_after:.1f} MB")
            logger.info(f"Process memory freed: {freed:.1f} MB ({(freed/mem_before*100):.1f}%)")
            logger.info(f"System: {sys_used_after:.1f}GB used, {sys_available_after:.1f}GB available")
            
            self.last_cleanup = datetime.now()
            
            return {
                'mem_before': mem_before,
                'mem_after': mem_after,
                'freed': freed,
                'freed_percent': (freed / mem_before * 100) if mem_before > 0 else 0,
                'sys_available': sys_available_after
            }
        
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return {}
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        # Check time interval
        elapsed = (datetime.now() - self.last_cleanup).total_seconds()
        time_triggered = elapsed > self.cleanup_interval
        
        # Check memory threshold
        current_mem = self.get_memory_usage()
        mem_triggered = current_mem > self.memory_threshold
        
        return time_triggered or mem_triggered
    
    def run(self):
        """Main cleanup loop"""
        logger.info("Memory Cleaner Service Started")
        logger.info(f"Process ID: {os.getpid()}")
        
        try:
            while True:
                if self.should_cleanup():
                    result = self.cleanup()
                    
                    # Log summary
                    if result:
                        logger.info(
                            f"Cleanup: {result['freed']:.1f}MB freed "
                            f"({result['freed_percent']:.1f}%), "
                            f"System available: {result['sys_available']:.1f}GB"
                        )
                
                # Check every 60 seconds
                time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Memory Cleaner Service Stopped")
        except Exception as e:
            logger.error(f"Fatal error in Memory Cleaner: {str(e)}")
            raise

def main():
    """Entry point"""
    cleaner = MemoryCleaner()
    cleaner.run()

if __name__ == "__main__":
    main()
