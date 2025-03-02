import runpod
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class QueueManager:
    def __init__(self):
        self.model_ready = False
    
    async def publish(self, job_id: str, params: dict, priority: int = 0) -> Dict[str, Any]:
        """
        RunPod serverless platformuna iş gönderir
        
        Args:
            job_id: Benzersiz iş tanımlayıcısı
            params: İş parametreleri
            priority: İş önceliği (RunPod'un desteklediği durumlarda)
            
        Returns:
            RunPod'dan dönen yanıt
        """
        try:
            # RunPod formatına uygun input
            input_data = {
                "input": params,
                "id": job_id
            }
            
            # Opsiyonel öncelik parametresi
            if priority > 0:
                input_data["priority"] = priority
                
            return runpod.serverless.job.submit(input_data)
        except Exception as e:
            logger.error(f"Error publishing job to RunPod: {str(e)}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        RunPod'dan iş durumunu sorgular
        
        Args:
            job_id: İş tanımlayıcısı
            
        Returns:
            İş durum bilgileri
        """
        try:
            return runpod.serverless.job.status(job_id)
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return None
    
    async def get_queue_length(self) -> int:
        """
        Mevcut kuyruk uzunluğunu döndürür
        
        Returns:
            Kuyruk uzunluğu
        """
        try:
            jobs = runpod.serverless.jobs.get_jobs()
            return len(jobs) if jobs else 0
        except Exception as e:
            logger.error(f"Error getting queue length: {str(e)}")
            return 0