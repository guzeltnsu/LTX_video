import base64
import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_runpod_endpoint():
    ENDPOINT = "YOUR_RUNPOD_ENDPOINT"
    API_KEY = "YOUR_API_KEY"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": "a red car driving on a sunny road",
            "num_frames": 8,
            "height": 256,
            "width": 256,
            "num_inference_steps": 20,
            "fps": 8
        }
    }
    
    # Submit job
    response = requests.post(
        f"{ENDPOINT}/run",
        json=payload,
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error submitting job: {response.text}")
        return
        
    job_id = response.json()["id"]
    print(f"Job submitted successfully. ID: {job_id}")
    
    # Poll for results
    while True:
        status_response = requests.get(
            f"{ENDPOINT}/status/{job_id}",
            headers=headers
        )
        
        status = status_response.json()
        print(f"Job status: {status['status']}")
        
        if status["status"] == "COMPLETED":
            # Save output video
            video_data = base64.b64decode(status["output"]["video"])
            with open("output.mp4", "wb") as f:
                f.write(video_data)
            print("Video saved as output.mp4")
            break
        elif status["status"] == "FAILED":
            print(f"Job failed: {status.get('error')}")
            break
            
        time.sleep(5)

if __name__ == "__main__":
    test_runpod_endpoint()