import requests
import json
import base64
import time

# Endpoint bilgileri
endpoint_id = "tp21056el6pt5l"  # Kendi endpoint ID'nizi girin
api_key = "rpa_9TKQRBCQXC9ZBF1AVO68YXA4YBZO6CQ1L3BA2RNL5sunrr"  # Kendi API anahtarınızı girin
endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

# API Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Test 1: Metin tabanlı video oluşturma
def test_text_to_video():
    payload = {
        "input": {
            "prompt": "A beautiful sunset over mountains with a lake reflecting the colors",
            "num_frames": 16,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "fps": 8
        }
    }
    
    print("🚀 Metin tabanlı video oluşturma isteği gönderiliyor...")
    response = requests.post(endpoint_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        job_id = result.get("id")
        print(f"✅ İstek kabul edildi. İş ID: {job_id}")
        
        # İşin tamamlanmasını bekle ve sonucu kontrol et
        wait_for_completion(job_id)
    else:
        print(f"❌ Hata: {response.status_code}")
        print(response.text)

# İş durumunu kontrol etme ve bekletme fonksiyonu
def wait_for_completion(job_id):
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    
    print("⏳ İş durumu kontrol ediliyor...")
    max_attempts = 30
    attempts = 0
    
    while attempts < max_attempts:
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()
        
        status = status_data.get("status")
        print(f"   Durum: {status}")
        
        if status == "COMPLETED":
            print("✅ İş tamamlandı!")
            handle_completed_job(status_data)
            break
        elif status in ["FAILED", "CANCELLED"]:
            print(f"❌ İş başarısız oldu: {status}")
            if "error" in status_data:
                print(f"   Hata: {status_data['error']}")
            break
        
        attempts += 1
        print("   5 saniye bekleniyor...")
        time.sleep(5)
    
    if attempts >= max_attempts:
        print("⚠️ Zaman aşımı: İş hala tamamlanmadı")

# Tamamlanmış iş sonucunu işleme
def handle_completed_job(status_data):
    if "output" in status_data and status_data["output"]:
        output = status_data["output"]
        
        if "video" in output:
            # Base64 video'yu kaydet
            video_base64 = output["video"]
            video_data = base64.b64decode(video_base64)
            
            output_file = "generated_video.mp4"
            with open(output_file, "wb") as f:
                f.write(video_data)
            print(f"📹 Video kaydedildi: {output_file}")
        else:
            print("⚠️ Çıktıda video bulunamadı")
    else:
        print("⚠️ Çıktı verisi bulunamadı")

# İşlemi başlat
if __name__ == "__main__":
    test_text_to_video()