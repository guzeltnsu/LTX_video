import sys
import os
import torch
from handler import ModelHandler
import time

def basic_test():
    print("Starting basic test...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Device available (MPS): {torch.backends.mps.is_available()}")
    print(f"Device available (CUDA): {torch.cuda.is_available()}")
    
    # Test parametreleri - daha basit bir test için
    test_params = {
        "prompt": "a car",
        "num_frames": 4,  # Daha az frame
        "height": 128,    # Daha düşük çözünürlük
        "width": 128,     # Daha düşük çözünürlük
        "num_inference_steps": 10,  # Daha az adım
        "fps": 8
    }
    
    try:
        model_handler = ModelHandler()
        print("Model loaded successfully")
        
        output = model_handler.generate(test_params)
        print("Generation successful!")
        if output and hasattr(output, 'keys'):
            print("Output keys:", output.keys())
        else:
            print("Output type:", type(output))
        
        return True
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = basic_test()
    print(f"\nTest {'passed' if success else 'failed'}")