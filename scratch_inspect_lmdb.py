import lmdb
import struct
import numpy as np
import cv2
from pathlib import Path

def inspect_lmdb():
    lmdb_path = "data/DocTamperV1-TrainingSet" 
    
    print(f"Opening LMDB at {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin() as txn:
        # Check num_samples
        num_samples_bytes = txn.get(b'num_samples')
        if num_samples_bytes:
            # Struct pack could be >I or <I etc
            num_samples = int.from_bytes(num_samples_bytes, byteorder='big')
            try:
                num_samples2 = int.from_bytes(num_samples_bytes, byteorder='little')
            except:
                pass
        
        # Read the first few keys
        print("\n--- Inspecting first 20 samples ---")
        
        sample_id = 0
        valid_samples = 0
        
        while valid_samples < 20 and sample_id < 1000:
            key_img = f"image-{sample_id:09d}".encode('utf-8')
            key_lbl = f"label-{sample_id:09d}".encode('utf-8')
            
            img_data = txn.get(key_img)
            lbl_data = txn.get(key_lbl)
            
            if img_data is not None:
                is_tampered = False
                label_val = "N/A"
                
                if lbl_data is not None:
                    # Try to read as PNG/image
                    np_arr = np.frombuffer(lbl_data, np.uint8)
                    mask = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        max_val = np.max(mask)
                        label_val = f"Mask (max={max_val})"
                        if max_val > 0:
                            is_tampered = True
                    else:
                        label_val = "Invalid Mask Data"
                else:
                    label_val = "No Mask Found"
                    is_tampered = False
                    
                status = "Tampered" if is_tampered else "Authentic"
                print(f"Sample ID: {sample_id} | Raw Label: {label_val} | Class: {status}")
                valid_samples += 1
            
            sample_id += 1

if __name__ == "__main__":
    inspect_lmdb()
