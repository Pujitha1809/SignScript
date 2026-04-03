import zipfile
import json
import os

original_model = r"model\sign_model_savedmodel.keras"
fixed_model = r"model\sign_model_fixed.keras"

print(f"Cleaning model: {original_model}...")

with zipfile.ZipFile(original_model, 'r') as zin:
    with zipfile.ZipFile(fixed_model, 'w') as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            
            # Target the architecture config file
            if item.filename == "config.json":
                config_data = json.loads(content.decode('utf-8'))
                
                # Recursively clean up incompatible version keys
                def clean_keys(d):
                    if isinstance(d, dict):
                        # 1. Remove Keras 2 key that breaks Keras 3
                        d.pop("quantization_config", None)
                        
                        # 2. Convert Keras 3 keys back to Keras 2 format
                        if "batch_shape" in d:
                            d["batch_input_shape"] = d.pop("batch_shape")
                        d.pop("optional", None)
                        
                        # 3. Simplify DType dictionaries to strings if present
                        if "dtype" in d and isinstance(d["dtype"], dict):
                            if "config" in d["dtype"] and "name" in d["dtype"]["config"]:
                                d["dtype"] = d["dtype"]["config"]["name"]
                                
                        for k, v in d.items():
                            clean_keys(v)
                    elif isinstance(d, list):
                        for item in d:
                            clean_keys(item)
                
                clean_keys(config_data)
                content = json.dumps(config_data).encode('utf-8')
            
            zout.writestr(item, content)

print(f"Success! Fixed model saved as: {fixed_model}")