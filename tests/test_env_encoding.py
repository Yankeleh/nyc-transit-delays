# test_env_file.py
import os
from dotenv import load_dotenv

def check_env_file():
    try:
        # Try loading with explicit encoding
        load_dotenv(encoding='utf-8')
        print("✅ .env file loaded successfully")
        
        # Check if keys exist
        mta_key = os.getenv('MTA_API_KEY')
        weather_key = os.getenv('OPENWEATHER_API_KEY')
        
        print(f"MTA key exists: {mta_key is not None}")
        print(f"Weather key exists: {weather_key is not None}")
        
        if mta_key:
            print(f"MTA key length: {len(mta_key)}")
            # Check for hidden characters
            print(f"MTA key repr: {repr(mta_key[:20])}...")
            
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")

check_env_file()