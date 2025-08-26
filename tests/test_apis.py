import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_mta_api():
    """Test MTA API key"""
    key = os.getenv('MTA_API_KEY')
    url = f"http://bustime.mta.info/api/siri/vehicle-monitoring.json?key={key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ MTA API key working")
            return True
        else:
            print(f"❌ MTA API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MTA API error: {e}")
        return False

def test_weather_api():
    """Test OpenWeatherMap API key"""
    key = os.getenv('OPENWEATHER_API_KEY')
    url = f"http://api.openweathermap.org/data/2.5/weather?q=New York&appid={key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ OpenWeatherMap API key working")
            return True
        else:
            print(f"❌ Weather API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return False

def test_nyc_open_data():
    """Test NYC Open Data (no key required)"""
    url = "https://data.cityofnewyork.us/resource/btm5-ppia.json?$limit=1"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ NYC Open Data accessible")
            return True
        else:
            print(f"❌ NYC Open Data error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ NYC Open Data error: {e}")
        return False

def test_google_maps_api():
    """Test Google Maps API key"""

    key = os.getenv('GOOGLE_MAPS_API_KEY')
    url = f"https://maps.googleapis.com/maps/api/js?key={key}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Google Maps API key working")
            return True
        else:
            print(f"❌ Google Maps API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Google Maps API error: {e}")
        return False

if __name__ == "__main__":
    print("Testing API keys...\n")
    
    test_mta_api()
    test_weather_api() 
    test_nyc_open_data()
    test_google_maps_api()
    
    print("\n✅ API testing complete!")