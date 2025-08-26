"""
Collect weather data from OpenWeatherMap API
Will run hourly to get conditions

To edit: get different conditions for different boroughs, will compare with different bus lines depending on boroughs.
"""
import requests
import polars as pl
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class WeatherDataCollector:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def collect_current_weather(self, city='Brooklyn'):
        """Get current weather conditions"""
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'imperial'  # Fahrenheit
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            else:
                print(f"Weather API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Weather collection error: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        """Convert API response to clean format"""
        
        weather_record = {
            'timestamp': datetime.now(),
            'temperature': raw_data['main']['temp'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'wind_speed': raw_data.get('wind', {}).get('speed', 0),
            'weather_main': raw_data['weather'][0]['main'],
            'weather_description': raw_data['weather'][0]['description'],
            'precipitation': raw_data.get('rain', {}).get('1h', 0),  # Rain in last hour
            'visibility': raw_data.get('visibility', 10000)  # Visibility in meters
        }
        
        return pl.DataFrame([weather_record])
    
    def save_data(self, df):
        """Append weather data to daily file"""
        if df.is_empty():
            return
            
        # One file per day
        today = datetime.now().strftime("%Y%m%d")
        filename = f"data/raw/weather_{today}.csv"
        
        os.makedirs("data/raw", exist_ok=True)
        
        # Append to existing file or create new one
        if os.path.exists(filename):
            with open(f"{filename}", mode="ab") as f:
                df.write_csv(f, include_header=False)
        else:
            df.write_csv(filename)
            
        print(f"Weather data saved to {filename}")

# Usage
if __name__ == "__main__":
    collector = WeatherDataCollector()
    weather_data = collector.collect_current_weather()
    print(weather_data)
    
    if weather_data is not None:
        collector.save_data(weather_data)