"""
Collect real-time bus position data from MTA API

To add: train position and time collection
To run periodically
"""

import requests
import polars as pl
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import time

load_dotenv()

class MTABusDataCollector:
    def __init__(self):
        self.api_key = os.getenv('MTA_API_KEY')
        self.base_url = "http://bustime.mta.info/api/siri/vehicle-monitoring.json"


    def collect_bus_positions(self, route_id="B6") -> pl.DataFrame:
        #Collect current bus positions for specific route
        
        params = {
            'key': self.api_key,
            'VehicleMonitoringDetailLevel': 'calls',
            'LineRef': route_id
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_bus_data(data, route_id)
            else:
                print(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Collection error: {e}")
            return None
        

    def _parse_bus_data(self, raw_data, route_id):
        #Convert API Response into DataFrame
        
        buses=[]

        try:
            vehicles = raw_data['Siri']['ServiceDelivery'][0]['VehicleActivity']
            for vehicle in vehicles:
                bus_info = {
                    'timestamp': datetime.now(),
                    'route_id': route_id,
                    'vehicle_id': vehicle['MonitoredVehicleJourney']['VehicleRef'],
                    'longitude': vehicle['MonitoredVehicleJourney']['VehicleLocation']['Longitude'],
                    'destination': vehicle['MonitoredVehicleJourney']['DestinationName'],
                    'next_stop': vehicle['MonitoredVehicleJourney']['MonitoredCall']['StopPointName'],
                    'delay_seconds': self._calculate_delay(vehicle)
                }

            return pl.DataFrame(buses)

        except KeyError as e:
            print(f"Data parsing error: {e}")
            return pl.DataFrame()
        

    def _calculate_delay(self,vehicle):
        #Calculate vehicle delay

        try:
            expected = vehicle['MonitoredVehicleJourney']['MonitoredCall']['ExpectedArrivalTime']
            aimed = vehicle['MonitoredVehicleJourney']['MonitoredCall']['AimedArrivalTime']
            delay = aimed-expected
            return delay
        
        except:
            return None
        
    def save_data(self, df: pl.DataFrame, route_id):
        #Save data to csv

        if df.is_empty:
            print("No data to save")
            return
        
        # Create filename each day
        today = datetime.now().strftime("%Y%m%d")
        filename= f"data/raw/mta_buses_{route_id}_{today}.csv"

        os.makedirs("data/raw", exist_ok=True)

        df.write_csv(filename)
        print(f"Saved {len(df)} records to {filename}")


# Usage example

if __name__ == "__main__":
    collector = MTABusDataCollector()

    # Will do this for al routes eventually
    routes = ['B6', 'B9', 'B11']

    for route in routes:
        print(f"Collecting data for route {route}...")
        bus_data = collector.collect_bus_positions(route)

        if bus_data is not None:
            collector.save_data(bus_data, route)

        #API rate limit
        time.sleep(2)