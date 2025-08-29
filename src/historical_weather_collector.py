"""
NOAA Weather Data Collector for NYC Transit Analysis

Downloads historical weather data from NOAA API for NYC area
Processes data to match MTA dataset timeframes (2020-2025)
"""

import requests
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

class NOAAWeatherCollector:
    def __init__(self, token=None):
        """
        Initialize NOAA data collector
        
        Get your free API token from: https://www.ncdc.noaa.gov/cdo-web/token
        """
        self.token = token or os.getenv('NOAA_API_TOKEN')
        self.base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        
        # NYC area weather stations
        self.nyc_stations = {
            'CENTRAL_PARK': 'GHCND:USW00094728',      # Central Park
            'JFK_AIRPORT': 'GHCND:USW00094789',       # JFK Airport
            'LAGUARDIA': 'GHCND:USW00014732',         # LaGuardia
            'BROOKLYN': 'GHCND:USW00094741',          # Brooklyn
            'STATEN_ISLAND': 'GHCND:USW00094745'      # Staten Island
        }
        
        # Weather variables we want
        self.weather_vars = [
            'TMAX',    # Maximum temperature
            'TMIN',    # Minimum temperature
            'TAVG',    # Average temperature
            'PRCP',    # Precipitation
            'SNOW',    # Snowfall
            'SNWD',    # Snow depth
            'AWND',    # Average wind speed
            'WSF2',    # Fastest 2-minute wind speed
            'PGTM',    # Peak gust time
        ]
        
        if not self.token:
            print("⚠️  NOAA API token not found. Get one from: https://www.ncdc.noaa.gov/cdo-web/token")
            print("   Set as NOAA_API_TOKEN environment variable or pass to constructor")
    
    def get_station_info(self):
        """Get information about NYC weather stations"""
        headers = {'token': self.token}
        
        for name, station_id in self.nyc_stations.items():
            try:
                url = f"{self.base_url}/stations/{station_id}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    station_data = response.json()
                    print(f"{name}: {station_data.get('name', 'Unknown')}")
                    print(f"  Location: {station_data.get('latitude', 'N/A')}, {station_data.get('longitude', 'N/A')}")
                    print(f"  Period: {station_data.get('mindate', 'N/A')} to {station_data.get('maxdate', 'N/A')}")
                else:
                    print(f"❌ Failed to get info for {name}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error getting station info for {name}: {e}")
            
            time.sleep(0.2)  # Rate limiting
    
    def fetch_weather_data(self, start_date, end_date, station='CENTRAL_PARK'):
        """
        Fetch weather data for specified date range
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format  
            station (str): Station name from self.nyc_stations
        """
        if not self.token:
            print("❌ NOAA API token required")
            return None
            
        headers = {'token': self.token}
        station_id = self.nyc_stations.get(station)
        
        if not station_id:
            print(f"❌ Station {station} not found")
            return None
        
        # NOAA API has limits, so we'll fetch in chunks if needed
        params = {
            'datasetid': 'GHCND',
            'stationid': station_id,
            'startdate': start_date,
            'enddate': end_date,
            'datatypeid': ','.join(self.weather_vars),
            'limit': 1000,  # Max records per request
            'units': 'standard'  # Standard units
        }
        
        print(f"Fetching weather data for {station} from {start_date} to {end_date}...")
        
        all_data = []
        offset = 1
        
        while True:
            params['offset'] = offset
            
            try:
                response = requests.get(f"{self.base_url}/data", 
                                      headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data and data['results']:
                        all_data.extend(data['results'])
                        
                        # Check if we got all data
                        if len(data['results']) < 1000:
                            break
                        else:
                            offset += 1000
                    else:
                        break
                        
                elif response.status_code == 429:
                    print("Rate limited, waiting...")
                    time.sleep(10)
                    continue
                    
                else:
                    print(f"❌ API Error {response.status_code}: {response.text}")
                    break
                    
            except Exception as e:
                print(f"❌ Request error: {e}")
                break
            
            time.sleep(0.2)  # Rate limiting
        
        if all_data:
            print(f"✅ Fetched {len(all_data)} weather records")
            return self._process_weather_data(all_data, station)
        else:
            print("❌ No weather data retrieved")
            return None
    
    def _process_weather_data(self, raw_data, station_name):
        """Process raw NOAA data into clean DataFrame"""
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        if df.empty:
            return None
        
        # Pivot data so each weather variable is a column
        df_pivot = df.pivot(index='date', columns='datatype', values='value').reset_index()
        
        # Clean up
        df_pivot['date'] = pd.to_datetime(df_pivot['date'])
        df_pivot['station'] = station_name
        df_pivot['year'] = df_pivot['date'].dt.year
        df_pivot['month'] = df_pivot['date'].dt.month
        df_pivot['day'] = df_pivot['date'].dt.day
        df_pivot['day_of_year'] = df_pivot['date'].dt.dayofyear
        df_pivot['day_of_week'] = df_pivot['date'].dt.dayofweek
        df_pivot['is_weekend'] = df_pivot['day_of_week'].isin([5, 6])
        df_pivot['year_month'] = df_pivot['date'].dt.to_period('M')
        
        # Convert temperature from tenths of Celsius to Fahrenheit if needed
        temp_cols = ['TMAX', 'TMIN', 'TAVG']
        for col in temp_cols:
            if col in df_pivot.columns:
                # NOAA temps are in tenths of degrees C
                df_pivot[col] = (df_pivot[col] / 10) * 9/5 + 32  # Convert to F
        
        # Convert precipitation from tenths of mm to inches
        if 'PRCP' in df_pivot.columns:
            df_pivot['PRCP'] = df_pivot['PRCP'] / 254  # tenths mm to inches
        
        # Add derived weather features
        if 'TMAX' in df_pivot.columns and 'TMIN' in df_pivot.columns:
            df_pivot['TEMP_RANGE'] = df_pivot['TMAX'] - df_pivot['TMIN']
        
        # Weather severity indicators
        if 'PRCP' in df_pivot.columns:
            df_pivot['HEAVY_RAIN'] = (df_pivot['PRCP'] > 0.5).astype(int)
            df_pivot['ANY_PRECIP'] = (df_pivot['PRCP'] > 0).astype(int)
        
        if 'SNOW' in df_pivot.columns:
            df_pivot['SNOW_EVENT'] = (df_pivot['SNOW'] > 0).astype(int)
        
        if 'AWND' in df_pivot.columns:
            df_pivot['HIGH_WIND'] = (df_pivot['AWND'] > 20).astype(int)  # >20 mph
        
        # Seasonal indicators
        df_pivot['SEASON'] = df_pivot['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        print(f"✅ Processed weather data: {len(df_pivot)} days")
        print(f"   Date range: {df_pivot['date'].min()} to {df_pivot['date'].max()}")
        print(f"   Variables: {[col for col in df_pivot.columns if col in self.weather_vars]}")
        
        return df_pivot
    
    def collect_historical_weather(self, start_year=2020, end_year=2025, station='CENTRAL_PARK'):
        """
        Collect weather data for multiple years
        
        Args:
            start_year (int): Starting year
            end_year (int): Ending year  
            station (str): Weather station name
        """
        all_weather_data = []
        
        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"
            
            # Don't go beyond current date
            if year == datetime.now().year:
                end_date = datetime.now().strftime("%Y-%m-%d")
            else:
                end_date = f"{year}-12-31"
            
            print(f"\n📅 Collecting weather data for {year}...")
            
            year_data = self.fetch_weather_data(start_date, end_date, station)
            
            if year_data is not None:
                all_weather_data.append(year_data)
            
            # Rate limiting between years
            time.sleep(1)
        
        if all_weather_data:
            # Combine all years
            combined_data = pd.concat(all_weather_data, ignore_index=True)
            combined_data = combined_data.sort_values('date').reset_index(drop=True)
            
            print(f"\n✅ Total weather data collected: {len(combined_data)} days")
            print(f"   Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            
            return combined_data
        
        return None
    
    def aggregate_to_monthly(self, daily_weather_df):
        """Aggregate daily weather data to monthly for consistency with MTA data"""
        
        if daily_weather_df is None or daily_weather_df.empty:
            return None
        
        print("Aggregating weather data to monthly...")
        
        # Define aggregation functions for each variable type
        agg_funcs = {
            'TMAX': ['mean', 'max', 'std'],
            'TMIN': ['mean', 'min', 'std'], 
            'TAVG': ['mean', 'std'],
            'TEMP_RANGE': ['mean', 'std'],
            'PRCP': ['sum', 'mean', 'max'],  # Total monthly precip
            'SNOW': ['sum', 'max'],
            'SNWD': ['mean', 'max'],
            'AWND': ['mean', 'max'],
            'WSF2': ['max'],
            'HEAVY_RAIN': 'sum',      # Days with heavy rain
            'ANY_PRECIP': 'sum',      # Days with any precipitation
            'SNOW_EVENT': 'sum',      # Days with snow
            'HIGH_WIND': 'sum',       # Days with high wind
            'is_weekend': 'mean',     # Proportion of weekend days
            'date': ['min', 'max'],   # Date range
            'station': 'first'
        }
        
        # Only aggregate columns that exist
        final_agg_funcs = {}
        for col, funcs in agg_funcs.items():
            if col in daily_weather_df.columns:
                final_agg_funcs[col] = funcs
        
        # Group by year_month and aggregate
        monthly_weather = daily_weather_df.groupby('year_month').agg(final_agg_funcs)
        
        # Flatten column names
        monthly_weather.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                  for col in monthly_weather.columns]
        
        monthly_weather = monthly_weather.reset_index()
        
        # Add metadata
        monthly_weather['frequency'] = 'monthly'
        monthly_weather['data_source'] = 'noaa_weather'
        
        print(f"✅ Created monthly weather aggregation: {len(monthly_weather)} records")
        
        return monthly_weather
    
    def save_weather_data(self, daily_df=None, monthly_df=None):
        """Save weather datasets"""
        
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        if daily_df is not None:
            daily_file = output_dir / f"noaa_weather_daily_{datetime.now().strftime('%Y%m%d')}.csv"
            daily_df.to_csv(daily_file, index=False)
            saved_files.append(daily_file)
            print(f"✅ Saved daily weather data: {daily_file}")
        
        if monthly_df is not None:
            monthly_file = output_dir / f"noaa_weather_monthly_{datetime.now().strftime('%Y%m%d')}.csv"
            monthly_df.to_csv(monthly_file, index=False)
            saved_files.append(monthly_file)
            print(f"✅ Saved monthly weather data: {monthly_file}")
        
        return saved_files


def main():
    """Main collection function"""
    
    # Initialize collector
    collector = NOAAWeatherCollector()
    
    if not collector.token:
        print("❌ Please set your NOAA API token")
        print("   Get one from: https://www.ncdc.noaa.gov/cdo-web/token")
        print("   Set as NOAA_API_TOKEN environment variable")
        return None, None
    
    # Show available stations
    print("🌡️  Available NYC Weather Stations:")
    collector.get_station_info()
    
    # Collect historical data (2020-2025)
    print(f"\n🌦️  Collecting historical weather data...")
    daily_weather = collector.collect_historical_weather(
        start_year=2020, 
        end_year=2025,
        station='CENTRAL_PARK'  # Change this if you prefer a different station
    )
    
    if daily_weather is not None:
        # Create monthly aggregation
        monthly_weather = collector.aggregate_to_monthly(daily_weather)
        
        # Save both datasets
        saved_files = collector.save_weather_data(daily_weather, monthly_weather)
        
        print(f"\n✅ Weather data collection complete!")
        for file in saved_files:
            print(f"📁 Saved: {file}")
        
        return daily_weather, monthly_weather
    
    else:
        print("❌ Failed to collect weather data")
        return None, None


if __name__ == "__main__":
    daily_weather, monthly_weather = main()