"""
Collect real-time subway data from MTA GTFS-RT feeds

Three feeds available:
- Vehicle Positions: Real-time train locations
- Trip Updates: Delay predictions and schedule updates  
- Service Alerts: Disruptions and service changes

Requires: pip install gtfs-realtime-bindings protobuf
"""

import requests
import polars as pl
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import time
from google.transit import gtfs_realtime_pb2
from google.protobuf.message import DecodeError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class MTASubwayDataCollector:
    def __init__(self):
        self.api_key = os.getenv('MTA_API_KEY')  # Optional for subway feeds
        
        # MTA GTFS-RT Feed URLs - Updated with correct endpoints
        self.feeds = {
            'ace': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-ace',
            'bdfm': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-bdfm', 
            'g': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-g',
            'jz': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-jz',
            'l': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-l',
            'nqrw': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-nqrw',
            '7': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-7',
            'sir': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-si',
            '456': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-456',
            '123456': 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr'
        }
        
        # Service alerts (applies to all routes)
        self.alerts_feed = 'https://api-endpoint.mta.info/Dataservice/mtagtfsrealtime/gtfsr-alerts'
        
        # Common subway routes for filtering
        self.subway_routes = {
            '1', '2', '3', '4', '5', '6', '6X', '7', '7X',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'L', 'M', 'N', 'Q', 'R', 'W', 'Z',
            'SIR'  # Staten Island Railway
        }

    def _make_request(self, feed_url: str) -> bytes:
        """Make request to MTA GTFS-RT feed (API key optional for subway feeds)"""
        headers = {}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        
        try:
            response = requests.get(feed_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {feed_url}: {e}")
            logger.info("Note: If you're getting 401/403 errors, you may need an API key from https://api.mta.info/")
            return None

    def collect_vehicle_positions(self, feed_group: str = '123456') -> pl.DataFrame:
        """Collect real-time train positions from specific feed group
        
        Args:
            feed_group: Feed to query ('ace', 'bdfm', 'g', 'jz', 'l', 'nqrw', '7', 'sir', '456', '123456')
        """
        
        logger.info(f"Collecting subway vehicle positions from {feed_group} feed...")
        
        if feed_group not in self.feeds:
            logger.error(f"Invalid feed group: {feed_group}. Available: {list(self.feeds.keys())}")
            return pl.DataFrame()
            
        feed_data = self._make_request(self.feeds[feed_group])
        
        if not feed_data:
            return pl.DataFrame()
            
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(feed_data)
            
            vehicles = []
            
            for entity in feed.entity:
                if entity.HasField('vehicle'):
                    vehicle = entity.vehicle
                    
                    # Extract route ID for filtering
                    route_id = vehicle.trip.route_id if vehicle.HasField('trip') else 'Unknown'
                    
                    vehicle_data = {
                        'timestamp': datetime.now(timezone.utc),
                        'vehicle_id': vehicle.vehicle.id if vehicle.HasField('vehicle') else entity.id,
                        'route_id': route_id,
                        'trip_id': vehicle.trip.trip_id if vehicle.HasField('trip') else None,
                        'direction_id': vehicle.trip.direction_id if vehicle.HasField('trip') else None,
                        'latitude': vehicle.position.latitude if vehicle.HasField('position') else None,
                        'longitude': vehicle.position.longitude if vehicle.HasField('position') else None,
                        'bearing': vehicle.position.bearing if vehicle.HasField('position') else None,
                        'speed': vehicle.position.speed if vehicle.HasField('position') else None,
                        'current_stop_sequence': vehicle.current_stop_sequence if vehicle.HasField('current_stop_sequence') else None,
                        'current_status': self._get_vehicle_status(vehicle.current_status) if vehicle.HasField('current_status') else 'Unknown',
                        'stop_id': vehicle.stop_id if vehicle.HasField('stop_id') else None,
                        'congestion_level': self._get_congestion_level(vehicle.congestion_level) if vehicle.HasField('congestion_level') else 'Unknown'
                    }
                    
                    vehicles.append(vehicle_data)
            
            logger.info(f"Collected {len(vehicles)} vehicle positions")
            return pl.DataFrame(vehicles)
            
        except DecodeError as e:
            logger.error(f"Failed to decode protobuf data: {e}")
            return pl.DataFrame()
        except Exception as e:
            logger.error(f"Error processing vehicle positions: {e}")
            return pl.DataFrame()

    def collect_trip_updates(self, feed_group: str = '123456') -> pl.DataFrame:
        """Collect trip updates with delay information from specific feed group"""
        
        logger.info(f"Collecting subway trip updates from {feed_group} feed...")
        
        if feed_group not in self.feeds:
            logger.error(f"Invalid feed group: {feed_group}. Available: {list(self.feeds.keys())}")
            return pl.DataFrame()
            
        feed_data = self._make_request(self.feeds[feed_group])
        
        if not feed_data:
            return pl.DataFrame()
            
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(feed_data)
            
            trip_updates = []
            
            for entity in feed.entity:
                if entity.HasField('trip_update'):
                    trip_update = entity.trip_update
                    route_id = trip_update.trip.route_id
                    
                    # Process each stop time update
                    for stop_time_update in trip_update.stop_time_update:
                        update_data = {
                            'timestamp': datetime.now(timezone.utc),
                            'trip_id': trip_update.trip.trip_id,
                            'route_id': route_id,
                            'direction_id': trip_update.trip.direction_id if trip_update.trip.HasField('direction_id') else None,
                            'start_date': trip_update.trip.start_date if trip_update.trip.HasField('start_date') else None,
                            'vehicle_id': trip_update.vehicle.id if trip_update.HasField('vehicle') else None,
                            'stop_sequence': stop_time_update.stop_sequence if stop_time_update.HasField('stop_sequence') else None,
                            'stop_id': stop_time_update.stop_id,
                            'arrival_delay': stop_time_update.arrival.delay if stop_time_update.HasField('arrival') and stop_time_update.arrival.HasField('delay') else None,
                            'arrival_time': datetime.fromtimestamp(stop_time_update.arrival.time, timezone.utc) if stop_time_update.HasField('arrival') and stop_time_update.arrival.HasField('time') else None,
                            'departure_delay': stop_time_update.departure.delay if stop_time_update.HasField('departure') and stop_time_update.departure.HasField('delay') else None,
                            'departure_time': datetime.fromtimestamp(stop_time_update.departure.time, timezone.utc) if stop_time_update.HasField('departure') and stop_time_update.departure.HasField('time') else None,
                            'schedule_relationship': self._get_schedule_relationship(stop_time_update.schedule_relationship) if stop_time_update.HasField('schedule_relationship') else 'SCHEDULED'
                        }
                        
                        trip_updates.append(update_data)
            
            logger.info(f"Collected {len(trip_updates)} trip updates")
            return pl.DataFrame(trip_updates)
            
        except DecodeError as e:
            logger.error(f"Failed to decode protobuf data: {e}")
            return pl.DataFrame()
        except Exception as e:
            logger.error(f"Error processing trip updates: {e}")
            return pl.DataFrame()

    def collect_service_alerts(self) -> pl.DataFrame:
        """Collect service alerts and disruptions"""
        
        logger.info("Collecting service alerts...")
        feed_data = self._make_request(self.alerts_feed)
        
        if not feed_data:
            return pl.DataFrame()
            
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(feed_data)
            
            alerts = []
            
            for entity in feed.entity:
                if entity.HasField('alert'):
                    alert = entity.alert
                    
                    # Extract affected routes
                    affected_routes = []
                    for selector in alert.informed_entity:
                        if selector.HasField('route_id'):
                            affected_routes.append(selector.route_id)
                    
                    alert_data = {
                        'timestamp': datetime.now(timezone.utc),
                        'alert_id': entity.id,
                        'cause': self._get_alert_cause(alert.cause) if alert.HasField('cause') else 'Unknown',
                        'effect': self._get_alert_effect(alert.effect) if alert.HasField('effect') else 'Unknown',
                        'severity_level': self._get_severity_level(alert.severity_level) if alert.HasField('severity_level') else 'Unknown',
                        'header_text': alert.header_text.translation[0].text if alert.header_text.translation else '',
                        'description_text': alert.description_text.translation[0].text if alert.description_text.translation else '',
                        'affected_routes': ','.join(affected_routes) if affected_routes else None,
                        'active_period_start': datetime.fromtimestamp(alert.active_period[0].start, timezone.utc) if alert.active_period and alert.active_period[0].HasField('start') else None,
                        'active_period_end': datetime.fromtimestamp(alert.active_period[0].end, timezone.utc) if alert.active_period and alert.active_period[0].HasField('end') else None
                    }
                    
                    alerts.append(alert_data)
            
            logger.info(f"Collected {len(alerts)} service alerts")
            return pl.DataFrame(alerts)
            
        except DecodeError as e:
            logger.error(f"Failed to decode protobuf data: {e}")
            return pl.DataFrame()
        except Exception as e:
            logger.error(f"Error processing service alerts: {e}")
            return pl.DataFrame()

    def _get_vehicle_status(self, status):
        """Convert vehicle status enum to readable string"""
        status_map = {
            0: 'INCOMING_AT',
            1: 'STOPPED_AT', 
            2: 'IN_TRANSIT_TO'
        }
        return status_map.get(status, 'UNKNOWN')

    def _get_congestion_level(self, level):
        """Convert congestion level enum to readable string"""
        congestion_map = {
            0: 'UNKNOWN_CONGESTION_LEVEL',
            1: 'RUNNING_SMOOTHLY',
            2: 'STOP_AND_GO',
            3: 'CONGESTION',
            4: 'SEVERE_CONGESTION'
        }
        return congestion_map.get(level, 'UNKNOWN')

    def _get_schedule_relationship(self, relationship):
        """Convert schedule relationship enum to readable string"""
        relationship_map = {
            0: 'SCHEDULED',
            1: 'SKIPPED',
            2: 'NO_DATA',
            3: 'UNSCHEDULED'
        }
        return relationship_map.get(relationship, 'SCHEDULED')

    def _get_alert_cause(self, cause):
        """Convert alert cause enum to readable string"""
        cause_map = {
            1: 'UNKNOWN_CAUSE',
            2: 'OTHER_CAUSE',
            3: 'TECHNICAL_PROBLEM',
            4: 'STRIKE',
            5: 'DEMONSTRATION',
            6: 'ACCIDENT',
            7: 'HOLIDAY',
            8: 'WEATHER',
            9: 'MAINTENANCE',
            10: 'CONSTRUCTION',
            11: 'POLICE_ACTIVITY',
            12: 'MEDICAL_EMERGENCY'
        }
        return cause_map.get(cause, 'UNKNOWN_CAUSE')

    def _get_alert_effect(self, effect):
        """Convert alert effect enum to readable string"""
        effect_map = {
            1: 'NO_SERVICE',
            2: 'REDUCED_SERVICE', 
            3: 'SIGNIFICANT_DELAYS',
            4: 'DETOUR',
            5: 'ADDITIONAL_SERVICE',
            6: 'MODIFIED_SERVICE',
            7: 'OTHER_EFFECT',
            8: 'UNKNOWN_EFFECT',
            9: 'STOP_MOVED',
            10: 'NO_EFFECT',
            11: 'ACCESSIBILITY_ISSUE'
        }
        return effect_map.get(effect, 'UNKNOWN_EFFECT')

    def _get_severity_level(self, level):
        """Convert severity level enum to readable string"""
        severity_map = {
            1: 'UNKNOWN_SEVERITY',
            2: 'INFO',
            3: 'WARNING', 
            4: 'SEVERE'
        }
        return severity_map.get(level, 'UNKNOWN_SEVERITY')

    def save_data(self, df: pl.DataFrame, data_type: str, route_filter: str = "all"):
        """Save data to CSV with appropriate naming"""
        if df.is_empty():
            logger.warning(f"No {data_type} data to save")
            return
        
        # Create filename with data type and date
        today = datetime.now().strftime("%Y%m%d")
        route_suffix = f"_{route_filter}" if route_filter != "all" else ""
        filename = f"data/raw/mta_subway_{data_type}{route_suffix}_{today}.csv"
        
        os.makedirs("data/raw", exist_ok=True)
        
        # Append to existing file or create new one
        if os.path.exists(filename):
            with open(filename, mode="ab") as f:
                df.write_csv(f, include_header=False)
        else:
            df.write_csv(filename)
        
        logger.info(f"Saved {len(df)} {data_type} records to {filename}")

    def collect_all_data(self, feed_groups: list = None):
        """Collect all types of subway data from specified feed groups
        
        Args:
            feed_groups: List of feed groups to collect from. If None, collects from all feeds.
        """
        
        if feed_groups is None:
            feed_groups = list(self.feeds.keys())
        
        logger.info(f"Starting comprehensive subway data collection from feeds: {feed_groups}")
        
        all_vehicles = []
        all_trips = []
        
        for feed_group in feed_groups:
            logger.info(f"Collecting from {feed_group} feed...")
            
            # Collect vehicle positions
            vehicles_df = self.collect_vehicle_positions(feed_group)
            if not vehicles_df.is_empty():
                all_vehicles.append(vehicles_df)
            
            # Small delay to avoid overwhelming the API
            time.sleep(1)
            
            # Collect trip updates
            trips_df = self.collect_trip_updates(feed_group)
            if not trips_df.is_empty():
                all_trips.append(trips_df)
            
            time.sleep(1)
        
        # Combine all data
        vehicles_combined = pl.concat(all_vehicles) if all_vehicles else pl.DataFrame()
        trips_combined = pl.concat(all_trips) if all_trips else pl.DataFrame()
        
        # Collect service alerts once (applies to all routes)
        alerts_df = self.collect_service_alerts()
        
        # Save combined data
        if not vehicles_combined.is_empty():
            self.save_data(vehicles_combined, "vehicles", "all_feeds")
        
        if not trips_combined.is_empty():
            self.save_data(trips_combined, "trip_updates", "all_feeds")
        
        if not alerts_df.is_empty():
            self.save_data(alerts_df, "alerts")
        
        logger.info("Subway data collection complete!")
        
        return {
            'vehicles': vehicles_combined,
            'trip_updates': trips_combined,
            'service_alerts': alerts_df
        }


# Usage example
if __name__ == "__main__":
    collector = MTASubwayDataCollector()
    
    # Test with specific feed groups first
    test_feeds = ['l', '456']  # L train and 4/5/6 lines
    
    logger.info(f"Testing subway data collection for feeds: {test_feeds}")
    
    # Collect all data types
    data = collector.collect_all_data(feed_groups=test_feeds)
    
    # Display summary
    for data_type, df in data.items():
        if not df.is_empty():
            print(f"\n{data_type.upper()} Summary:")
            print(f"Records collected: {len(df)}")
            print(f"Columns: {df.columns}")
            if len(df) > 0:
                print(f"Routes found: {sorted(df['route_id'].unique().to_list()) if 'route_id' in df.columns else 'N/A'}")
            print(f"Sample data:")
            print(df.head(2))
        else:
            print(f"\n{data_type.upper()}: No data collected")