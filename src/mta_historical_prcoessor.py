"""
Process MTA Historical Datasets for Transit Delay Analysis

Handles:
- MTA Subway Customer Journey-Focused Metrics (2022-2024) - Monthly
- MTA Bus Customer Journey-Focused Metrics (2022-2024) - Monthly  
- MTA Daily Ridership Data (2022-2024) - Daily

Filters data to 2022-2024 timeframe and creates analysis-ready datasets
Uses Polars for efficient data processing
"""

import polars as pl
import numpy as np
from datetime import datetime, date
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MTAHistoricalProcessor:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Date filtering range
        self.start_date = date(2022, 1, 1)
        self.end_date = date(2024, 12, 31)
        
        print(f"📅 Filtering data to: {self.start_date} to {self.end_date}")
        
    def load_subway_metrics(self, filepath):
        """Load and process subway customer journey metrics (monthly)"""
        print("Loading subway customer journey metrics...")
        
        try:
            # Load with Polars
            df = pl.read_csv(filepath)
            
            print(f"   Raw subway data: {df.shape[0]} records, {df.shape[1]} columns")
            
            # Standardize column names
            df = df.rename({col: col.lower().replace(' ', '_').replace('-', '_') 
                           for col in df.columns})
            
            # Delete unwanted columns
            columns_to_drop = ['division', 'total_apt', 'total_att', 'over_five_mins', 'over_five_mins_perc']
            existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
            if existing_cols_to_drop:
                df = df.drop(existing_cols_to_drop)

            # Rename columns
            rename_dict = {
                'additional_platform_time': 'avg_additional_stop_time',
                'additional_train_time': 'avg_additional_travel_time',
                'line' : 'route_id'
            }
            # Only rename columns that exist
            existing_renames = {old: new for old, new in rename_dict.items() if old in df.columns}
            if existing_renames:
                df = df.rename(existing_renames)

            # Find date column
            date_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['date', 'month', 'period', 'beginning'])]
            
            if not date_cols:
                print(f"❌ No date column found in subway data")
                return None
                
            date_col = date_cols[0]
            print(f"   Using date column: {date_col}")
            
            # Parse dates and filter
            df = df.with_columns([
                pl.col(date_col).str.to_date(strict=False).alias('date')
            ]).filter(
                (pl.col('date') >= self.start_date) & 
                (pl.col('date') <= self.end_date)
            )
            
            # Add derived date columns
            df = df.with_columns([
                pl.col('date').dt.year().alias('year'),
                pl.col('date').dt.month().alias('month'),
                pl.col('date').dt.strftime('%Y-%m').alias('year_month'),
                pl.lit('subway_metrics').alias('data_source'),
                pl.lit('monthly').alias('frequency'),
                pl.lit('subway').alias('transit_mode')
            ])
            
            print(f"   Processed subway data: {df.shape[0]} records")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading subway metrics: {e}")
            return None
    
    def load_bus_metrics(self, filepath):
        """Load and process bus customer journey metrics (monthly)"""
        print("Loading bus customer journey metrics...")
        
        try:
            df = pl.read_csv(filepath)
            
            print(f"   Raw bus data: {df.shape[0]} records, {df.shape[1]} columns")
            
            # Standardize column names
            df = df.rename({col: col.lower().replace(' ', '_').replace('-', '_') 
                           for col in df.columns})
            
            # Delete unwanted columns
            columns_to_drop = ['borough', 'trip_type']
            existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
            if existing_cols_to_drop:
                df = df.drop(existing_cols_to_drop)

            # Rename columns
            rename_dict = {
                'number_of_customers': 'num_passengers',
                'additional_bus_stop_time': 'avg_additional_stop_time',
                'additional_travel_time': 'avg_additional_travel_time'
            }

            # Only rename columns that exist
            existing_renames = {old: new for old, new in rename_dict.items() if old in df.columns}
            if existing_renames:
                df = df.rename(existing_renames)
            
            # Find date column
            date_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['date', 'month', 'period', 'beginning'])]
            
            if not date_cols:
                print(f"❌ No date column found in bus data")
                return None
                
            date_col = date_cols[0]
            print(f"   Using date column: {date_col}")
            
            # Parse dates and filter
            df = df.with_columns([
                pl.col(date_col).str.to_date(strict=False).alias('date')
            ]).filter(
                (pl.col('date') >= self.start_date) & 
                (pl.col('date') <= self.end_date)
            )
            
            # Add derived date columns
            df = df.with_columns([
                pl.col('date').dt.year().alias('year'),
                pl.col('date').dt.month().alias('month'),
                pl.col('date').dt.strftime('%Y-%m').alias('year_month'),
                pl.lit('bus_metrics').alias('data_source'),
                pl.lit('monthly').alias('frequency'),
                pl.lit('bus').alias('transit_mode')
            ])
            
            print(f"   Processed bus data: {df.shape[0]} records")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading bus metrics: {e}")
            return None
    
    """
    def load_daily_ridership(self, filepath):
        #Load and process daily ridership data
        print("Loading daily ridership data...")
        
        try:
            df = pl.read_csv(filepath)
            
            print(f"   Raw ridership data: {df.shape[0]} records, {df.shape[1]} columns")
            
            # Standardize column names
            df = df.rename({col: col.lower().replace(' ', '_').replace('-', '_') 
                           for col in df.columns})
            
            # Find date column
            date_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['date', 'day', 'timestamp'])]
            
            if not date_cols:
                print(f"❌ No date column found in ridership data")
                return None
                
            date_col = date_cols[0]
            print(f"   Using date column: {date_col}")
            
            # Parse dates and filter
            df = df.with_columns([
                pl.col(date_col).str.to_date(strict=False).alias('date')
            ]).filter(
                (pl.col('date') >= self.start_date) & 
                (pl.col('date') <= self.end_date)
            )
            
            # Add derived date columns
            df = df.with_columns([
                pl.col('date').dt.year().alias('year'),
                pl.col('date').dt.month().alias('month'),
                pl.col('date').dt.weekday().alias('day_of_week'),
                pl.col('date').dt.strftime('%A').alias('day_name'),
                (pl.col('date').dt.weekday() >= 6).alias('is_weekend'),
                pl.col('date').dt.strftime('%Y-%m').alias('year_month'),
                pl.lit('daily_ridership').alias('data_source'),
                pl.lit('daily').alias('frequency'),
                pl.lit('combined').alias('transit_mode')
            ])
            
            print(f"   Processed ridership data: {df.shape[0]} records")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading daily ridership: {e}")
            return None
    
    def aggregate_daily_to_monthly(self, daily_df):
        #Aggregate daily ridership to monthly for consistency with other datasets
        print("Aggregating daily data to monthly...")
        
        if daily_df is None or daily_df.shape[0] == 0:
            return None
        
        # Identify ridership columns (numeric columns excluding metadata)
        exclude_cols = ['date', 'year', 'month', 'day_of_week', 'day_name', 
                       'is_weekend', 'year_month', 'data_source', 'frequency', 'transit_mode']
        
        numeric_cols = []
        for col in daily_df.columns:
            if col not in exclude_cols:
                if daily_df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                    numeric_cols.append(col)
        
        print(f"   Aggregating {len(numeric_cols)} numeric columns")
        
        # Build aggregation expressions
        agg_exprs = []
        
        # Aggregate ridership columns
        for col in numeric_cols:
            agg_exprs.extend([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).sum().alias(f"{col}_sum"), 
                pl.col(col).std().alias(f"{col}_std")
            ])
        
        # Add other aggregations
        agg_exprs.extend([
            pl.col('is_weekend').mean().alias('weekend_proportion'),
            pl.col('date').min().alias('date_min'),
            pl.col('date').max().alias('date_max'),
            pl.col('year').first().alias('year'),
            pl.col('month').first().alias('month'),
            pl.col('year_month').first().alias('year_month')
        ])
        
        # Group by year_month and aggregate
        monthly_agg = daily_df.group_by('year_month').agg(agg_exprs)
        
        # Add metadata
        monthly_agg = monthly_agg.with_columns([
            pl.lit('daily_ridership_monthly_agg').alias('data_source'),
            pl.lit('monthly').alias('frequency'),
            pl.lit('combined').alias('transit_mode')
        ])
        
        print(f"   Created monthly aggregation: {monthly_agg.shape[0]} records")
        
        return monthly_agg
        """
    
    
    def create_master_dataset(self, subway_df=None, bus_df=None, ridership_df=None):
        """Combine all datasets into master analysis dataset"""
        print("Creating master dataset...")
        
        monthly_datasets = []
        
        # Add subway metrics
        if subway_df is not None:
            monthly_datasets.append(subway_df)
            print(f"   Added subway metrics: {subway_df.shape[0]} records")
        
        # Add bus metrics  
        if bus_df is not None:
            monthly_datasets.append(bus_df)
            print(f"   Added bus metrics: {bus_df.shape[0]} records")
        
        """
        # Process ridership data
        ridership_monthly = None
        if ridership_df is not None:
            ridership_monthly = self.aggregate_daily_to_monthly(ridership_df)
            if ridership_monthly is not None:
                monthly_datasets.append(ridership_monthly)
                print(f"   Added ridership monthly agg: {ridership_monthly.shape[0]} records")
        """
        
        # Combine monthly datasets
        monthly_master = None
        if monthly_datasets:
            # Get all unique columns across datasets
            all_columns = set()
            for df in monthly_datasets:
                all_columns.update(df.columns)
            
            # Ensure all datasets have the same columns (fill missing with nulls)
            aligned_datasets = []
            for df in monthly_datasets:
                missing_cols = all_columns - set(df.columns)
                if missing_cols:
                    # Add missing columns as null
                    df = df.with_columns([
                        pl.lit(None).alias(col) for col in missing_cols
                    ])
                aligned_datasets.append(df)
            
            # Concatenate
            monthly_master = pl.concat(aligned_datasets, how='vertical')
            
            print(f"   Master monthly dataset: {monthly_master.shape[0]} records")
            
            # Show transit mode breakdown
            mode_counts = monthly_master.group_by('transit_mode').len()
            print(f"   Transit modes:")
            for row in mode_counts.iter_rows():
                print(f"     {row[0]}: {row[1]} records")
        
        return monthly_master, ridership_df
    
    def save_processed_data(self, monthly_df, daily_df=None):
        """Save processed datasets"""
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Save monthly master dataset
        if monthly_df is not None:
            monthly_file = output_dir / f"mta_monthly_master_{datetime.now().strftime('%Y%m%d')}.csv"
            monthly_df.write_csv(monthly_file)
            saved_files.append(monthly_file)
            print(f"✅ Saved monthly master dataset: {monthly_file}")
        
        # Save daily dataset
        if daily_df is not None:
            daily_file = output_dir / f"mta_daily_ridership_{datetime.now().strftime('%Y%m%d')}.csv"
            daily_df.write_csv(daily_file)
            saved_files.append(daily_file)
            print(f"✅ Saved daily ridership dataset: {daily_file}")
        
        return saved_files
    
    def generate_summary_stats(self, df):
        """Generate summary statistics for the dataset"""
        if df is None or df.shape[0] == 0:
            print("❌ No data to analyze")
            return
        
        print("\n" + "="*50)
        print("📊 DATASET SUMMARY")
        print("="*50)
        
        print(f"Total records: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")
        
        # Date range
        if 'date' in df.columns:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        elif 'year_month' in df.columns:
            print(f"Period range: {df['year_month'].min()} to {df['year_month'].max()}")
        
        # Transit modes
        if 'transit_mode' in df.columns:
            mode_counts = df.group_by('transit_mode').len().sort('len', descending=True)
            print(f"Transit modes:")
            for row in mode_counts.iter_rows():
                print(f"  {row[0]}: {row[1]} records")
        
        # Data sources
        if 'data_source' in df.columns:
            source_counts = df.group_by('data_source').len().sort('len', descending=True)
            print(f"Data sources:")
            for row in source_counts.iter_rows():
                print(f"  {row[0]}: {row[1]} records")
        
        # Numeric columns summary
        numeric_cols = []
        for col in df.columns:
            if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                numeric_cols.append(col)
        
        if numeric_cols:
            print(f"\nNumeric columns: {len(numeric_cols)}")
            
            # Sample statistics for first few numeric columns
            sample_cols = numeric_cols[:5]
            stats_df = df.select(sample_cols).describe()
            print(f"Sample statistics (first 5 numeric columns):")
            print(stats_df)
        
        # Missing data analysis
        print(f"\nMissing data analysis:")
        total_cells = df.shape[0] * df.shape[1]
        
        missing_summary = []
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                pct = (null_count / df.shape[0]) * 100
                missing_summary.append((col, null_count, pct))
        
        if missing_summary:
            missing_summary.sort(key=lambda x: x[2], reverse=True)  # Sort by percentage
            print(f"Columns with missing data:")
            for col, count, pct in missing_summary[:10]:  # Show top 10
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("✅ No missing data found")


def main():
    """Main processing function"""
    processor = MTAHistoricalProcessor()
    
    print("🚇 MTA Historical Data Processing")
    print("="*50)
    
    # File paths - updated as requested
    subway_file = "data/raw/mta_subway_metrics_20250829.csv"
    bus_file = "data/raw/mta_bus_metrics_20250829.csv"
    #ridership_file = "data/raw/mta_daily_ridership_20250829.csv"
    
    # Load datasets
    subway_df = None
    bus_df = None
    #ridership_df = None
    
    # Try to load each dataset
    if os.path.exists(subway_file):
        subway_df = processor.load_subway_metrics(subway_file)
    else:
        print(f"⚠️  Subway metrics file not found: {subway_file}")
    
    if os.path.exists(bus_file):
        bus_df = processor.load_bus_metrics(bus_file)
    else:
        print(f"⚠️  Bus metrics file not found: {bus_file}")
    

    """
    if os.path.exists(ridership_file):
        ridership_df = processor.load_daily_ridership(ridership_file)
    else:
        print(f"⚠️  Daily ridership file not found: {ridership_file}")
    """
    
    # Check if we have any data: add ridership_df if using ridership data
    if all(df is None for df in [subway_df, bus_df]):
        print("❌ No datasets could be loaded. Please check file paths.")
        return None, None
    
    # Process and combine data
    monthly_master, daily_master = processor.create_master_dataset(
        subway_df, bus_df, #ridership_df
    )
    
    if monthly_master is not None:
        # Generate summary statistics
        processor.generate_summary_stats(monthly_master)
        
        # Save processed data
        saved_files = processor.save_processed_data(monthly_master, daily_master)
        
        print(f"\n🎉 Processing complete!")
        for file in saved_files:
            print(f"📁 Saved: {file}")
            
        return monthly_master, daily_master
    
    else:
        print("❌ Failed to create master dataset")
        return None, None


if __name__ == "__main__":
    monthly_data, daily_data = main()