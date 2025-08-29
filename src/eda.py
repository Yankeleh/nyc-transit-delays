"""
NYC Transit Delay Analysis - Exploratory Data Analysis & Regression Framework

Combines MTA historical data with weather data for comprehensive analysis
Includes visualization tools and baseline regression models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical modeling
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class TransitDelayAnalyzer:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.mta_monthly = None
        self.mta_daily = None  
        self.weather_monthly = None
        self.weather_daily = None
        self.combined_monthly = None
        self.models = {}
        
        # Set up plotting style
        self.setup_plotting()
    
    def setup_plotting(self):
        """Configure plotting aesthetics"""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        sns.set_palette("husl")
    
    def load_processed_data(self):
        """Load all processed datasets"""
        print("🔄 Loading processed datasets...")
        
        # Find most recent files
        try:
            # MTA monthly data
            mta_monthly_files = list(self.data_dir.glob("mta_monthly_master_*.csv"))
            if mta_monthly_files:
                latest_mta_monthly = sorted(mta_monthly_files)[-1]
                self.mta_monthly = pd.read_csv(latest_mta_monthly)
                self.mta_monthly['date'] = pd.to_datetime(self.mta_monthly['date'])
                print(f"✅ Loaded MTA monthly data: {len(self.mta_monthly)} records from {latest_mta_monthly}")
            
            # MTA daily data
            mta_daily_files = list(self.data_dir.glob("mta_daily_ridership_*.csv"))
            if mta_daily_files:
                latest_mta_daily = sorted(mta_daily_files)[-1]
                self.mta_daily = pd.read_csv(latest_mta_daily)
                self.mta_daily['date'] = pd.to_datetime(self.mta_daily['date'])
                print(f"✅ Loaded MTA daily data: {len(self.mta_daily)} records from {latest_mta_daily}")
            
            # Weather monthly data
            weather_monthly_files = list(self.data_dir.glob("noaa_weather_monthly_*.csv"))
            if weather_monthly_files:
                latest_weather_monthly = sorted(weather_monthly_files)[-1]
                self.weather_monthly = pd.read_csv(latest_weather_monthly)
                print(f"✅ Loaded weather monthly data: {len(self.weather_monthly)} records from {latest_weather_monthly}")
            
            # Weather daily data
            weather_daily_files = list(self.data_dir.glob("noaa_weather_daily_*.csv"))
            if weather_daily_files:
                latest_weather_daily = sorted(weather_daily_files)[-1]
                self.weather_daily = pd.read_csv(latest_weather_daily)
                self.weather_daily['date'] = pd.to_datetime(self.weather_daily['date'])
                print(f"✅ Loaded weather daily data: {len(self.weather_daily)} records from {latest_weather_daily}")
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def merge_datasets(self):
        """Merge MTA and weather data for analysis"""
        print("🔗 Merging datasets...")
        
        if self.mta_monthly is None or self.weather_monthly is None:
            print("❌ Required datasets not loaded")
            return False
        
        # Merge monthly datasets on year_month
        try:
            # Ensure year_month is consistent
            self.mta_monthly['year_month'] = pd.to_datetime(self.mta_monthly['year_month'].astype(str))
            self.weather_monthly['year_month'] = pd.to_datetime(self.weather_monthly['year_month'].astype(str))
            
            # Merge on year_month
            self.combined_monthly = pd.merge(
                self.mta_monthly, 
                self.weather_monthly,
                on='year_month',
                how='inner',
                suffixes=('_mta', '_weather')
            )
            
            print(f"✅ Created combined monthly dataset: {len(self.combined_monthly)} records")
            print(f"   Date range: {self.combined_monthly['year_month'].min()} to {self.combined_monthly['year_month'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return False
    
    def explore_data_quality(self):
        """Analyze data quality and completeness"""
        print("\n" + "="*60)
        print("📊 DATA QUALITY ANALYSIS")
        print("="*60)
        
        datasets = {
            'MTA Monthly': self.mta_monthly,
            'MTA Daily': self.mta_daily,
            'Weather Monthly': self.weather_monthly,
            'Weather Daily': self.weather_daily,
            'Combined Monthly': self.combined_monthly
        }
        
        for name, df in datasets.items():
            if df is not None:
                print(f"\n📋 {name}")
                print(f"   Shape: {df.shape}")
                
                # Missing data
                missing = df.isnull().sum()
                missing_pct = (missing / len(df)) * 100
                missing_summary = missing[missing > 0]
                
                if len(missing_summary) > 0:
                    print("   Missing data:")
                    for col, count in missing_summary.head(10).items():
                        print(f"     {col}: {count} ({missing_pct[col]:.1f}%)")
                else:
                    print("   ✅ No missing data")
                
                # Date coverage
                if 'date' in df.columns:
                    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
                    
                    # Check for gaps in daily data
                    if 'daily' in name.lower() and len(df) > 1:
                        date_diff = df['date'].diff().dt.days
                        gaps = date_diff[date_diff > 1]
                        if len(gaps) > 0:
                            print(f"   ⚠️  Date gaps found: {len(gaps)} gaps")
                        else:
                            print("   ✅ No date gaps")
    
    def create_target_variables(self):
        """Create delay/performance target variables from the datasets"""
        print("🎯 Creating target variables...")
        
        if self.combined_monthly is None:
            print("❌ Combined dataset not available")
            return
        
        # Look for delay-related columns in the data
        delay_cols = [col for col in self.combined_monthly.columns 
                     if any(keyword in col.lower() 
                           for keyword in ['delay', 'late', 'time', 'performance', 'reliability'])]
        
        print(f"   Found potential delay/performance columns: {delay_cols}")
        
        # Create derived performance metrics
        numeric_cols = self.combined_monthly.select_dtypes(include=[np.number]).columns
        
        # If we have ridership data, create efficiency metrics
        ridership_cols = [col for col in numeric_cols if 'ridership' in col.lower()]
        if ridership_cols:
            # Create ridership volatility (higher = less predictable service)
            self.combined_monthly['ridership_volatility'] = (
                self.combined_monthly[ridership_cols].std(axis=1)
            )
        
        # Weather severity score
        weather_severity_components = []
        
        if 'HEAVY_RAIN_sum' in self.combined_monthly.columns:
            weather_severity_components.append('HEAVY_RAIN_sum')
        if 'SNOW_EVENT_sum' in self.combined_monthly.columns:
            weather_severity_components.append('SNOW_EVENT_sum')
        if 'HIGH_WIND_sum' in self.combined_monthly.columns:
            weather_severity_components.append('HIGH_WIND_sum')
        
        if weather_severity_components:
            # Normalize and combine weather severity indicators
            weather_df = self.combined_monthly[weather_severity_components].fillna(0)
            scaler = StandardScaler()
            weather_normalized = scaler.fit_transform(weather_df)
            self.combined_monthly['weather_severity_score'] = weather_normalized.mean(axis=1)
        
        print(f"   Created derived variables: weather_severity_score, ridership_volatility")
    
    def plot_time_series_overview(self):
        """Create overview time series plots"""
        print("📈 Creating time series visualizations...")
        
        if self.combined_monthly is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NYC Transit & Weather Time Series Overview', fontsize=16, y=0.98)
        
        # Plot 1: Temperature trends
        if 'TAVG_mean' in self.combined_monthly.columns:
            axes[0,0].plot(self.combined_monthly['year_month'], 
                          self.combined_monthly['TAVG_mean'], 
                          color='red', alpha=0.7)
            axes[0,0].set_title('Average Temperature Over Time')
            axes[0,0].set_ylabel('Temperature (°F)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Precipitation patterns
        if 'PRCP_sum' in self.combined_monthly.columns:
            axes[0,1].bar(self.combined_monthly['year_month'], 
                         self.combined_monthly['PRCP_sum'], 
                         color='blue', alpha=0.6, width=20)
            axes[0,1].set_title('Monthly Precipitation')
            axes[0,1].set_ylabel('Precipitation (inches)')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Weather severity score
        if 'weather_severity_score' in self.combined_monthly.columns:
            axes[1,0].plot(self.combined_monthly['year_month'], 
                          self.combined_monthly['weather_severity_score'], 
                          color='orange', marker='o', markersize=3)
            axes[1,0].set_title('Weather Severity Score')
            axes[1,0].set_ylabel('Severity Score (normalized)')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Ridership trends (if available)
        ridership_cols = [col for col in self.combined_monthly.columns if 'ridership' in col.lower()]
        if ridership_cols:
            main_ridership = ridership_cols[0]  # Use first ridership column
            axes[1,1].plot(self.combined_monthly['year_month'], 
                          self.combined_monthly[main_ridership], 
                          color='green', alpha=0.7)
            axes[1,1].set_title('Transit Ridership Trends')
            axes[1,1].set_ylabel('Ridership')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_weather_transit_correlation(self):
        """Analyze correlations between weather and transit metrics"""
        print("🔍 Analyzing weather-transit correlations...")
        
        if self.combined_monthly is None:
            return
        
        # Identify weather and transit columns
        weather_cols = [col for col in self.combined_monthly.columns 
                       if any(w in col for w in ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'WIND', 'weather_severity'])]
        
        transit_cols = [col for col in self.combined_monthly.columns 
                       if any(t in col.lower() for t in ['ridership', 'delay', 'performance', 'reliability'])]
        
        if len(weather_cols) > 0 and len(transit_cols) > 0:
            # Create correlation matrix
            corr_cols = weather_cols + transit_cols
            corr_matrix = self.combined_monthly[corr_cols].corr()
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                       center=0, fmt='.2f', linewidths=0.5)
            plt.title('Weather-Transit Correlation Matrix')
            plt.tight_layout()
            plt.show()
            
            # Print strongest correlations
            print("\n🔥 Strongest Weather-Transit Correlations:")
            
            # Get cross-correlations between weather and transit
            weather_transit_corr = corr_matrix.loc[weather_cols, transit_cols]
            
            # Flatten and sort
            corr_pairs = []
            for weather_var in weather_cols:
                for transit_var in transit_cols:
                    corr_val = weather_transit_corr.loc[weather_var, transit_var]
                    if not pd.isna(corr_val):
                        corr_pairs.append((weather_var, transit_var, abs(corr_val), corr_val))
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for weather_var, transit_var, abs_corr, corr in corr_pairs[:10]:
                direction = "📈" if corr > 0 else "📉"
                print(f"   {direction} {weather_var} ↔ {transit_var}: {corr:.3f}")
        
        else:
            print("❌ Insufficient weather or transit columns for correlation analysis")
    
    def build_baseline_models(self):
        """Build baseline regression models"""
        print("🤖 Building baseline regression models...")
        
        if self.combined_monthly is None:
            print("❌ Combined dataset not available")
            return
        
        # Identify potential target variables
        target_candidates = [col for col in self.combined_monthly.columns 
                           if any(keyword in col.lower() 
                                 for keyword in ['ridership', 'delay', 'performance', 
                                               'reliability', 'volatility'])]
        
        if not target_candidates:
            print("❌ No suitable target variables found")
            return
        
        print(f"   Target variable candidates: {target_candidates}")
        
        # Use the first suitable target
        target_col = target_candidates[0]
        print(f"   Using target variable: {target_col}")
        
        # Prepare features
        feature_cols = []
        
        # Weather features
        weather_features = [col for col in self.combined_monthly.columns 
                          if any(w in col for w in ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'SNOW', 
                                                   'WIND', 'weather_severity'])]
        feature_cols.extend(weather_features)
        
        # Temporal features
        self.combined_monthly['month'] = pd.to_datetime(self.combined_monthly['year_month']).dt.month
        self.combined_monthly['year'] = pd.to_datetime(self.combined_monthly['year_month']).dt.year
        self.combined_monthly['quarter'] = pd.to_datetime(self.combined_monthly['year_month']).dt.quarter
        
        feature_cols.extend(['month', 'year', 'quarter'])
        
        # Categorical features (if any)
        cat_cols = [col for col in self.combined_monthly.columns 
                   if self.combined_monthly[col].dtype == 'object' and col != target_col]
        
        for col in cat_cols[:3]:  # Limit to first 3 categorical columns
            if self.combined_monthly[col].nunique() < 20:  # Only if not too many categories
                le = LabelEncoder()
                self.combined_monthly[f'{col}_encoded'] = le.fit_transform(self.combined_monthly[col].fillna('unknown'))
                feature_cols.append(f'{col}_encoded')
        
        # Filter to available features and remove any remaining categorical
        available_features = []
        for col in feature_cols:
            if col in self.combined_monthly.columns:
                if self.combined_monthly[col].dtype in ['int64', 'float64']:
                    available_features.append(col)
        
        print(f"   Using features: {available_features[:10]}{'...' if len(available_features) > 10 else ''}")
        print(f"   Total features: {len(available_features)}")
        
        # Prepare data
        X = self.combined_monthly[available_features].fillna(0)
        y = self.combined_monthly[target_col].fillna(y.median())
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n   Training {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if 'Forest' in name:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"     RMSE: {rmse:.4f}")
            print(f"     MAE:  {mae:.4f}")
            print(f"     R²:   {r2:.4f}")
        
        self.models = results
        self.feature_names = available_features
        self.target_name = target_col
        
        # Plot model comparison
        self.plot_model_comparison()
        
        return results
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        if not self.models:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance metrics comparison
        model_names = list(self.models.keys())
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        r2_scores = [self.models[name]['r2'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        
        axes[0].bar(x_pos, rmse_scores, color='lightcoral', alpha=0.7)
        axes[0].set_title('Model RMSE Comparison')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('RMSE')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(model_names, rotation=45)
        
        # R² comparison
        axes[1].bar(x_pos, r2_scores, color='lightblue', alpha=0.7)
        axes[1].set_title('Model R² Comparison')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('R² Score')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(model_names, rotation=45)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Actual vs Predicted for best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        best_model_results = self.models[best_model_name]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(best_model_results['y_test'], best_model_results['y_pred'], 
                   alpha=0.6, color='blue')
        
        # Perfect prediction line
        min_val = min(best_model_results['y_test'].min(), best_model_results['y_pred'].min())
        max_val = max(best_model_results['y_test'].max(), best_model_results['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel(f'Actual {self.target_name}')
        plt.ylabel(f'Predicted {self.target_name}')
        plt.title(f'Actual vs Predicted: {best_model_name}\n(R² = {best_model_results["r2"]:.3f})')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance from the models"""
        if not self.models or not hasattr(self, 'feature_names'):
            print("❌ No models available for feature importance analysis")
            return
        
        print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\n🌲 Random Forest Feature Importance (Top 10):")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'], color='lightgreen')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
    
    def generate_insights_summary(self):
        """Generate summary of key insights"""
        print("\n" + "="*60)
        print("🎯 KEY INSIGHTS SUMMARY")
        print("="*60)
        
        if self.combined_monthly is not None:
            print(f"📊 Dataset Overview:")
            print(f"   • Total monthly records: {len(self.combined_monthly)}")
            print(f"   • Date range: {self.combined_monthly['year_month'].min()} to {self.combined_monthly['year_month'].max()}")
            
            # Weather patterns
            if 'weather_severity_score' in self.combined_monthly.columns:
                avg_severity = self.combined_monthly['weather_severity_score'].mean()
                max_severity = self.combined_monthly['weather_severity_score'].max()
                print(f"\n🌦️  Weather Patterns:")
                print(f"   • Average weather severity: {avg_severity:.3f}")
                print(f"   • Maximum weather severity: {max_severity:.3f}")
                
                # Find months with extreme weather
                extreme_weather = self.combined_monthly.nlargest(3, 'weather_severity_score')[['year_month', 'weather_severity_score']]
                print(f"   • Most severe weather months:")
                for _, row in extreme_weather.iterrows():
                    print(f"     - {row['year_month'].strftime('%Y-%m')}: {row['weather_severity_score']:.3f}")
        
        if self.models:
            print(f"\n🤖 Model Performance:")
            best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
            best_r2 = self.models[best_model_name]['r2']
            print(f"   • Best model: {best_model_name} (R² = {best_r2:.3f})")
            
            if best_r2 > 0.7:
                print("   • ✅ Strong predictive performance")
            elif best_r2 > 0.4:
                print("   • ⚠️  Moderate predictive performance")
            else:
                print("   • ❌ Weak predictive performance - consider more features")
        
        print(f"\n💡 Recommendations:")
        print(f"   • Collect more granular delay/performance data")
        print(f"   • Add route-specific analysis")
        print(f"   • Include special events data (holidays, construction)")
        print(f"   • Consider hourly weather-transit relationships")
        print(f"   • Implement real-time prediction pipeline")


def main():
    """Main analysis workflow"""
    print("🚇 NYC Transit Delay Analysis - EDA & Regression")
    print("="*60)
    
    # Initialize analyzer
    analyzer = TransitDelayAnalyzer()
    
    # Load data
    if not analyzer.load_processed_data():
        print("❌ Failed to load processed data")
        return
    
    # Merge datasets
    if not analyzer.merge_datasets():
        print("❌ Failed to merge datasets")
        return
    
    # Data quality analysis
    analyzer.explore_data_quality()
    
    # Create target variables
    analyzer.create_target_variables()
    
    # Time series visualization
    analyzer.plot_time_series_overview()
    
    # Correlation analysis
    analyzer.analyze_weather_transit_correlation()
    
    # Build baseline models
    analyzer.build_baseline_models()
    
    # Feature importance
    analyzer.feature_importance_analysis()
    
    # Summary insights
    analyzer.generate_insights_summary()
    
    print(f"\n✅ Analysis complete!")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()