import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MeterReading:
    def __init__(self, timestamp, kwh):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)

class Building:
    def __init__(self, name):
        self.name = name
        self.meter_readings = []
    
    def add_reading(self, reading):
        self.meter_readings.append(reading)
    
    def calculate_total_consumption(self):
        return sum(r.kwh for r in self.meter_readings)
    
    def generate_report(self):
        if not self.meter_readings:
            return pd.DataFrame()
        df = pd.DataFrame([{'timestamp': r.timestamp, 'kwh': r.kwh} for r in self.meter_readings])
        return df.groupby(df.timestamp.dt.date)['kwh'].sum()

class BuildingManager:
    def __init__(self):
        self.buildings = {}
    
    def add_building_data(self, building_name, readings_df):
        building = Building(building_name)
        for _, row in readings_df.iterrows():
            if pd.notna(row['kwh']) and pd.notna(row['timestamp']):
                building.add_reading(MeterReading(row['timestamp'], row['kwh']))
        self.buildings[building_name] = building
    
    def get_summary(self):
        summary = {}
        for name, building in self.buildings.items():
            total = building.calculate_total_consumption()
            summary[name] = {
                'total_kwh': total,
                'mean_kwh': total / len(building.meter_readings) if building.meter_readings else 0,
                'count_readings': len(building.meter_readings)
            }
        return pd.DataFrame(summary).T

def calculate_daily_totals(df):
    """Calculate daily totals by building"""
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    return df.groupby(['building', 'date'])['kwh'].sum().reset_index()

def calculate_weekly_aggregates(df):
    """Calculate weekly aggregates by building"""
    df = df.copy()
    df['week'] = df['timestamp'].dt.isocalendar().week
    df['year'] = df['timestamp'].dt.year
    return df.groupby(['building', 'year', 'week'])['kwh'].sum().reset_index()

def building_wise_summary(df):
    """Summary statistics per building"""
    return df.groupby('building')['kwh'].agg(['mean', 'min', 'max', 'sum']).round(2)

# Task 1: Data Ingestion and Validation
def ingest_data(data_dir='data'):
    """Read all CSV files from data directory and combine"""
    data_path = Path(data_dir)
    df_combined = pd.DataFrame()
    
    if not data_path.exists():
        logging.warning(f"Data directory {data_dir} not found. Creating empty DataFrame.")
        return df_combined, []
    
    loaded_files = []
    for csv_file in data_path.glob('*.csv'):
        try:
            building_name = csv_file.stem.replace('_', ' ').title()
            logging.info(f"Loading {building_name} from {csv_file.name}")
            
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            
            # Validate required columns
            required_cols = ['timestamp', 'kwh']
            if not all(col in df.columns for col in required_cols):
                logging.warning(f"{building_name}: Missing required columns. Expected {required_cols}")
                continue
            
            # Clean and prepare data
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['kwh'] = pd.to_numeric(df['kwh'], errors='coerce')
            df['building'] = building_name
            
            # Drop invalid rows
            df = df.dropna(subset=['timestamp', 'kwh'])
            
            df_combined = pd.concat([df_combined, df], ignore_index=True)
            loaded_files.append(csv_file.name)
            logging.info(f"Successfully loaded {len(df)} readings from {building_name}")
            
        except Exception as e:
            logging.error(f"Error loading {csv_file.name}: {str(e)}")
    
    logging.info(f"Combined {len(df_combined)} total readings from {len(loaded_files)} files")
    return df_combined, loaded_files

# Main execution
def main():
    print("=== Campus Energy Dashboard Pipeline ===\n")
    
    # Task 1: Data Ingestion
    print("📊 Task 1: Ingesting and validating data...")
    df_combined, loaded_files = ingest_data()
    
    if df_combined.empty:
        print("❌ No data found. Please add CSV files to /data/ directory.")
        print("Expected format: timestamp,kwh")
        return
    
    # Set datetime index for time-series operations
    df_combined.set_index('timestamp', inplace=True)
    
    # Task 2: Core Aggregation Logic
    print("\n📈 Task 2: Calculating aggregations...")
    daily_totals = calculate_daily_totals(df_combined.reset_index())
    weekly_aggregates = calculate_weekly_aggregates(df_combined.reset_index())
    building_summary = building_wise_summary(df_combined.reset_index())
    
    print(f"✅ Aggregations complete:")
    print(f"   - Daily totals: {len(daily_totals)} records")
    print(f"   - Weekly aggregates: {len(weekly_aggregates)} records")
    print(f"   - Building summaries: {len(building_summary)} buildings")
    
    # Task 3: Object-Oriented Modeling
    print("\n🏗️ Task 3: Building OOP models...")
    manager = BuildingManager()
    for building_name in df_combined['building'].unique():
        building_data = df_combined[df_combined['building'] == building_name].reset_index()
        manager.add_building_data(building_name, building_data)
    
    oop_summary = manager.get_summary()
    
    # Task 4: Visual Output
    print("\n📊 Task 4: Generating dashboard...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Daily consumption trends
    for building in daily_totals['building'].unique():
        building_data = daily_totals[daily_totals['building'] == building]
        ax1.plot(building_data['date'], building_data['kwh'], marker='o', label=building, linewidth=2)
    ax1.set_title('Daily Consumption Trends', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('kWh')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average weekly usage comparison
    weekly_avg = weekly_aggregates.groupby('building')['kwh'].mean()
    colors = plt.cm.Set3(range(len(weekly_avg)))
    bars = ax2.bar(weekly_avg.index, weekly_avg.values, color=colors, alpha=0.8)
    ax2.set_title('Average Weekly Usage by Building', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average kWh/Week')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, weekly_avg.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Peak hourly consumption
    hourly_peaks = df_combined.groupby([df_combined.index.hour, 'building'])['kwh'].sum().unstack(fill_value=0)
    for building in hourly_peaks.columns:
        ax3.plot(hourly_peaks.index, hourly_peaks[building], marker='o', linewidth=2, label=building)
    ax3.set_title('Hourly Peak Consumption', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Peak kWh')
    ax3.set_xticks(range(0, 24, 2))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Dashboard saved as 'dashboard.png'")
    
    # Task 5: Persistence and Executive Summary
    print("\n💾 Task 5: Exporting data and generating report...")
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Export cleaned data
    df_combined.reset_index().to_csv(output_dir / 'cleaned_energy_data.csv', index=False)
    
    # Export summaries
    building_summary.to_csv(output_dir / 'building_summary.csv')
    oop_summary.to_csv(output_dir / 'oop_building_summary.csv')
    
    # Generate executive summary
    total_campus = df_combined['kwh'].sum()
    top_building = building_summary['sum'].idxmax()
    top_consumption = building_summary.loc[top_building, 'sum']
    peak_hour = df_combined.groupby(df_combined.index.hour)['kwh'].sum().idxmax()
    
    summary_content = f"""CAMPUS ENERGY SUMMARY REPORT
{'='*40}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY METRICS:
• Total Campus Consumption: {total_campus:,.0f} kWh
• Highest-Consuming Building: {top_building} ({top_consumption:,.0f} kWh)
• Peak Load Hour: {peak_hour}:00 (highest daily usage)

BUILDING SUMMARY:
{building_summary[['mean', 'sum']].round(1).to_string()}

DATA SOURCES:
• Loaded {len(loaded_files)} files: {', '.join(loaded_files)}
• Total valid readings: {len(df_combined):,}

Visualizations: dashboard.png
Detailed data: output/cleaned_energy_data.csv
"""
    
    with open(output_dir / 'executive_summary.txt', 'w') as f:
        f.write(summary_content)
    
    print("✅ All outputs saved to /output/ directory")
    print("\n🎉 Pipeline complete! Check dashboard.png and output/ folder.")
    
    # Print key stats to console
    print(f"\n🏆 QUICK STATS:")
    print(f"   Total Consumption: {total_campus:,.0f} kWh")
    print(f"   Top Building: {top_building}")
    print(f"   Peak Hour: {peak_hour}:00")

if __name__ == "__main__":
    main()
