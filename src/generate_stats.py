import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parse XML data
tree = ET.parse('tripinfo.xml')
root = tree.getroot()

def safe_get(element, attr, default=0.0):
    """Safe XML attribute extraction with error handling"""
    try:
        return float(element.get(attr)) if element is not None else default
    except (TypeError, ValueError, AttributeError):
        return default

# Extract data with enhanced validation
data = []
for trip in root.findall('tripinfo'):
    emissions = trip.find('emissions')
    entry = {
        'id': trip.get('id'),
        'depart': safe_get(trip, 'depart'),
        'arrival': safe_get(trip, 'arrival'),
        'duration': safe_get(trip, 'duration'),
        'route': trip.get('id').split('.')[0],
        'CO2_abs': safe_get(emissions, 'CO2_abs'),
        'fuel_abs': safe_get(emissions, 'fuel_abs'),
        'NOx_abs': safe_get(emissions, 'NOx_abs'),
        'waitingTime': safe_get(trip, 'waitingTime'),
        'timeLoss': safe_get(trip, 'timeLoss')
    }
    data.append(entry)

# Create DataFrame and verify data
df = pd.DataFrame(data)
print("CO₂ Data Summary:")
print(df['CO2_abs'].describe())

# Set visualization style
plt.figure(figsize=(16, 10))
sns.set_style("darkgrid")
palette = sns.color_palette("rocket_r", as_cmap=True)

# Visualization 1: Time-CO₂ Relationship (Enhanced)
plt.subplot(2, 2, 1)
sc = sns.scatterplot(x='depart', y='duration', hue='CO2_abs', size='fuel_abs',
                    data=df, palette=palette, sizes=(20, 200), edgecolor='black')
plt.title('Trip Duration vs Departure Time\nCO₂ Emissions Intensity', fontsize=14)
plt.xlabel('Departure Time (s)', fontsize=12)
plt.ylabel('Duration (s)', fontsize=12)
norm = plt.Normalize(df['CO2_abs'].min(), df['CO2_abs'].max())
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
cbar = plt.colorbar(sm)
cbar.set_label('CO₂ Emissions (mg)', rotation=270, labelpad=20)

# Visualization 2: CO₂ Distribution Analysis
plt.subplot(2, 2, 2)
sns.histplot(df['CO2_abs'], kde=True, color='darkgreen', bins=30)
plt.title('CO₂ Emissions Distribution', fontsize=14)
plt.xlabel('CO₂ Emissions (mg)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df['CO2_abs'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["CO2_abs"].mean():.2f} mg')
plt.legend()

# Visualization 3: Route-based CO₂ Analysis
plt.subplot(2, 2, 3)
route_co2 = df.groupby('route')['CO2_abs'].agg(['mean', 'std']).reset_index()
sns.barplot(x='mean', y='route', data=route_co2, palette='viridis', xerr=route_co2['std'])
plt.title('Average CO₂ Emissions by Route\nwith Standard Deviation', fontsize=14)
plt.xlabel('Average CO₂ Emissions (mg)', fontsize=12)
plt.ylabel('Route', fontsize=12)

# Visualization 4: Time Loss vs CO₂ Emissions
plt.subplot(2, 2, 4)
sns.regplot(x='timeLoss', y='CO2_abs', data=df, scatter_kws={'alpha':0.6}, 
          line_kws={'color': 'red'})
plt.title('Time Loss vs CO₂ Emissions', fontsize=14)
plt.xlabel('Time Loss (s)', fontsize=12)
plt.ylabel('CO₂ Emissions (mg)', fontsize=12)

# Final adjustments and save
plt.tight_layout()
plt.savefig('enhanced_emissions_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
