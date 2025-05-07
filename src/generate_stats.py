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
    except (TypeError, ValueError):
        return default

# Extract data with error handling
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

# Create DataFrame
df = pd.DataFrame(data)

# Set style
sns.set(style="whitegrid", palette="pastel")
plt.figure(figsize=(12, 8))

# Visualization 1: Time vs Duration with Emissions
plt.subplot(2, 2, 1)
scatter = sns.scatterplot(x='depart', y='duration', hue='CO2_abs', 
                         size='fuel_abs', data=df, palette='viridis')
plt.title('Trip Duration vs Departure Time\n(Color: CO₂ Emissions, Size: Fuel Consumption)')
plt.xlabel('Departure Time (s)')
plt.ylabel('Duration (s)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Visualization 2: Emissions Distribution
plt.subplot(2, 2, 2)
sns.histplot(df['CO2_abs'], kde=True, color='teal')
plt.title('Distribution of CO₂ Emissions')
plt.xlabel('CO₂ Emissions (mg)')
plt.ylabel('Number of Trips')

# Visualization 3: Fuel vs CO2 Emissions
plt.subplot(2, 2, 3)
sns.regplot(x='fuel_abs', y='CO2_abs', data=df, scatter_kws={'alpha':0.5})
plt.title('Fuel Consumption vs CO₂ Emissions')
plt.xlabel('Fuel Consumption (ml)')
plt.ylabel('CO₂ Emissions (mg)')

# Visualization 4: Route Performance
plt.subplot(2, 2, 4)
route_metrics = df.groupby('route').agg({
    'duration': 'mean',
    'timeLoss': 'mean',
    'CO2_abs': 'mean'
}).reset_index()

melted = route_metrics.melt(id_vars='route', 
                           value_vars=['duration', 'timeLoss', 'CO2_abs'],
                           var_name='metric')

sns.barplot(x='value', y='route', hue='metric', data=melted, orient='h')
plt.title('Route Performance Comparison')
plt.xlabel('Value')
plt.ylabel('Route')
plt.legend(title='Metric')

# Adjust layout and save
plt.tight_layout()
plt.savefig('trip_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
