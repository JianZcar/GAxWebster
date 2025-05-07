import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def safe_get(element, attr, default=0.0):
    """Robust XML attribute extraction"""
    try:
        if element is None:
            return default
        value = element.get(attr)
        return float(value) if value not in [None, ''] else default
    except (TypeError, ValueError, AttributeError):
        return default

def parse_xml(xml_file):
    """Complete XML parsing with all required fields"""
    data = []
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag == 'tripinfo':
            emissions = elem.find('emissions')
            entry = {
                'id': elem.get('id'),
                'depart': safe_get(elem, 'depart'),
                'arrival': safe_get(elem, 'arrival'),
                'duration': safe_get(elem, 'duration'),
                'route': elem.get('id').split('.')[0],
                'CO2_abs': safe_get(emissions, 'CO2_abs'),
                'fuel_abs': safe_get(emissions, 'fuel_abs'),
                'NOx_abs': safe_get(emissions, 'NOx_abs'),
                'waitingTime': safe_get(elem, 'waitingTime'),
                'timeLoss': safe_get(elem, 'timeLoss')
            }
            data.append(entry)
            elem.clear()
    return pd.DataFrame(data)

# Load and validate data
try:
    df = parse_xml('tripinfos.xml')
    print("Data Validation:")
    print(f"Total entries: {len(df)}")
    print(f"Non-zero CO₂ entries: {df[df['CO2_abs'] > 0].shape[0]}")
    print("\nSample Data:")
    print(df[['id', 'CO2_abs', 'fuel_abs']].head())
except Exception as e:
    print(f"Error: {e}")
    exit()

# Visualization setup
plt.figure(figsize=(15, 10))
sns.set_style("whitegrid")

# 1. CO₂ Distribution Plot
plt.subplot(2, 2, 1)
sns.histplot(df['CO2_abs'], bins=50, kde=True, color='darkgreen')
plt.title('CO₂ Emissions Distribution')
plt.xlabel('CO₂ Emissions (mg)')

# 2. Temporal Emission Analysis
plt.subplot(2, 2, 2)
valid_co2 = df[df['CO2_abs'] > 0]
sns.scatterplot(x='depart', y='CO2_abs', data=valid_co2,
                hue=valid_co2['CO2_abs'], palette="viridis",
                size=valid_co2['fuel_abs'], sizes=(20, 200))
plt.title('CO₂ Emissions Over Time')
plt.xlabel('Departure Time (s)')
plt.ylabel('CO₂ Emissions (mg)')

# 3. Route Comparison
plt.subplot(2, 2, 3)
sns.boxplot(x='route', y='CO2_abs', data=valid_co2)
plt.title('Route-wise CO₂ Emission Comparison')
plt.xticks(rotation=45)

# 4. Emission-Time Relationship
plt.subplot(2, 2, 4)
sns.regplot(x='duration', y='CO2_abs', data=valid_co2,
            scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
plt.title('Trip Duration vs CO₂ Emissions')
plt.xlabel('Trip Duration (s)')
plt.ylabel('CO₂ Emissions (mg)')

plt.tight_layout()
plt.savefig('co2_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
