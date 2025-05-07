import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def safe_get(element, attr, default=0.0):
    """Enhanced safe XML attribute extraction"""
    try:
        if element is None:
            return default
        value = element.get(attr)
        return float(value) if value not in [None, ''] else default
    except (TypeError, ValueError, AttributeError):
        return default

# Parse XML with explicit closing check
def parse_xml(xml_file):
    data = []
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag == 'tripinfo':
            emissions = elem.find('emissions')
            entry = {
                'id': elem.get('id'),
                'CO2_abs': safe_get(emissions, 'CO2_abs'),
                # Add other fields as needed
            }
            data.append(entry)
            elem.clear()
    return pd.DataFrame(data)

# Load data with XML validation
try:
    df = parse_xml('tripinfo.xml')
    print("Data Validation:")
    print(f"Total entries: {len(df)}")
    print(f"Non-zero CO₂ entries: {df[df['CO2_abs'] > 0].shape[0]}")
    print(df[['id', 'CO2_abs']].head(10))
except ET.ParseError as e:
    print(f"XML Error: {e}")
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
