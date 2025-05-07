import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def safe_get(element, attr, default=0.0):
    """Enhanced XML attribute extraction with validation"""
    try:
        if element is None or attr not in element.attrib:
            return default
        return float(element.get(attr))
    except (TypeError, ValueError, AttributeError):
        return default

def parse_xml(xml_file):
    """XML parsing with data validation"""
    data = []
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag == 'tripinfo':
            emissions = elem.find('emissions')
            entry = {
                'id': elem.get('id'),
                'depart': safe_get(elem, 'depart'),
                'duration': safe_get(elem, 'duration'),
                'route': elem.get('id').split('.')[0],
                'CO2': safe_get(emissions, 'CO2_abs'),
                'fuel': safe_get(emissions, 'fuel_abs'),
                'NOx': safe_get(emissions, 'NOx_abs'),
                'waiting': safe_get(elem, 'waitingTime'),
                'timeLoss': safe_get(elem, 'timeLoss'),
                'speedFactor': safe_get(elem, 'speedFactor')
            }
            data.append(entry)
            elem.clear()
    return pd.DataFrame(data)


df = parse_xml('tripinfo.xml')

# Visualization setup
plt.figure(figsize=(20, 24))
gs = GridSpec(4, 2, figure=plt.gcf())
sns.set_theme(style="whitegrid", palette="muted")

# 1. Emission Distribution Matrix
ax1 = plt.subplot(gs[0, :])
sns.violinplot(data=df[['CO2', 'fuel', 'NOx']].apply(lambda x: x/x.max()), 
              inner="quartile", palette="Blues")
plt.title('Normalized Emission Distribution', fontsize=14)
plt.ylabel('Normalized Values')

# 2. Time-Emission Relationship
ax2 = plt.subplot(gs[1, 0])
sc = sns.scatterplot(x='depart', y='CO2', hue='route', size='fuel',
                    data=df, palette="tab10", sizes=(10, 200), ax=ax2)
plt.title('CO₂ Emissions vs Departure Time', fontsize=12)
plt.xlabel('Departure Time (s)')

# 3. Speed Factor Impact
ax3 = plt.subplot(gs[1, 1])
sns.regplot(x='speedFactor', y='CO2', data=df, 
           scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax3)
plt.title('Speed Factor vs CO₂ Emissions', fontsize=12)
plt.xlabel('Speed Factor')

# 4. Route Performance Analysis
ax4 = plt.subplot(gs[2, 0])
route_stats = df.groupby('route').agg({
    'CO2': 'mean',
    'duration': 'median',
    'timeLoss': 'sum'
}).reset_index()
sns.heatmap(route_stats.set_index('route'), annot=True, fmt=".1f", 
           cmap="YlGnBu", ax=ax4)
plt.title('Route Performance Metrics', fontsize=12)

# 5. Time Loss Composition
ax5 = plt.subplot(gs[2, 1])
time_components = df[['duration', 'waiting', 'timeLoss']]
time_components.columns = ['Driving', 'Waiting', 'Time Loss']
sns.boxplot(data=time_components, palette="Set2", ax=ax5)
plt.title('Time Component Distribution', fontsize=12)
plt.ylabel('Seconds')

# 6. Emission Correlation Matrix
ax6 = plt.subplot(gs[3, :])
corr_matrix = df[['CO2', 'fuel', 'NOx', 'duration', 'timeLoss']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax6)
plt.title('Emission Correlation Matrix', fontsize=14)

plt.tight_layout()
plt.savefig('detailed_emission_analysis.png', dpi=300, bbox_inches='tight')
plt.close()  # Prevent display in CI/CD environments

print("Visualization saved as detailed_emission_analysis.png")
