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
    """XML parsing with traffic data focus"""
    data = []
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag == 'tripinfo':
            entry = {
                'id': elem.get('id'),
                'depart': safe_get(elem, 'depart'),
                'arrival': safe_get(elem, 'arrival'),
                'duration': safe_get(elem, 'duration'),
                'waiting': safe_get(elem, 'waitingTime'),
                'timeLoss': safe_get(elem, 'timeLoss'),
                'route': elem.get('id').split('.')[0],
                'speedFactor': safe_get(elem, 'speedFactor'),
                'departLane': elem.get('departLane', ''),
                'arrivalLane': elem.get('arrivalLane', ''),
                'departSpeed': safe_get(elem, 'departSpeed'),
                'arrivalSpeed': safe_get(elem, 'arrivalSpeed'),
                'routeLength': safe_get(elem, 'routeLength'),
                'vType': elem.get('vType', 'car')
            }
            # Calculate derived metrics
            entry['actualTravelTime'] = entry['duration'] - entry['waiting'] - entry['timeLoss']
            entry['avgSpeed'] = entry['routeLength'] / entry['actualTravelTime'] if entry['actualTravelTime'] > 0 else 0
            data.append(entry)
            elem.clear()
    return pd.DataFrame(data)

# Data processing
try:
    df = parse_xml('tripinfo.xml')
    print("Data Summary:")
    print(df[['duration', 'waiting', 'timeLoss', 'avgSpeed']].describe())
except Exception as e:
    print(f"Error: {str(e)}")
    exit()

# Visualization setup
plt.figure(figsize=(24, 28))
gs = GridSpec(4, 3, figure=plt.gcf())
sns.set_theme(style="whitegrid", palette="pastel")

# 1. Temporal Traffic Patterns
ax1 = plt.subplot(gs[0, :])
df['hour'] = df['depart'] // 3600
hourly_traffic = df.groupby('hour').size()
sns.lineplot(x=hourly_traffic.index, y=hourly_traffic.values, ax=ax1)
plt.title('Hourly Departure Patterns', fontsize=14)
plt.xlabel('Hour of Day')
plt.ylabel('Number of Vehicles')

# 2. Route Performance Analysis
ax2 = plt.subplot(gs[1, 0])
sns.boxplot(x='route', y='duration', data=df, ax=ax2)
plt.title('Trip Duration by Route', fontsize=12)
plt.xticks(rotation=45)

# 3. Speed Analysis
ax3 = plt.subplot(gs[1, 1])
sns.scatterplot(x='speedFactor', y='timeLoss', hue='route', 
               data=df, ax=ax3, alpha=0.6)
plt.title('Speed Factor vs Time Loss', fontsize=12)
plt.xlabel('Speed Factor')
plt.ylabel('Time Loss (s)')

# 4. Lane Utilization
ax4 = plt.subplot(gs[1, 2])
lane_counts = df['departLane'].value_counts().head(10)
sns.barplot(x=lane_counts.values, y=lane_counts.index, ax=ax4)
plt.title('Top 10 Busiest Departure Lanes', fontsize=12)
plt.xlabel('Vehicle Count')

# 5. Time Component Breakdown
ax5 = plt.subplot(gs[2, 0])
time_components = df[['duration', 'waiting', 'timeLoss']]
time_components.columns = ['Total', 'Waiting', 'Time Loss']
sns.histplot(time_components, element='step', ax=ax5)
plt.title('Time Component Distribution', fontsize=12)
plt.xlabel('Time (s)')

# 6. Speed Distribution
ax6 = plt.subplot(gs[2, 1])
sns.violinplot(x='route', y='avgSpeed', data=df[df['avgSpeed'] > 0], ax=ax6)
plt.title('Average Speed Distribution by Route', fontsize=12)
plt.xticks(rotation=45)

# 7. Congestion Analysis
ax7 = plt.subplot(gs[2, 2])
congestion = df.groupby('route').agg({
    'timeLoss': 'mean',
    'waiting': 'mean',
    'duration': 'count'
}).reset_index()
sns.scatterplot(x='timeLoss', y='waiting', size='duration', 
               hue='route', data=congestion, ax=ax7, sizes=(50, 300))
plt.title('Route Congestion Patterns', fontsize=12)
plt.xlabel('Average Time Loss')
plt.ylabel('Average Waiting Time')

# 8. Temporal Density
ax8 = plt.subplot(gs[3, :])
df['depart_min'] = df['depart'] // 60
time_route_density = df.pivot_table(index='depart_min', 
                                  columns='route', 
                                  values='id', 
                                  aggfunc='count')
sns.heatmap(time_route_density.fillna(0), cmap="YlGnBu", ax=ax8)
plt.title('Traffic Density by Time and Route', fontsize=14)
plt.xlabel('Route')
plt.ylabel('Minutes Since Simulation Start')

plt.tight_layout()
plt.savefig('detailed_traffic_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as detailed_traffic_analysis.png")
