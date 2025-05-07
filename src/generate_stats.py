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
                'routeLength': safe_get(elem, 'routeLength'),
            }
            data.append(entry)
            elem.clear()
    return pd.DataFrame(data)

# Data processing
try:
    df = parse_xml('tripinfo.xml')
    if df.empty:
        raise ValueError("No valid trip data found in XML file")
        
    print("Data Summary:")
    print(df[['duration', 'waiting', 'timeLoss']].describe())
    
    # Calculate time bins based on simulation duration
    max_time = df['depart'].max()
    time_interval = max(10, int(max_time/20))  # Dynamic bin sizing
    
except Exception as e:
    print(f"Error: {str(e)}")
    exit()

# Visualization setup
plt.figure(figsize=(24, 28))
gs = GridSpec(4, 3, figure=plt.gcf())
sns.set_theme(style="whitegrid", palette="pastel")

# 1. Temporal Traffic Patterns - Enhanced
ax1 = plt.subplot(gs[0, :])
df['time_bin'] = (df['depart'] // time_interval) * time_interval
time_bins = df.groupby('time_bin').size()
sns.barplot(x=time_bins.index, y=time_bins.values, ax=ax1)
plt.title(f'Departure Patterns ({time_interval}s Intervals)', fontsize=14)
plt.xlabel(f'Time Bins ({time_interval} seconds)')
plt.ylabel('Number of Vehicles')

# 2. Route Performance Analysis
ax2 = plt.subplot(gs[1, 0])
sns.boxplot(x='route', y='duration', data=df, ax=ax2)
plt.title('Trip Duration by Route', fontsize=12)
plt.xticks(rotation=45)

# 3. Congestion Analysis
ax3 = plt.subplot(gs[1, 1])
congestion = df.groupby('route').agg({
    'timeLoss': 'mean',
    'waiting': 'mean',
    'duration': 'count'
}).reset_index()
sns.scatterplot(x='timeLoss', y='waiting', size='duration', 
               hue='route', data=congestion, ax=ax3, sizes=(50, 300))
plt.title('Route Congestion Patterns', fontsize=12)
plt.xlabel('Average Time Loss (s)')
plt.ylabel('Average Waiting Time (s)')

# 4. Lane Utilization - Enhanced
ax4 = plt.subplot(gs[1, 2])
lane_counts = df['departLane'].value_counts().nlargest(10)
sns.barplot(x=lane_counts.values, y=lane_counts.index, ax=ax4)
plt.title('Top 10 Busiest Departure Lanes', fontsize=12)
plt.xlabel('Vehicle Count')

# 5. Speed Analysis
ax5 = plt.subplot(gs[2, 0])
sns.histplot(df['speedFactor'], bins=20, kde=True, ax=ax5)
plt.title('Speed Factor Distribution', fontsize=12)
plt.xlabel('Speed Factor')

# 6. Temporal Density - Enhanced
ax6 = plt.subplot(gs[2, 1:])
df['time_window'] = (df['depart'] // (time_interval*6))  # 6 bins across simulation
time_route_density = df.groupby(['time_window', 'route']).size().unstack(fill_value=0)
sns.heatmap(time_route_density.T, cmap="YlGnBu", ax=ax6, 
           cbar_kws={'label': 'Vehicle Count'})
plt.title('Traffic Density by Time Windows and Route', fontsize=14)
plt.xlabel(f'Time Windows ({time_interval*6}s each)')
plt.ylabel('Route')

# 7. Time Component Analysis
ax7 = plt.subplot(gs[3, 0])
sns.scatterplot(x='duration', y='timeLoss', hue='route', 
               data=df, ax=ax7, alpha=0.6)
plt.title('Trip Duration vs Time Loss', fontsize=12)
plt.xlabel('Total Duration (s)')
plt.ylabel('Time Loss (s)')

# 8. Route Efficiency
ax8 = plt.subplot(gs[3, 1])
route_efficiency = df.groupby('route').agg({
    'routeLength': 'mean',
    'duration': 'median'
}).reset_index()
route_efficiency['efficiency'] = route_efficiency['routeLength'] / route_efficiency['duration']
sns.barplot(x='route', y='efficiency', data=route_efficiency, ax=ax8)
plt.title('Route Efficiency (Distance/Time)', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('enhanced_traffic_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as enhanced_traffic_analysis.png")
