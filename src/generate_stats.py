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
    """XML parsing with route validation"""
    data = []
    valid_routes = {'left_right', 'right_left', 'right_down', 'left_down'}  # Add expected routes
    
    for event, elem in ET.iterparse(xml_file, events=('end',)):
        if elem.tag == 'tripinfo':
            route = elem.get('id').split('.')[0]
            if '_' in route:  # Enhanced route validation
                route = route.split('_', 1)[1]  # Get flow direction
                
            entry = {
                'id': elem.get('id'),
                'depart': safe_get(elem, 'depart'),
                'duration': safe_get(elem, 'duration'),
                'waiting': safe_get(elem, 'waitingTime'),
                'timeLoss': safe_get(elem, 'timeLoss'),
                'route': route if route in valid_routes else 'other',
                'routeLength': safe_get(elem, 'routeLength'),
            }
            data.append(entry)
            elem.clear()
    return pd.DataFrame(data)

# Data processing
try:
    df = parse_xml('tripinfo.xml')
    
    print("Route Distribution:")
    print(df['route'].value_counts())
    
    # Filter meaningful routes with minimum trips
    route_counts = df['route'].value_counts()
    valid_routes = route_counts[route_counts > 5].index
    filtered_df = df[df['route'].isin(valid_routes)]

except Exception as e:
    print(f"Error: {str(e)}")
    exit()

# Visualization setup
plt.figure(figsize=(24, 20))
gs = GridSpec(3, 2, figure=plt.gcf())
sns.set_theme(style="whitegrid", palette="pastel")

# 1. Route Performance Metrics (Improved)
ax1 = plt.subplot(gs[0, 0])
route_stats = filtered_df.groupby('route').agg({
    'duration': ['mean', 'std'],
    'timeLoss': 'median',
    'routeLength': 'first'
}).reset_index()

sns.barplot(x='route', y=('duration', 'mean'), 
           data=route_stats, ax=ax1)
plt.title('Average Trip Duration by Route', fontsize=14)
plt.xlabel('Route')
plt.ylabel('Duration (s)')

# 2. Route Congestion Patterns (Fixed)
ax2 = plt.subplot(gs[0, 1])
congestion = filtered_df.groupby('route').agg({
    'timeLoss': 'mean',
    'waiting': 'mean',
    'id': 'count'
}).rename(columns={'id': 'count'}).reset_index()

sns.scatterplot(x='timeLoss', y='waiting', size='count',
               hue='route', data=congestion, ax=ax2,
               sizes=(100, 500), alpha=0.8)
plt.title('Route Congestion Patterns', fontsize=14)
plt.xlabel('Average Time Loss (s)')
plt.ylabel('Average Waiting Time (s)')

# 3. Route Efficiency Analysis
ax3 = plt.subplot(gs[1, 0])
route_stats['efficiency'] = route_stats['routeLength'] / route_stats['duration']['mean']
sns.barplot(x='route', y='efficiency', data=route_stats, ax=ax3)
plt.title('Route Efficiency (Distance/Duration)', fontsize=14)
plt.ylabel('Efficiency (m/s)')

# 4. Time Loss Distribution
ax4 = plt.subplot(gs[1, 1])
sns.boxplot(x='route', y='timeLoss', data=filtered_df, ax=ax4)
plt.title('Time Loss Distribution by Route', fontsize=14)
plt.xticks(rotation=45)

# 5. Route Length Comparison
ax5 = plt.subplot(gs[2, 0])
sns.barplot(x='route', y='routeLength', data=route_stats, ax=ax5)
plt.title('Route Length Comparison', fontsize=14)
plt.ylabel('Distance (m)')

# 6. Duration vs Route Length
ax6 = plt.subplot(gs[2, 1])
sns.scatterplot(x='routeLength', y='duration', hue='route',
               data=filtered_df, ax=ax6, alpha=0.6)
plt.title('Trip Duration vs Route Length', fontsize=14)
plt.xlabel('Route Length (m)')
plt.ylabel('Duration (s)')

plt.tight_layout()
plt.savefig('enhanced_route_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as enhanced_route_analysis.png")
