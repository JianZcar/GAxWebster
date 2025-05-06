import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Load XML
tree = ET.parse('veh_delays.xml')
root = tree.getroot()

# Collect travel times per direction
travel_times = defaultdict(list)

for vehicle in root.findall('vehicle'):
    vehicle_id = vehicle.attrib['id']
    depart = float(vehicle.attrib['depart'])
    arrival = float(vehicle.attrib['arrival'])
    direction = re.match(r"flow_([a-z_]+)\.\d+", vehicle_id).group(1)

    travel_time = arrival - depart
    travel_times[direction].append(travel_time)

# Compute average travel time per direction
average_times = {direction: sum(times)/len(times) for direction,
                 times in travel_times.items()}

# Plot
plt.figure(figsize=(10, 6))
plt.bar(average_times.keys(), average_times.values(), color='skyblue')
plt.xlabel('Route Direction')
plt.ylabel('Average Travel Time (s)')
plt.title('Average Vehicle Travel Time per Route Direction')
plt.tight_layout()
plt.savefig('avg_travel_times.png')
