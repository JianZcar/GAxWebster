import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
xml_file = 'tripinfo.xml'  # Input XML file
output_image = 'traffic_report.png'  # Output figure

# --- Parse XML ---
root = ET.parse(xml_file).getroot()
records = []
for trip in root.findall('tripinfo'):
    data = {
        'id': trip.get('id'),
        'depart': float(trip.get('depart')),
        'arrival': float(trip.get('arrival')),
        'duration': float(trip.get('duration')),
        'routeLength': float(trip.get('routeLength')),
        'waitingTime': float(trip.get('waitingTime')),
        'stopTime': float(trip.get('stopTime')),
        'timeLoss': float(trip.get('timeLoss')),
        'departDelay': float(trip.get('departDelay')),
        'waitingCount': int(trip.get('waitingCount')),
        'vType': trip.get('vType'),
    }
    # derive route direction from id or departLane
    data['route'] = trip.get('id').split('.')[0]
    records.append(data)

df = pd.DataFrame(records)

# --- Summary Statistics ---
summary = pd.Series({
    'Total Trips': len(df),
    'Avg Duration (s)': df['duration'].mean(),
    'Avg Time Loss (s)': df['timeLoss'].mean(),
    'Avg Waiting Time (s)': df['waitingTime'].mean(),
    'Total Stops': df['waitingCount'].sum(),
    'Trips Delayed at Start': (df['departDelay'] > 0).sum(),
    'Avg Depart Delay (s)': df['departDelay'].mean(),
})

# Print summary
print("--- Traffic Report Summary ---")
for k, v in summary.items():
    print(f"{k}: {v:.2f}")

# --- Visualization ---
plt.figure(figsize=(14, 10))

# 1. Histogram of time loss
plt.subplot(2, 2, 1)
plt.hist(df['timeLoss'], bins=10)
plt.title('Time Loss Distribution (s)')
plt.xlabel('Time Loss (s)')
plt.ylabel('Number of Trips')

# 2. Scatter of duration vs. waitingTime
plt.subplot(2, 2, 2)
plt.scatter(df['duration'], df['waitingTime'], alpha=0.7)
plt.title('Trip Duration vs. Waiting Time')
plt.xlabel('Duration (s)')
plt.ylabel('Waiting Time (s)')

# 3. Bar chart of average timeLoss per route
plt.subplot(2, 2, 3)
avg_loss = df.groupby('route')['timeLoss'].mean().sort_values()
avg_loss.plot(kind='bar')
plt.title('Avg Time Loss per Route')
plt.ylabel('Avg Time Loss (s)')
plt.xticks(rotation=45)

# 4. Pie chart of delayed vs on-time departures
plt.subplot(2, 2, 4)
delayed = (df['departDelay'] > 0).sum()
on_time = len(df) - delayed
plt.pie([delayed, on_time], labels=['Delayed', 'On-Time'], autopct='%1.1f%%')
plt.title('Departures: Delayed vs On-Time')

plt.tight_layout()
plt.savefig(output_image, dpi=300)
print(f"Report image saved to {output_image}")
