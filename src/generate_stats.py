import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and parse the XML file from disk
tree = ET.parse('tripinfo.xml')
root = tree.getroot()

# 2. Extract key fields into a DataFrame
records = []
for trip in root.findall('tripinfo'):
    rec = {
        'id': trip.get('id'),
        'duration': float(trip.get('duration')),
        'routeLength': float(trip.get('routeLength')),
        'waitingTime': float(trip.get('waitingTime')),
        'timeLoss': float(trip.get('timeLoss')),
        # pick two representative emissions metrics
        'CO2_abs': float(trip.find('emissions').get('CO2_abs')),
        'fuel_abs': float(trip.find('emissions').get('fuel_abs')),
    }
    records.append(rec)

df = pd.DataFrame(records)
df.to_csv('tripinfo.csv', index=False)
print("Saved aggregated data to tripinfo.csv")

# 3. Display the DataFrame (optional interactive)
print(df)

# 4. Bar chart of trip durations
plt.figure()
plt.bar(df['id'], df['duration'])
plt.xlabel('Trip ID')
plt.ylabel('Duration (s)')
plt.title('Trip Duration by ID')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('duration_bar.png')
plt.close()

# 5. Scatter: Route Length vs Duration
plt.figure()
plt.scatter(df['routeLength'], df['duration'], marker='o')
for i, txt in enumerate(df['id']):
    plt.annotate(txt, (df['routeLength'][i], df['duration'][i]))
plt.xlabel('Route Length (m)')
plt.ylabel('Duration (s)')
plt.title('Duration vs Route Length')
plt.tight_layout()
plt.savefig('duration_vs_length.png')
plt.close()

# 6. Histogram of time loss
plt.figure()
plt.hist(df['timeLoss'], bins=5)
plt.xlabel('Time Loss (s)')
plt.ylabel('Number of Trips')
plt.title('Distribution of Time Loss')
plt.tight_layout()
plt.savefig('timeLoss_hist.png')
plt.close()

# 7. CO2 Emissions per trip
plt.figure()
plt.bar(df['id'], df['CO2_abs'])
plt.xlabel('Trip ID')
plt.ylabel('CO₂ Emissions (g)')
plt.title('CO₂ Emissions by Trip')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('CO2_emissions_bar.png')
plt.close()

# 8. Fuel consumption per trip
plt.figure()
plt.bar(df['id'], df['fuel_abs'])
plt.xlabel('Trip ID')
plt.ylabel('Fuel Consumed (ml)')
plt.title('Fuel Consumption by Trip')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fuel_consumption_bar.png')
plt.close()

print("Plots saved: duration_bar.png, duration_vs_length.png, timeLoss_hist.png, CO2_emissions_bar.png, fuel_consumption_bar.png")
