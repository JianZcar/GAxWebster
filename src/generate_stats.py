import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import math

# 1. Parse the XML from file
tree = ET.parse('tripinfo.xml')
root = tree.getroot()

# 2. Extract data, with safe handling if <emissions> is missing
records = []
for trip in root.findall('tripinfo'):
    # Base attributes
    rec = {
        'id': trip.get('id'),
        'duration': float(trip.get('duration') or math.nan),
        'routeLength': float(trip.get('routeLength') or math.nan),
        'waitingTime': float(trip.get('waitingTime') or math.nan),
        'timeLoss': float(trip.get('timeLoss') or math.nan),
    }
    # Find any child whose tag ends with 'emissions'
    emis = None
    for child in trip:
        if child.tag.endswith('emissions'):
            emis = child
            break
    # Populate emissions metrics if present
    if emis is not None:
        rec['CO2_abs']  = float(emis.get('CO2_abs' ) or math.nan)
        rec['fuel_abs'] = float(emis.get('fuel_abs') or math.nan)
    else:
        rec['CO2_abs']  = math.nan
        rec['fuel_abs'] = math.nan

    records.append(rec)

# 3. Build DataFrame and save
df = pd.DataFrame(records)
df.to_csv('tripinfo.csv', index=False)
print("Saved tripinfo.csv:")
print(df)

# 4. Visualizations

# 4a. Bar chart: Trip Duration
plt.figure()
plt.bar(df['id'], df['duration'])
plt.xlabel('Trip ID'); plt.ylabel('Duration (s)')
plt.title('Trip Duration by ID')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('duration_bar.png')
plt.close()

# 4b. Scatter: Route Length vs Duration
plt.figure()
plt.scatter(df['routeLength'], df['duration'])
for i, txt in enumerate(df['id']):
    plt.annotate(txt,
                 (df['routeLength'].iat[i], df['duration'].iat[i]),
                 textcoords="offset points", xytext=(0,5), ha='center')
plt.xlabel('Route Length (m)'); plt.ylabel('Duration (s)')
plt.title('Duration vs Route Length')
plt.tight_layout()
plt.savefig('duration_vs_length.png')
plt.close()

# 4c. Histogram: Time Loss
plt.figure()
plt.hist(df['timeLoss'].dropna(), bins=5)
plt.xlabel('Time Loss (s)'); plt.ylabel('Number of Trips')
plt.title('Distribution of Time Loss')
plt.tight_layout()
plt.savefig('timeLoss_hist.png')
plt.close()

# 4d. Bar: CO₂ Emissions
plt.figure()
plt.bar(df['id'], df['CO2_abs'])
plt.xlabel('Trip ID'); plt.ylabel('CO₂ Emissions (g)')
plt.title('CO₂ Emissions by Trip')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('CO2_emissions_bar.png')
plt.close()

# 4e. Bar: Fuel Consumption
plt.figure()
plt.bar(df['id'], df['fuel_abs'])
plt.xlabel('Trip ID'); plt.ylabel('Fuel Consumed (ml)')
plt.title('Fuel Consumption by Trip')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fuel_consumption_bar.png')
plt.close()

print("Charts saved:")
print(" duration_bar.png")
print(" duration_vs_length.png")
print(" timeLoss_hist.png")
print(" CO2_emissions_bar.png")
print(" fuel_consumption_bar.png")
