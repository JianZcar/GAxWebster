#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# 1. Parse XML
tree = ET.parse('tripinfo.xml')
root = tree.getroot()

# 2. Extract data
records = []
for ti in root.findall('tripinfo'):
    attrib = ti.attrib

    # Try to find emissions; if missing, set all to 0
    em_elem = ti.find('emissions')
    if em_elem is not None:
        em = em_elem.attrib
    else:
        em = {k: '0' for k in ('fuel_abs', 'CO2_abs', 'NOx_abs', 'CO_abs', 'HC_abs', 'PMx_abs')}

    # direction from id
    flow = attrib['id'].split('.')[0].replace('flow_', '')
    records.append({
        'id':               attrib.get('id', ''),
        'flow':             flow,
        'duration':         float(attrib.get('duration', 0)),
        'routeLength':      float(attrib.get('routeLength', 0)),
        'timeLoss':         float(attrib.get('timeLoss', 0)),
        'waitingTime':      float(attrib.get('waitingTime', 0)),
        'fuel_abs':         float(em.get('fuel_abs', 0)),
        'CO2_abs':          float(em.get('CO2_abs', 0)),
        'NOx_abs':          float(em.get('NOx_abs', 0)),
        'CO_abs':           float(em.get('CO_abs', 0)),
        'HC_abs':           float(em.get('HC_abs', 0)),
        'PMx_abs':          float(em.get('PMx_abs', 0)),
        'speedFactor':      float(attrib.get('speedFactor', 0)),
        'arrivalSpeed':     float(attrib.get('arrivalSpeed', 0)),
    })

df = pd.DataFrame(records)

# Helper to save a figure
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(name)
    plt.close(fig)

# 3. Plot 1: Duration vs. Route Length
fig = plt.figure()
plt.scatter(df['routeLength'], df['duration'])
plt.xlabel('Route Length (m)')
plt.ylabel('Duration (s)')
plt.title('Trip Duration vs. Route Length')
save_fig(fig, 'duration_vs_route_length.png')

# 4. Plot 2: Stacked Emissions per Vehicle
fig, ax = plt.subplots()
em_cols = ['CO2_abs', 'NOx_abs', 'CO_abs', 'HC_abs', 'PMx_abs']
df.set_index('id')[em_cols].plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Vehicle ID')
ax.set_ylabel('Emissions (mg)')
ax.set_title('Stacked Emissions per Trip')
ax.tick_params(axis='x', rotation=45, ha='right')
save_fig(fig, 'emissions_per_vehicle.png')

# 5. Plot 3: Fuel vs. CO2
fig = plt.figure()
plt.scatter(df['fuel_abs'], df['CO2_abs'])
plt.xlabel('Fuel Consumed (mg)')
plt.ylabel('CO2 Emitted (mg)')
plt.title('Fuel Consumption vs. CO2 Emissions')
save_fig(fig, 'fuel_vs_co2.png')

# 6. Plot 4: Time Loss by Flow Direction (Box Plot)
fig, ax = plt.subplots()
df.boxplot(column='timeLoss', by='flow', ax=ax)
ax.set_xlabel('Flow Direction')
ax.set_ylabel('Time Loss (s)')
ax.set_title('Time Loss by Flow Direction')
fig.suptitle('')  # remove automatic subtitle
save_fig(fig, 'time_loss_by_flow.png')

# 7. Plot 5: Speed Factor vs. Final Speed
fig = plt.figure()
plt.scatter(df['speedFactor'], df['arrivalSpeed'])
plt.xlabel('Speed Factor')
plt.ylabel('Arrival Speed (m/s)')
plt.title('Speed Factor vs. Final Speed')
save_fig(fig, 'speed_factor_vs_final_speed.png')

print("All plots generated and saved as PNG files.")
