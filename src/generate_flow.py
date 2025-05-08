import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# Load detector counts
tree = ET.parse('det_counts.xml')
root = tree.getroot()

# Parse intervals into DataFrame
records = []
for interval in root.findall('interval'):
    records.append({
        'detector_id': interval.get('id'),
        'begin': float(interval.get('begin')),
        'end': float(interval.get('end')),
        'nVehEntered': int(interval.get('nVehEntered'))
    })
df = pd.DataFrame(records)

# Total simulation duration
sim_end = df['end'].max()

# Signal cycle and green windows
cycle = 85.0
green_windows = {}
for k in range(int(sim_end // cycle) + 1):
    green_windows.update({
        f'det_left_in_0': green_windows.get('det_left_in_0', []) + [(0 + k*cycle, 20 + k*cycle)],
        f'det_right_in_0': green_windows.get('det_right_in_0', []) + [(0 + k*cycle, 20 + k*cycle)],
        f'det_left_out_0': green_windows.get('det_left_out_0', []) + [(0 + k*cycle, 20 + k*cycle)],
        f'det_right_out_0': green_windows.get('det_right_out_0', []) + [(0 + k*cycle, 20 + k*cycle)],
        f'det_right_in_1': green_windows.get('det_right_in_1', []) + [(0 + k*cycle, 20 + k*cycle)],
        f'det_left_in_1': green_windows.get('det_left_in_1', []) + [(25 + k*cycle, 40 + k*cycle)],
        f'det_left_out_1': green_windows.get('det_left_out_1', []) + [(25 + k*cycle, 40 + k*cycle)],
        f'det_right_out_1': green_windows.get('det_right_out_1', []) + [(45 + k*cycle, 60 + k*cycle)],
        f'det_down_in_0': green_windows.get('det_down_in_0', []) + [(65 + k*cycle, 80 + k*cycle)],
        f'det_down_out_0': green_windows.get('det_down_out_0', []) + [(65 + k*cycle, 80 + k*cycle)],
        f'det_down_in_1': green_windows.get('det_down_in_1', []) + [(65 + k*cycle, 80 + k*cycle)],
        f'det_down_out_1': green_windows.get('det_down_out_1', []) + [(65 + k*cycle, 80 + k*cycle)],
    })

# Compute flows
results = []
for det_id, windows in green_windows.items():
    det_df = df[df['detector_id'] == det_id]
    total_vehicles = det_df['nVehEntered'].sum()
    avg_flow = total_vehicles / sim_end * 3600  # veh/h

    # Compute saturation flow (peak during any green phase)
    phase_rates = []
    for start, end in windows:
        mask = (det_df['begin'] >= start) & (det_df['end'] <= end)
        count = det_df.loc[mask, 'nVehEntered'].sum()
        duration = end - start
        if duration > 0:
            phase_rates.append(count / duration * 3600)
    sat_flow = max(phase_rates) if phase_rates else 0

    results.append({
        'detector_id': det_id,
        'avg_flow_veh_h': round(avg_flow, 2),
        'sat_flow_peak_veh_h': round(sat_flow, 2)
    })

results_df = pd.DataFrame(results)

# Show in terminal
print(results_df.to_string(index=False))

# Save table as PNG
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
table.scale(1, 1.5)
plt.savefig("flow_results.png", bbox_inches='tight', dpi=300)
plt.close()
