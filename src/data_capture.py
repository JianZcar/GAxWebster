import math
import xml.etree.ElementTree as ET
import traci
import sumolib

def get_saturation_flow():
    """Calculate saturation flow using calibrated Krauss model parameters"""
    try:
        tree = ET.parse("road-configuration/routes.xml")
        root = tree.getroot()
        vtype = root.find('vType')

        params = {
            'length': 5.0,
            'minGap': 2.5,
            'accel': 2.6,
            'decel': 4.5,
            'tau': 1.0,
            'sigma': 0.38  # Calibrated value for observed 768 veh/h
        }
        
        # Load parameters from XML or use defaults
        for key in params:
            params[key] = float(vtype.get(key, str(params[key])))

        # Krauss model adjustment factor (empirically derived)
        krauss_factor = 0.835  # Accounts for acceleration/deceleration dynamics
        
        # Effective clearance distance
        clearance = params['length'] + params['minGap']
        
        # Adjusted acceleration considering driver behavior
        effective_accel = params['accel'] * (1 - params['sigma'] * 0.6)
        
        # Time to clear gap with leader
        acceleration_time = math.sqrt(2 * clearance / effective_accel)
        
        # Final headway calculation
        headway = params['tau'] + (acceleration_time * krauss_factor)
        
        saturation_flow = 3600 / headway

        print(f"Calibrated saturation flow: {saturation_flow:.2f} veh/hour/lane")
        return saturation_flow
        
    except Exception as e:
        print(f"Error calculating saturation flow: {str(e)}")
        return 0

def get_average_flow():
    """
    Calculate average traffic flow (veh/hour) for incoming edges to junction center,
    with fallback for non-traffic-light scenarios.
    """
    # Start SUMO simulation
    sumo_cmd = ["sumo", 
        "-n", "road-configuration/net.xml", 
        "-r", "road-configuration/routes.xml"]
    traci.start(sumo_cmd)
    print("All edges:", traci.edge.getIDList())
    
    # Initialize data structures
    incoming_edges = ['W_in', 'S_in', 'N_in','E_in']
    vehicle_counts = {edge: 0 for edge in incoming_edges}
    has_traffic_light = False

    try:
        # Check if traffic light exists
        tl_ids = traci.trafficlight.getIDList()
        if 'J0' in traci.trafficlight.getIDList():
            tl_program = traci.trafficlight.getCompleteRedYellowGreenDefinition('J0')
            print("Traffic light phases:", tl_program)
            
            # Example mapping (adjust based on actual phases):
            phase_incoming = {
                0: 'E_in',  # Phase 0: East green
                1: 'N_in',  # Phase 1: North green
                2: 'W_in',  # Phase 2: West green
                3: 'S_in'   # Phase 3: South green
            }
    except traci.TraCIException:
        has_traffic_light = False

    # Track vehicles that have been counted
    counted_vehicles = set()
    previous_vehicles = {edge: set() for edge in incoming_edges}

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        if has_traffic_light:
            try:
                current_phase = traci.trafficlight.getPhase(junction_id)
                active_incoming = phase_incoming.get(current_phase % 3, None)
            except traci.TraCIException:
                # Fallback if traffic light disappears during simulation
                has_traffic_light = False
                active_incoming = None
        else:
            # No traffic light mode: count all incoming edges
            active_incoming = None

        if has_traffic_light and active_incoming:
            # Traffic light-aware counting
            current_vehicles = set(traci.edge.getLastStepVehicleIDs(active_incoming))
            new_vehicles = current_vehicles - previous_vehicles[active_incoming]
            
            for veh_id in new_vehicles:
                if veh_id not in counted_vehicles:
                    vehicle_counts[active_incoming] += 1
                    counted_vehicles.add(veh_id)
            
            previous_vehicles[active_incoming] = current_vehicles.copy()
        else:
            # Count all incoming edges continuously
            for edge in incoming_edges:
                current_vehicles = set(traci.edge.getLastStepVehicleIDs(edge))
                new_vehicles = current_vehicles - previous_vehicles[edge]
                
                for veh_id in new_vehicles:
                    if veh_id not in counted_vehicles:
                        vehicle_counts[edge] += 1
                        counted_vehicles.add(veh_id)
                
                previous_vehicles[edge] = current_vehicles.copy()

    total_time = traci.simulation.getTime()
    traci.close()

    # Calculate flows
    average_flows = {}
    for edge in incoming_edges:
        if total_time == 0:
            average_flows[edge] = 0
        else:
            average_flows[edge] = (vehicle_counts[edge] / total_time) * 3600

    print(f"Average flows ({'with' if has_traffic_light else 'without'} traffic lights):")
    print(average_flows)
    return average_flows
    
if __name__ == "__main__":
    get_saturation_flow()
    get_average_flow()
