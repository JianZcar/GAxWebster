from GAxWebsters import compute_signal_config_with_poisson, run_evolution, IntersectionParams, generate_population
from insert_sumo import generate_tl_logic
import get_sumo

saturation_flow = get_sumo.get_saturation_flow()
average_flows = get_sumo.get_average_flow()

intersection_params = IntersectionParams(
    saturation_flows=[saturation_flow, saturation_flow, saturation_flow], 
    lambda_rates=[round(average_flows['E_in']/60, 2), round(average_flows['W_in']/60, 2), round(average_flows['S_in']/60, 2)],
    reaction_time=1.0,                    # s
    road_widths=[3.2, 3.2, 3.2],          # m
    vehicle_speed=13.89,                  # m/s
    deceleration_rate=4.5,                # m/s^2
    vehicle_length=5                      # m
)

population = generate_population(
    size=20, intersection_params=intersection_params)
    
print(population)

pop = run_evolution(population)



