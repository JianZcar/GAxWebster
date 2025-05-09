from GAxWebsters import compute_signal_config_with_poisson, run_evolution, IntersectionParams, generate_population
from insert_sumo import generate_tl_logic
from generate_stats import generate_traffic_report
import get_sumo
import subprocess
import os

saturation_flow = get_sumo.get_saturation_flow()
average_flows = get_sumo.get_average_flow()

subprocess.run(
        [
            "sumo",
            "-n", "road-configuration/net.xml",
            "-r", "road-configuration/routes.xml",
            "--tripinfo-output", "tripinfo.xml",
            "--verbose"
        ],
        check=True,       # raises if SUMO exits non-zero
        capture_output=True,
        text=True
    )

generate_traffic_report("tripinfo.xml", "Initial_traffic_bySUMO.png")
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

tl_xml = generate_tl_logic(list(population.values())[0])
with open("traffic_lights.add.xml", "w") as f:
    f.write(tl_xml)

subprocess.run(
        [
            "sumo",
            "-n", "road-configuration/net.xml",
            "-r", "road-configuration/routes.xml",
            "--tripinfo-output", "tripinfo.xml",
            "--additional-files", "traffic_lights.add.xml",
            "--verbose"
        ],
        check=True,       # raises if SUMO exits non-zero
        capture_output=True,
        text=True
    )

generate_traffic_report("tripinfo.xml", "Initial_traffic_byWebsters.png")


pop = run_evolution(population)

tl_xml = generate_tl_logic(pop[0][0])
with open("traffic_lights.add.xml", "w") as f:
    f.write(tl_xml)

subprocess.run(
        [
            "sumo",
            "-n", "road-configuration/net.xml",
            "-r", "road-configuration/routes.xml",
            "--tripinfo-output", "tripinfo.xml",
            "--additional-files", "traffic_lights.add.xml",
            "--verbose"
        ],
        check=True,       # raises if SUMO exits non-zero
        capture_output=True,
        text=True
    )

generate_traffic_report("tripinfo.xml", "finalGA.png")
print("Initial Websters")
print(list(population.values())[0])


print("Final GA")   
print(pop[0][0])



