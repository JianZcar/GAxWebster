from GAxWebsters import compute_signal_config_with_poisson, run_evolution, IntersectionParams, generate_population
from xml_generators import generate_tl_logic
from export_data import generate_traffic_report
import data_capture
import subprocess
import pprint
import os

saturation_flow = data_capture.get_saturation_flow()
import run_sim_without_trafficlights
average_flows = data_capture.get_average_flow()

subprocess.run(
        [
            "netconvert",
            "-n", "road-configuration/nodes.xml",
            "-e", "road-configuration/edges.xml",
            "-x", "road-configuration/connections.xml",
            "-o", "road-configuration/net.xml",
            "--verbose"
        ],
        check=True,       # raises if SUMO exits non-zero
        capture_output=True,
        text=True
    )

subprocess.run(
        [
            "sumo",
            "-n", "road-configuration/net.xml",
            "-r", "road-configuration/routes.xml",
            "--tripinfo-output", "tripinfo.xml",
            "--queue-output", "q_.xml",
            "--verbose"
        ],
        check=True,       # raises if SUMO exits non-zero
        capture_output=True,
        text=True
    )

print(data_capture.average_queue_length_per_edge("q_.xml"))
generate_traffic_report("tripinfo.xml", "Initial_traffic_bySUMO.png")
intersection_params = IntersectionParams(
    saturation_flows=[saturation_flow, saturation_flow, saturation_flow, saturation_flow], 
    lambda_rates=[round(average_flows['W_in']/60, 2), round(average_flows['E_in']/60, 2), round(average_flows['N_in']/60, 2)],
    reaction_time=1.0,                    # s
    road_widths=[3.2, 3.2, 3.2, 3.2],          # m
    vehicle_speed=13.89,                  # m/s
    deceleration_rate=4.5,                # m/s^2
    vehicle_length=5                      # m
)


population = generate_population(
    size=20, intersection_params=intersection_params)

generate_tl_logic('road-configuration/connections.xml', "tl_logic.xml", list(population.values())[0])

subprocess.run(
        [
            "sumo",
            "-n", "road-configuration/net.xml",
            "-r", "road-configuration/routes.xml",
            "--tripinfo-output", "tripinfo.xml",
            "--additional-files", "tl_logic.xml",
            "--verbose"
        ],
        check=True,       # raises if SUMO exits non-zero
        capture_output=True,
        text=True
    )

generate_traffic_report("tripinfo.xml", "Initial_traffic_byWebsters.png")


pop = run_evolution(population)

tl_xml = generate_tl_logic('road-configuration/connections.xml', "tl_logic.xml", pop[0][0])

subprocess.run(
        [
            "sumo",
            "-n", "road-configuration/net.xml",
            "-r", "road-configuration/routes.xml",
            "--tripinfo-output", "tripinfo.xml",
            "--additional-files", "tl_logic.xml",
            "--verbose"
        ],
        check=True,       # raises if SUMO exits non-zero
        capture_output=True,
        text=True
    )

generate_traffic_report("tripinfo.xml", "finalGA.png")
print("Initial Websters")
population = list(population.values())
#population.sort(key=lambda config: fitness(config))
print(population[0])


print("Final GA")   
print(pop[0][0])



