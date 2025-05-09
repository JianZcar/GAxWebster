from GAxWebsters import compute_signal_config_with_poisson, run_evolution, IntersectionParams, generate_population
from insert_sumo import generate_tl_logic
from generate_stats import generate_traffic_report
import get_sumo
import subprocess
import os

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

generate_traffic_report("tripinfo.xml", "Initial_notrafficlight_bySUMO.png")


