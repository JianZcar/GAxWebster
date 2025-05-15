from collections import namedtuple
from functools import partial
from random import choices, randint, random, randrange, uniform, sample
from typing import Callable, List, Tuple
from math import ceil
from dataclasses import dataclass
from xml_generators import generate_tl_logic
import numpy as np
import copy
import pprint
import subprocess
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os

###############
### TYPINGS ###
###############


@dataclass
class IntersectionParams:
    saturation_flows: List[float]  # veh/hour per phase
    lambda_rates: List[float]      # mean arrivals (veh/min)
    reaction_time: float           # driver reaction (s)
    road_widths: List[float]       # approach width per phase (m)
    vehicle_speed: float           # vehicle speed (m/s)
    deceleration_rate: float       # comfortable deceleration (m/s^2)
    vehicle_length: float          # average vehicle length (m)


@dataclass
class PhaseConfig:
    green: float
    amber: float
    all_red: float


TrafficConfiguration = List[PhaseConfig]
Population = List[TrafficConfiguration]
FitnessFunc = Callable[[TrafficConfiguration], int]

PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc],
                         Tuple[TrafficConfiguration, TrafficConfiguration]]
CrossoverFunc = Callable[[TrafficConfiguration, TrafficConfiguration],
                         Tuple[TrafficConfiguration, TrafficConfiguration, TrafficConfiguration]]
MutationFunc = Callable[[TrafficConfiguration], TrafficConfiguration]


#########################
### TRAFFIC FUNCTIONS ###
#########################
def compute_amber_time(
    tr: int | float,
    v: int | float,
    a: int | float
) -> int:
    """Compute the amber (yellow) interval

    Args:
        tr (float): driver reaction time (s)
        v (float): vehicle approach speed (m/s)
        a (float): comfortable deceleration rate (m/s²)

    Returns:
        int: amber (yellow) interval
    """
    return ceil(tr + (v / (2 * a)))


def compute_all_red_time(
    W: int | float,
    L: int | float,
    v: int | float
) -> int:
    """Compute the all-red clearance interval

    Args:
        W (int | float): width of the lane approach (m)
        L (int | float): average vehicle length (m)
        v (int | float): approach speed (m/s)

    Returns:
        int: all‑red clearance interval
    """
    return ceil((W + L) / v)


def websters_method(
    L: int,
    Y: float
) -> int:
    """Compute for the Optimal Cycle Length

    Args:
        L (int): Total Lost Time (s)
        Y (float): Total Critical Ratio

    Returns:
        int: Optimal Cycle Length (s)
    """
    return ceil(1.5 * L + 5) / (1 - Y)


def compute_green_time(
    y: int | float,
    Y: int | float,
    C: int,
    L: int
) -> int:
    return ceil((y * (C - L)) / Y)


def simulate_poisson_arrival_rate(q: float) -> int:
    """
    Return a single Poisson‐distributed random draw representing
    the number of arrivals in one time interval, given mean q.

    Args:
        q: Expected number of arrivals in that interval.
    Returns:
        An integer count of arrivals.
    """
    return max(1, ceil(np.random.poisson(lam=q)))


def compute_signal_config_with_poisson(
    saturation_flows: List[float],  # veh/hour per phase
    lambda_rates: List[float],      # mean arrivals (veh/min)
    reaction_time: float,           # driver reaction (s)
    road_widths: List[float],       # approach width per phase (m)
    vehicle_speed: float,           # vehicle speed (m/s)
    deceleration_rate: float,       # comfortable deceleration (m/s^2)
    vehicle_length: float           # average vehicle length (m)
) -> List[PhaseConfig]:
    """
    Returns a list of PhaseConfig tuples (green, amber, red) for each phase.
    If Y >= 1, recomputes with a new Poisson realization until Y < 1.
    """
    while True:
        # 1) Simulate total arrivals per minute using Poisson
        normal_flows_per_minute = [
            simulate_poisson_arrival_rate(q=lam) for lam in lambda_rates]

        # 2) Compute flow ratios
        flow_ratios = [(q * 60) / s if s > 0 else 0 for q,
                       s in zip(normal_flows_per_minute, saturation_flows)]
        Y = sum(flow_ratios)

        if Y < 1:
            break  # Acceptable configuration found

    # 3) Lost time per phase (reaction + max clearance)
    amber_times = []
    all_red_times = []
    for W in road_widths:
        amber_time = compute_amber_time(
            tr=reaction_time, v=vehicle_speed, a=deceleration_rate)
        amber_times.append(amber_time)

        all_red_time = compute_all_red_time(
            W=W, L=vehicle_length, v=vehicle_speed)
        all_red_times.append(all_red_time)

    L = sum(amber_times + all_red_times)

    # 4) Webster cycle length
    C = websters_method(L=L, Y=Y)

    # 5) Allocate green times
    green_times = [compute_green_time(y=y, Y=Y, C=C, L=L) for y in flow_ratios]

    # 6) Build phase configurations
    tl_config = []
    for i in range(len(green_times)):
        tl_config.append(PhaseConfig(
            green=green_times[i],
            amber=amber_times[i],
            all_red=all_red_times[i]
        ))

    return tl_config


def generate_population(
    size: int,
    intersection_params: IntersectionParams
) -> List[List[PhaseConfig]]:
    population = dict()

    for idx in range(size):
        signal_config = compute_signal_config_with_poisson(
            saturation_flows=intersection_params.saturation_flows,
            lambda_rates=intersection_params.lambda_rates,
            reaction_time=intersection_params.reaction_time,
            road_widths=intersection_params.road_widths,
            vehicle_speed=intersection_params.vehicle_speed,
            deceleration_rate=intersection_params.deceleration_rate,
            vehicle_length=intersection_params.vehicle_length
        )
        population[idx + 1] = signal_config

    return population


intersection_params = IntersectionParams(
    saturation_flows=[1800, 1600, 2000],  # veh/hr
    lambda_rates=[7, 6, 7],               # veh/min
    reaction_time=1.0,                    # s
    road_widths=[7.0, 6.0, 8.0],          # m
    vehicle_speed=11.11,                  # m/s
    deceleration_rate=3.0,                # m/s^2
    vehicle_length=4.5                    # m
)

#########################
### GENETIC ALGORITHM ###
#########################
def selection(
    population: Population,
    fitness_func: FitnessFunc
) -> Population:
    """
    Selects two individuals from the population based on fitness-proportional selection (roulette wheel).

    Each individual is assigned a selection probability proportional to its fitness score.
    Higher-fitness individuals are more likely to be selected, allowing the algorithm to favor
    better-performing configurations while maintaining diversity.

    Args:
        population (Population): A list of traffic configurations (individuals).
        fitness_func (FitnessFunc): A function that takes a traffic configuration and returns its fitness score.

    Returns:
        Population: A list containing two selected individuals from the input population.
    """
    return choices(
        population=population,
        weights=[fitness_func(traffic_configuration)
                 for traffic_configuration in population],
        k=2
    )


def fitness(traffic_configuration: TrafficConfiguration) -> float:
    """Evaluate traffic config using SUMO simulation metrics"""
    try:
        min_green = 5
        if any(phase.green < min_green for phase in traffic_configuration):  # Remove 
            return float('inf')
        # 1. Generate traffic light XML
        tl_xml = generate_tl_logic(traffic_configuration)
        with open("traffic_lights.add.xml", "w") as f:
            f.write(tl_xml)

        # 2. Run SUMO simulation
        sumo_cmd = [
            "sumo",
            "-n", "road-configuration/net.xml",        # Your network file
            "-r", "road-configuration/routes.xml",         # Your routes file
            "--additional-files", "traffic_lights.add.xml",
            "--tripinfo-output", "tripinfo.xml",
        ]
        result = subprocess.run(sumo_cmd, 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               check=True)

        # 3. Parse simulation results
        tree = ET.parse('tripinfo.xml')
        root = tree.getroot()
        
        metrics = {
            'total_time_loss': 0.0,
            'total_waiting': 0.0,
            'total_stops': 0,
            'delayed_departures': 0,
            'count': 0
        }

        for trip in root.findall('tripinfo'):
            metrics['count'] += 1
            metrics['total_time_loss'] += float(trip.get('timeLoss', 0))
            metrics['total_waiting'] += float(trip.get('waitingTime', 0))
            metrics['total_stops'] += int(trip.get('waitingCount', 0))
            if float(trip.get('departDelay', 0)) > 0:
                metrics['delayed_departures'] += 1

        if metrics['count'] == 0:
            return float('inf')

        # 4. Calculate fitness components
        avg_time_loss = metrics['total_time_loss'] / metrics['count']
        avg_waiting = metrics['total_waiting'] / metrics['count']
        stops_per_trip = metrics['total_stops'] / metrics['count']
        delayed_ratio = metrics['delayed_departures'] / metrics['count']

        # 5. Weighted fitness score (lower = better)
        weights = {
            'time_loss': 0.5,
            'waiting': 0.4,
            'stops': 0.4,
            'delays': 0.05
        }
        score = (
            weights['time_loss'] * avg_time_loss +
            weights['waiting'] * avg_waiting +
            weights['stops'] * stops_per_trip +
            weights['delays'] * delayed_ratio
        )

        # 6. Add penalty for invalid configurations
        if any(phase.green < 5 for phase in traffic_configuration):
            score *= 1.5
        
        print(f"Simulation stats:")
        print(f"- Avg time loss: {avg_time_loss:.1f}s")
        print(f"- Avg waiting: {avg_waiting:.1f}s")
        print(f"- Stops per trip: {stops_per_trip:.1f}")

        return round(score, 2)

    except Exception as e:
        print(f"Fitness calculation failed: {str(e)}")
        return float('inf')

def linear_crossover(
    x: int, y: int
) -> Tuple[int, int, int]:
    """
    Performs linear crossover with minimum green time enforcement.
    """
    alpha = uniform(-0.5, 1.5)
    min_green = 5  # Minimum allowed green time
    
    c1 = max(min_green, int(x + alpha * (y - x)))
    c2 = max(min_green, int(y + alpha * (x - y)))
    c3 = max(min_green, int(0.5 * (x + y)))
    
    return c1, c2, c3


def crossover(
    a: TrafficConfiguration,
    b: TrafficConfiguration
) -> Tuple[TrafficConfiguration, TrafficConfiguration, TrafficConfiguration]:
    """
    Performs linear crossover between two parent traffic signal configurations.

    For each corresponding phase in the parent configurations `a` and `b`, this function
    applies linear crossover on the `green`, `amber`, and `all_red` time values to 
    generate three offspring. Each offspring receives a new combination of these values 
    while preserving the structure and number of phases.

    Linear crossover uses a random blending factor to interpolate or extrapolate 
    between parent values, generating diverse yet meaningful offspring.

    Args:
        a (TrafficConfiguration): The first parent traffic configuration.
        b (TrafficConfiguration): The second parent traffic configuration.

    Returns:
        Tuple[TrafficConfiguration, TrafficConfiguration, TrafficConfiguration]:
            Three offspring configurations resulting from crossover between `a` and `b`.
    """
    child1 = copy.deepcopy(a)
    child2 = copy.deepcopy(a)
    child3 = copy.deepcopy(a)

    for i, (phase_a, phase_b) in enumerate(zip(a, b)):
        g1, g2, g3 = linear_crossover(phase_a.green, phase_b.green)
        a1, a2, a3 = linear_crossover(phase_a.amber, phase_b.amber)
        r1, r2, r3 = linear_crossover(phase_a.all_red, phase_b.all_red)

        child1[i].green = g1
        child1[i].amber = a1
        child1[i].all_red = r1

        child2[i].green = g2
        child2[i].amber = a2
        child2[i].all_red = r2

        child3[i].green = g3
        child3[i].amber = a3
        child3[i].all_red = r3

    return child1, child2, child3


def mutation(
    traffic_configuration: TrafficConfiguration,
    delta: float = 5.0,
    min_green: float = 5.0
) -> TrafficConfiguration:
    """
    Applies Green Time Shift Mutation to a traffic signal configuration.

    This mutation operator selects a random signal phase and adjusts its green time 
    by a small random value in the range [-delta, +delta]. To maintain the overall 
    cycle time, the change is proportionally redistributed among the other phases. 
    Green times are clamped to a minimum threshold to ensure safety and feasibility.

    Args:
        traffic_configuration (TrafficConfiguration): The traffic signal configuration 
            to be mutated, consisting of multiple phases.
        delta (float, optional): Maximum magnitude of the mutation shift in seconds. 
            Defaults to 5.0.
        min_green (float, optional): Minimum allowed green time per phase in seconds. 
            Defaults to 5.0.

    Returns:
        TrafficConfiguration: A new traffic configuration with the green time 
        of one phase mutated and the rest adjusted accordingly.
    """
    num_phases = len(traffic_configuration)
    if num_phases <= 1:
        return traffic_configuration

    mutated_configuration = copy.deepcopy(traffic_configuration)

    i = randint(0, num_phases - 1)
    mutation = uniform(-delta, delta)

    mutated_configuration[i].green += mutation

    if mutated_configuration[i].green < min_green:
        mutation += min_green - mutated_configuration[i].green
        mutated_configuration[i].green = min_green

    redistribution = -mutation / (num_phases - 1)

    for j in range(num_phases):
        if j == i:
            continue
        mutated_configuration[j].green += redistribution
        if mutated_configuration[j].green < min_green:
            mutated_configuration[j].green = min_green
    
    for phase in mutated_configuration:
        if phase.green < min_green:
            phase.green = min_green

    return mutated_configuration

# Declare other functions for the main evolutionary loop

# Evolutionary main loop


def run_evolution(
    population_dict: dict,
    fitness_func: FitnessFunc = fitness,
    generation_limit: int = 20
) -> Tuple[Population, int]:
    # Convert dictionary to list of configurations
    population = list(population_dict.values())
    
    for generation in range(generation_limit):
        print(f"\n--- Generation {generation+1}/{generation_limit} ---")
        # Evaluate and sort population by fitness (lower is better)
        population.sort(key=lambda config: fitness_func(config))
        
        # Keep top 20% elites
        elite_size = max(2, int(len(population) * 0.2))
        next_gen = population[:elite_size]
        
        # Fill remaining slots through selection, crossover and mutation
        while len(next_gen) < len(population):
            # Select parents using tournament selection
            parents = [
                min(sample(population, k=3), 
                key=lambda x: fitness_func(x)
            ) for _ in range(2)
            ]
            
            # Generate offspring
            offspring = crossover(parents[0], parents[1])
            
            # Mutate and add to new generation
            next_gen.extend([
                mutation(child) for child in offspring
                if len(next_gen) < len(population)
            ])
        
        population = next_gen
        
    best_score = fitness_func(population[0])
    print(f"Best fitness: {best_score}")
    print(f"Worst fitness: {fitness_func(population[-1])}")
    
    # Return sorted population and generation count
    return sorted(population, key=lambda config: fitness_func(config)), generation
