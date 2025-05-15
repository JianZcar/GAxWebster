DIRECTION_PRIORITY = {
    # [left_turn, straight, right_turn] for right-hand traffic
    'north': ['west', 'south', 'east'],
    'south': ['east', 'north', 'west'],
    'east': ['north', 'west', 'south'],
    'west': ['south', 'east', 'north']
}


def smart_lane_assignment(directions, num_lanes):
    result = [[] for _ in range(num_lanes)]
    count = len(directions)
    if count == 1:
        for lane in result:
            lane.append(directions[0])
    elif count == 2:
        if num_lanes == 1:
            result[0].append(directions[0])  # Prefer first
        elif num_lanes == 2:
            result[0].append(directions[0])  # Left
            result[1].append(directions[1])  # Right
        else:
            result[0].append(directions[0])
            result[-1].append(directions[1])
            for i in range(1, num_lanes - 1):
                result[i] = directions[:]  # Shared
    elif count == 3:
        if num_lanes >= 3:
            result[0].append(directions[0])   # Left
            result[1].append(directions[1])   # Straight
            result[-1].append(directions[2])  # Right
            for i in range(2, num_lanes - 1):
                result[i] = directions[:]      # Share mid lanes
        else:
            for i in range(num_lanes):
                result[i] = directions[:]  # Share all
    return result

def calculate_lane_assignments(intersection):
    all_roads = set(intersection.keys())
    assignments = {}

    for road, info in intersection.items():
        num_lanes = info['lanes']
        allowed_dirs = [d for d in DIRECTION_PRIORITY[road] if d in all_roads]
        assignments[road] = smart_lane_assignment(allowed_dirs, num_lanes)

    return assignments

def generate_safe_phases(assignments):
    all_movements = []
    for road in assignments:
        for lane_idx, directions in enumerate(assignments[road]):
            for to_road in directions:
                all_movements.append((road, to_road))

    def has_conflict(p1, p2):
        fr1, to1 = p1
        fr2, to2 = p2
        opposites = {'north': 'south', 'south': 'north',
                     'east': 'west', 'west': 'east'}
        # Opposite roads conflict check
        if fr2 == opposites.get(fr1):
            t1 = DIRECTION_PRIORITY[fr1].index(to1)
            t2 = DIRECTION_PRIORITY[fr2].index(to2)
            # Allow only if both are straight
            if not (t1 == 1 and t2 == 1):
                return True
        # Perpendicular roads conflict check
        perp = {'north': ['east', 'west'], 'south': ['east', 'west'],
                'east': ['north', 'south'], 'west': ['north', 'south']}
        if fr2 in perp.get(fr1, []):
            t1 = DIRECTION_PRIORITY[fr1].index(to1)
            t2 = DIRECTION_PRIORITY[fr2].index(to2)
            # Conflict unless both are right turns
            if not (t1 == 2 and t2 == 2):
                return True
        return False

    phases = []
    while all_movements:
        phase = []
        remaining = []
        for move in all_movements:
            if any(has_conflict(move, p) for p in phase):
                remaining.append(move)
            else:
                phase.append(move)
        phases.append(phase)
        all_movements = remaining
    return phases
    



# Example usage
if __name__ == "__main__":
    t_intersection = {
        'south': {'lanes': 2},
        'east': {'lanes': 2},
        'west': {'lanes': 2},
    }
    
    assignments = calculate_lane_assignments(t_intersection)
    print("Lane Assignments:")
    for road in assignments:
        print(f"{road}: {assignments[road]}")
    
    phases = generate_safe_phases(assignments)
    print("\nSafe Phases:")
    for i, phase in enumerate(phases):
        print(f"Phase {i+1}: {phase}")
