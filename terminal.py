import math
import heapq

# --- Cell type constants ---
CELL_TYPE_EMPTY = 'EMPTY'
CELL_TYPE_OBSTACLE = 'OBSTACLE'
CELL_TYPE_TRAP1 = 'TRAP1'
CELL_TYPE_TRAP2 = 'TRAP2'
CELL_TYPE_TRAP3 = 'TRAP3'
CELL_TYPE_TRAP4 = 'TRAP4'
CELL_TYPE_REWARD1 = 'REWARD1'
CELL_TYPE_REWARD2 = 'REWARD2'
CELL_TYPE_TREASURE = 'TREASURE'
CELL_TYPE_ENTRY = 'ENTRY'

# --- HEX_MAP definition ---
HEX_MAP = {
    (0, 0): CELL_TYPE_ENTRY,
    (0, 1): CELL_TYPE_EMPTY,
    (0, 2): CELL_TYPE_EMPTY,
    (0, 3): CELL_TYPE_OBSTACLE,
    (0, 4): CELL_TYPE_EMPTY,
    (0, 5): CELL_TYPE_EMPTY,

    (1, -1): CELL_TYPE_EMPTY,
    (1, 0): CELL_TYPE_TRAP1,
    (1, 1): CELL_TYPE_EMPTY,
    (1, 2): CELL_TYPE_REWARD1,
    (1, 3): CELL_TYPE_EMPTY,
    (1, 4): CELL_TYPE_EMPTY,

    (2, 0): CELL_TYPE_EMPTY,
    (2, 1): CELL_TYPE_EMPTY,
    (2, 2): CELL_TYPE_OBSTACLE,
    (2, 3): CELL_TYPE_EMPTY,
    (2, 4): CELL_TYPE_TRAP3,
    (2, 5): CELL_TYPE_EMPTY,

    (3, -1): CELL_TYPE_EMPTY,
    (3, 0): CELL_TYPE_TRAP4,
    (3, 1): CELL_TYPE_EMPTY,
    (3, 2): CELL_TYPE_OBSTACLE,
    (3, 3): CELL_TYPE_TREASURE,
    (3, 4): CELL_TYPE_EMPTY,

    (4, 0): CELL_TYPE_REWARD1,
    (4, 1): CELL_TYPE_TREASURE,
    (4, 2): CELL_TYPE_OBSTACLE,
    (4, 3): CELL_TYPE_EMPTY,
    (4, 4): CELL_TYPE_OBSTACLE,
    (4, 5): CELL_TYPE_EMPTY,

    (5, -1): CELL_TYPE_EMPTY,
    (5, 0): CELL_TYPE_EMPTY,
    (5, 1): CELL_TYPE_EMPTY,
    (5, 2): CELL_TYPE_TRAP3,
    (5, 3): CELL_TYPE_EMPTY,
    (5, 4): CELL_TYPE_REWARD2,

    (6, 0): CELL_TYPE_EMPTY,
    (6, 1): CELL_TYPE_TRAP3,
    (6, 2): CELL_TYPE_EMPTY,
    (6, 3): CELL_TYPE_OBSTACLE,
    (6, 4): CELL_TYPE_OBSTACLE,
    (6, 5): CELL_TYPE_EMPTY,

    (7, -1): CELL_TYPE_EMPTY,
    (7, 0): CELL_TYPE_EMPTY,
    (7, 1): CELL_TYPE_REWARD2,
    (7, 2): CELL_TYPE_TREASURE,
    (7, 3): CELL_TYPE_OBSTACLE,
    (7, 4): CELL_TYPE_EMPTY,

    (8, 0): CELL_TYPE_EMPTY,
    (8, 1): CELL_TYPE_OBSTACLE,
    (8, 2): CELL_TYPE_TRAP1,
    (8, 3): CELL_TYPE_EMPTY,
    (8, 4): CELL_TYPE_EMPTY,
    (8, 5): CELL_TYPE_EMPTY,

    (9, -1): CELL_TYPE_EMPTY,
    (9, 0): CELL_TYPE_EMPTY,
    (9, 1): CELL_TYPE_EMPTY,
    (9, 2): CELL_TYPE_TREASURE,
    (9, 3): CELL_TYPE_EMPTY,
    (9, 4): CELL_TYPE_EMPTY,
}

# Identify all treasure locations and assign unique bit positions for the mask.
TREASURE_LOCATIONS = {}
treasure_count = 0
for coord, cell_type in HEX_MAP.items():
    if cell_type == CELL_TYPE_TREASURE:
        TREASURE_LOCATIONS[f'{CELL_TYPE_TREASURE}_{treasure_count}'] = coord
        treasure_count += 1

TREASURE_NAMES = sorted(TREASURE_LOCATIONS.keys())
TREASURE_BIT_MAP = {name: i for i, name in enumerate(TREASURE_NAMES)}
NUM_TREASURES = len(TREASURE_NAMES)
ALL_TREASURES_COLLECTED_MASK = (1 << NUM_TREASURES) - 1

ENTRY_POINT = (0, 0)  # Starting point

def get_hex_neighbors(q, r):
    neighbors = []
    if q % 2 == 0:  # Even column
        directions = [
            (0, -1), (0, 1),
            (1, 0), (1, -1),
            (-1, 0), (-1, -1)
        ]
    else:  # Odd column
        directions = [
            (0, -1), (0, 1),
            (1, 1), (1, 0),
            (-1, 1), (-1, 0)
        ]
    for dq, dr in directions:
        neighbor_q, neighbor_r = q + dq, r + dr
        if (neighbor_q, neighbor_r) in HEX_MAP:
            neighbors.append((neighbor_q, neighbor_r))
    return neighbors

def hex_distance(q1, r1, q2, r2):
    s1 = -q1 - r1
    s2 = -q2 - r2
    return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2

class State:
    def __init__(self, q, r, collected_treasures_mask, removed_treasures_mask,
                 gravity_multiplier, speed_multiplier, last_move_direction):
        self.q = q
        self.r = r
        self.collected_treasures_mask = collected_treasures_mask
        self.removed_treasures_mask = removed_treasures_mask
        self.gravity_multiplier = gravity_multiplier
        self.speed_multiplier = speed_multiplier
        self.last_move_direction = last_move_direction

    def __hash__(self):
        return hash((self.q, self.r, self.collected_treasures_mask,
                     self.removed_treasures_mask, self.gravity_multiplier,
                     self.speed_multiplier, self.last_move_direction))

    def __eq__(self, other):
        return (isinstance(other, State) and
                self.q == other.q and
                self.r == other.r and
                self.collected_treasures_mask == other.collected_treasures_mask and
                self.removed_treasures_mask == other.removed_treasures_mask and
                self.gravity_multiplier == other.gravity_multiplier and
                self.speed_multiplier == other.speed_multiplier and
                self.last_move_direction == other.last_move_direction)

    def __lt__(self, other):
        return id(self) < id(other)

def heuristic(state):
    if (state.collected_treasures_mask | state.removed_treasures_mask) == ALL_TREASURES_COLLECTED_MASK:
        return 0

    min_dist_to_treasure = float('inf')
    found_uncollected_and_not_removed = False

    for t_name, t_coords in TREASURE_LOCATIONS.items():
        treasure_bit = TREASURE_BIT_MAP[t_name]
        if not (state.collected_treasures_mask & (1 << treasure_bit)) and \
           not (state.removed_treasures_mask & (1 << treasure_bit)):
            found_uncollected_and_not_removed = True
            dist = hex_distance(state.q, state.r, t_coords[0], t_coords[1])
            min_dist_to_treasure = min(min_dist_to_treasure, dist)

    if not found_uncollected_and_not_removed:
        return 0

    return min_dist_to_treasure

def solve_treasure_hunt():
    start_q, start_r = ENTRY_POINT
    initial_state = State(
        q=start_q,
        r=start_r,
        collected_treasures_mask=0,
        removed_treasures_mask=0,
        gravity_multiplier=1.0,
        speed_multiplier=1.0,
        last_move_direction=None
    )

    priority_queue = [(0 + heuristic(initial_state), 0.0, initial_state, [(start_q, start_r)])]
    g_costs = {initial_state: 0.0}
    visited_states = set()

    while priority_queue:
        f_cost, g_cost, current_state, path = heapq.heappop(priority_queue)
        if current_state in visited_states:
            continue
        visited_states.add(current_state)

        if (current_state.collected_treasures_mask | current_state.removed_treasures_mask) == ALL_TREASURES_COLLECTED_MASK:
            return path, g_cost

        current_q, current_r = current_state.q, current_state.r

        for next_q, next_r in get_hex_neighbors(current_q, current_r):
            cell_type = HEX_MAP.get((next_q, next_r))

            if cell_type == CELL_TYPE_OBSTACLE or cell_type == CELL_TYPE_TRAP4:
                continue

            base_step_cost = 1.0
            move_effective_steps = base_step_cost * current_state.speed_multiplier
            move_energy_cost = move_effective_steps * current_state.gravity_multiplier

            new_collected_treasures_mask = current_state.collected_treasures_mask
            new_removed_treasures_mask = current_state.removed_treasures_mask
            new_gravity_multiplier = current_state.gravity_multiplier
            new_speed_multiplier = current_state.speed_multiplier

            new_last_move_direction = (next_q - current_q, next_r - current_r)
            actual_next_location = (next_q, next_r)

            if cell_type == CELL_TYPE_TREASURE:
                for t_name, t_coords in TREASURE_LOCATIONS.items():
                    if t_coords == (next_q, next_r):
                        treasure_bit = TREASURE_BIT_MAP[t_name]
                        if not (new_collected_treasures_mask & (1 << treasure_bit)) and \
                           not (new_removed_treasures_mask & (1 << treasure_bit)):
                            new_collected_treasures_mask |= (1 << treasure_bit)
                        break
            elif cell_type == CELL_TYPE_TRAP1:
                new_gravity_multiplier *= 2.0
            elif cell_type == CELL_TYPE_TRAP2:
                new_speed_multiplier *= 2.0
            elif cell_type == CELL_TYPE_TRAP3:
                if current_state.last_move_direction is not None:
                    dq, dr = current_state.last_move_direction
                    forced_q, forced_r = next_q + dq * 2, next_r + dr * 2
                    if (forced_q, forced_r) in HEX_MAP and HEX_MAP[(forced_q, forced_r)] != CELL_TYPE_OBSTACLE:
                        actual_next_location = (forced_q, forced_r)
                        next_q, next_r = actual_next_location
                        move_energy_cost *= 3.0
                    else:
                        continue
            elif cell_type == CELL_TYPE_REWARD1:
                new_gravity_multiplier = max(1.0, new_gravity_multiplier / 2.0)
            elif cell_type == CELL_TYPE_REWARD2:
                new_speed_multiplier = max(1.0, new_speed_multiplier / 2.0)

            new_state = State(
                q=actual_next_location[0],
                r=actual_next_location[1],
                collected_treasures_mask=new_collected_treasures_mask,
                removed_treasures_mask=new_removed_treasures_mask,
                gravity_multiplier=new_gravity_multiplier,
                speed_multiplier=new_speed_multiplier,
                last_move_direction=new_last_move_direction
            )

            new_g_cost = g_cost + move_energy_cost

            if new_state in visited_states:
                continue

            if new_state not in g_costs or new_g_cost < g_costs[new_state]:
                g_costs[new_state] = new_g_cost
                f_cost = new_g_cost + heuristic(new_state)
                heapq.heappush(priority_queue, (f_cost, new_g_cost, new_state, path + [actual_next_location]))

    return None, None  # No solution found

# You can test the function like this:
if __name__ == "__main__":
    path, cost = solve_treasure_hunt()
    if path:
        print("Found path to collect all treasures:")
        for step in path:
            print(step)
        print(f"Total cost: {cost}")
    else:
        print("No path found.")
