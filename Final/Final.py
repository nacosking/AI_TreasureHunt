import pygame
import math
import heapq
import itertools

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
    (3, 0): CELL_TYPE_TRAP4, # This is T4
    (3, 1): CELL_TYPE_EMPTY,
    (3, 2): CELL_TYPE_OBSTACLE,
    (3, 3): CELL_TYPE_TREASURE, # T
    (3, 4): CELL_TYPE_EMPTY,

    (4, 0): CELL_TYPE_REWARD1,
    (4, 1): CELL_TYPE_TREASURE, # T
    (4, 2): CELL_TYPE_OBSTACLE,
    (4, 3): CELL_TYPE_EMPTY,
    (4, 4): CELL_TYPE_OBSTACLE,
    (4, 5): CELL_TYPE_EMPTY,

    (5, -1): CELL_TYPE_EMPTY,
    (5, 0): CELL_TYPE_EMPTY,
    (5, 1): CELL_TYPE_EMPTY,
    (5, 2): CELL_TYPE_TRAP3, # This is T3
    (5, 3): CELL_TYPE_EMPTY,
    (5, 4): CELL_TYPE_REWARD2,

    (6, 0): CELL_TYPE_EMPTY,
    (6, 1): CELL_TYPE_TRAP3, # This is T3
    (6, 2): CELL_TYPE_EMPTY,
    (6, 3): CELL_TYPE_OBSTACLE,
    (6, 4): CELL_TYPE_OBSTACLE,
    (6, 5): CELL_TYPE_EMPTY,

    (7, -1): CELL_TYPE_EMPTY,
    (7, 0): CELL_TYPE_EMPTY,
    (7, 1): CELL_TYPE_REWARD2,
    (7, 2): CELL_TYPE_TREASURE, # T
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
    (9, 2): CELL_TYPE_TREASURE, # T
    (9, 3): CELL_TYPE_EMPTY,
    (9, 4): CELL_TYPE_EMPTY,
}

COLOR_MAP = {
    CELL_TYPE_EMPTY: (255, 255, 255),
    CELL_TYPE_OBSTACLE: (102, 102, 102),
    CELL_TYPE_TRAP1: (207, 159, 255),
    CELL_TYPE_TRAP2: (207, 159, 255),
    CELL_TYPE_TRAP3: (207, 159, 255),
    CELL_TYPE_TRAP4: (207, 159, 255),
    CELL_TYPE_REWARD1: (52, 171, 130 ),
    CELL_TYPE_REWARD2: (52, 171, 130 ),
    CELL_TYPE_TREASURE: (244, 229, 73),
    CELL_TYPE_ENTRY: (173, 216, 230),
}

WIDTH, HEIGHT = 900, 700
HEX_SIZE = 40

# --- A* LOGIC AND HELPERS ---

# Identify all treasure locations and assign unique bit positions for the mask.
TREASURE_LOCATIONS = {}
treasure_count = 0
for coord, cell_type in HEX_MAP.items():
    if cell_type == CELL_TYPE_TREASURE:
        # Assign a unique identifier like 'TREASURE_0', 'TREASURE_1' etc.
        TREASURE_LOCATIONS[f'{CELL_TYPE_TREASURE}_{treasure_count}'] = coord
        treasure_count += 1

# These global variables are derived from the TREASURE_LOCATIONS.
# They are crucial for the A* algorithm's goal checking and heuristic.
TREASURE_NAMES = sorted(TREASURE_LOCATIONS.keys()) # Ensure consistent ordering for bitmasking
TREASURE_BIT_MAP = {name: i for i, name in enumerate(TREASURE_NAMES)}
NUM_TREASURES = len(TREASURE_NAMES)
# A mask where all bits are set, representing all treasures collected/removed
ALL_TREASURES_COLLECTED_MASK = (1 << NUM_TREASURES) - 1

# The entry point for the search.
ENTRY_POINT = (0, 0) # Based on the user's provided HEX_MAP

def get_hex_neighbors(q, r):
    neighbors = []
    # Axial directions for pointy-top hexes, adjusted for staggered columns.
    if q % 2 == 0: # Even column (q)
        directions = [
            (0, -1), (0, 1),   # straight up (r-1), straight down (r+1)
            (1, 0), (1, -1),   # right (q+1, r), right-up (q+1, r-1)
            (-1, 0), (-1, -1)  # left (q-1, r), left-up (q-1, r-1)
        ]
    else: # Odd column (q)
        directions = [
            (0, -1), (0, 1),   # straight up (r-1), straight down (r+1)
            (1, 1), (1, 0),    # right-down (q+1, r+1), right (q+1, r)
            (-1, 1), (-1, 0)   # left-down (q-1, r+1), left (q-1, r)
        ]

    for dq, dr in directions:
        neighbor_q, neighbor_r = q + dq, r + dr
        if (neighbor_q, neighbor_r) in HEX_MAP: # Check if neighbor exists on our map
            neighbors.append((neighbor_q, neighbor_r))
    return neighbors

def hex_distance(q1, r1, q2, r2):
    # Convert to cube coordinates (x, y, z) where x+y+z=0
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
        # Round multipliers to avoid floating point precision issues in hashing/equality checks.
        return hash((self.q, self.r, self.collected_treasures_mask,
                     self.removed_treasures_mask, round(self.gravity_multiplier, 5), # Increased precision for safety
                     round(self.speed_multiplier, 5), self.last_move_direction))

    def __eq__(self, other):
        return (isinstance(other, State) and
                self.q == other.q and
                self.r == other.r and
                self.collected_treasures_mask == other.collected_treasures_mask and
                self.removed_treasures_mask == other.removed_treasures_mask and
                abs(self.gravity_multiplier - other.gravity_multiplier) < 1e-9 and # Use tolerance for float equality
                abs(self.speed_multiplier - other.speed_multiplier) < 1e-9 and
                self.last_move_direction == other.last_move_direction)

    def __lt__(self, other):
        return id(self) < id(other)

def heuristic(state):
    # Sum of minimum distances to all remaining treasures (greedy nearest neighbor)
    remaining = []
    for t_name, t_coords in TREASURE_LOCATIONS.items():
        treasure_bit = TREASURE_BIT_MAP[t_name]
        if not (state.collected_treasures_mask & (1 << treasure_bit)) and \
           not (state.removed_treasures_mask & (1 << treasure_bit)):
            remaining.append(t_coords)

    if not remaining:
        return 0

    h = 0
    curr = (state.q, state.r)
    unvisited = set(remaining)
    while unvisited:
        dists = [(hex_distance(curr[0], curr[1], t[0], t[1]), t) for t in unvisited]
        min_dist, next_t = min(dists)
        h += min_dist
        curr = next_t
        unvisited.remove(next_t)

    base_heuristic = h * 0.25

    # Penalty if currently on a trap tile (encourages safer paths)
    tile = HEX_MAP.get((state.q, state.r), '')
    trap_penalty = 0
    if tile.startswith('TRAP'):
        trap_penalty += 5.0  # Heavily discourage paths through traps
    elif tile.startswith('REWARD'):
        trap_penalty -= 1.0  # Slight encouragement

    return base_heuristic + trap_penalty

def solve_treasure_hunt(allow_trap4_early_stepping=False):
    start_q, start_r = ENTRY_POINT
    initial_state = State(
        q=start_q, r=start_r,
        collected_treasures_mask=0, removed_treasures_mask=0,
        gravity_multiplier=1.0, speed_multiplier=1.0,
        last_move_direction=None
    )

    counter = itertools.count()
    priority_queue = [(heuristic(initial_state), next(counter), 0.0, initial_state, [(start_q, start_r)])]
    g_costs = {initial_state: 0.0}

    while priority_queue:
        f_cost, _, g_cost, current_state, path = heapq.heappop(priority_queue)
        if g_cost > g_costs.get(current_state, float('inf')):
            continue

        if (current_state.collected_treasures_mask | current_state.removed_treasures_mask) == ALL_TREASURES_COLLECTED_MASK:
            return path, g_cost

        current_q, current_r = current_state.q, current_state.r

        for next_q, next_r in get_hex_neighbors(current_q, current_r):
            cell_type_at_next = HEX_MAP.get((next_q, next_r))
            if cell_type_at_next == CELL_TYPE_OBSTACLE:
                continue
            if cell_type_at_next == CELL_TYPE_TRAP4 and not allow_trap4_early_stepping and \
               current_state.collected_treasures_mask != ALL_TREASURES_COLLECTED_MASK:
                continue

            # Copy current state
            new_collected_treasures_mask = current_state.collected_treasures_mask
            new_removed_treasures_mask = current_state.removed_treasures_mask
            new_gravity_multiplier = current_state.gravity_multiplier
            new_speed_multiplier = current_state.speed_multiplier
            new_last_move_direction = (next_q - current_q, next_r - current_r)
            final_agent_q, final_agent_r = next_q, next_r

            # Base move energy cost
            current_move_energy_cost = 1.0 * current_state.speed_multiplier * current_state.gravity_multiplier

            # Additional penalties for traps
            if cell_type_at_next.startswith('TRAP'):
                current_move_energy_cost += 3.0  # Base penalty for all traps

            # Handle tile effects
            if cell_type_at_next == CELL_TYPE_TREASURE:
                for t_name, t_coords in TREASURE_LOCATIONS.items():
                    if t_coords == (next_q, next_r):
                        treasure_bit = TREASURE_BIT_MAP[t_name]
                        if not (new_collected_treasures_mask & (1 << treasure_bit)) and \
                           not (new_removed_treasures_mask & (1 << treasure_bit)):
                            new_collected_treasures_mask |= (1 << treasure_bit)
                        break
            elif cell_type_at_next == CELL_TYPE_TRAP1:
                new_gravity_multiplier *= 2.0
            elif cell_type_at_next == CELL_TYPE_TRAP2:
                new_speed_multiplier *= 2.0
            elif cell_type_at_next == CELL_TYPE_REWARD1:
                new_gravity_multiplier /= 2.0
            elif cell_type_at_next == CELL_TYPE_REWARD2:
                new_speed_multiplier /= 2.0
            elif cell_type_at_next == CELL_TYPE_TRAP4:
                for t_name, t_coords in TREASURE_LOCATIONS.items():
                    treasure_bit = TREASURE_BIT_MAP[t_name]
                    if not (new_collected_treasures_mask & (1 << treasure_bit)):
                        new_removed_treasures_mask |= (1 << treasure_bit)

            # Forced move logic for TRAP3
            if cell_type_at_next == CELL_TYPE_TRAP3:
                if current_state.last_move_direction is not None:
                    dq, dr = current_state.last_move_direction
                    forced_q, forced_r = next_q + dq * 2, next_r + dr * 2
                    if (forced_q, forced_r) not in HEX_MAP or HEX_MAP[(forced_q, forced_r)] == CELL_TYPE_OBSTACLE:
                        continue
                    forced_type = HEX_MAP[(forced_q, forced_r)]
                    if forced_type == CELL_TYPE_TRAP4 and not allow_trap4_early_stepping and \
                       new_collected_treasures_mask != ALL_TREASURES_COLLECTED_MASK:
                        continue

                    final_agent_q, final_agent_r = forced_q, forced_r

                    # Add forced movement penalty
                    current_move_energy_cost += 4.0

                    if forced_type == CELL_TYPE_TREASURE:
                        for t_name, t_coords in TREASURE_LOCATIONS.items():
                            if t_coords == (forced_q, forced_r):
                                treasure_bit = TREASURE_BIT_MAP[t_name]
                                if not (new_collected_treasures_mask & (1 << treasure_bit)) and \
                                   not (new_removed_treasures_mask & (1 << treasure_bit)):
                                    new_collected_treasures_mask |= (1 << treasure_bit)
                                break
                    elif forced_type == CELL_TYPE_TRAP1:
                        new_gravity_multiplier *= 2.0
                    elif forced_type == CELL_TYPE_TRAP2:
                        new_speed_multiplier *= 2.0
                    elif forced_type == CELL_TYPE_REWARD1:
                        new_gravity_multiplier /= 2.0
                    elif forced_type == CELL_TYPE_REWARD2:
                        new_speed_multiplier /= 2.0
                    elif forced_type == CELL_TYPE_TRAP4:
                        for t_name, t_coords in TREASURE_LOCATIONS.items():
                            treasure_bit = TREASURE_BIT_MAP[t_name]
                            if not (new_collected_treasures_mask & (1 << treasure_bit)):
                                new_removed_treasures_mask |= (1 << treasure_bit)

            # Construct next state
            next_state = State(
                q=final_agent_q, r=final_agent_r,
                collected_treasures_mask=new_collected_treasures_mask,
                removed_treasures_mask=new_removed_treasures_mask,
                gravity_multiplier=new_gravity_multiplier,
                speed_multiplier=new_speed_multiplier,
                last_move_direction=new_last_move_direction
            )

            new_g_cost = g_cost + current_move_energy_cost
            if new_g_cost < g_costs.get(next_state, float('inf')):
                g_costs[next_state] = new_g_cost
                h_cost = heuristic(next_state)
                heapq.heappush(priority_queue, (
                    new_g_cost + h_cost, next(counter),
                    new_g_cost, next_state,
                    path + [(final_agent_q, final_agent_r)]
                ))

    return None, None


def textual_path_summary(path, total_cost):
    if path is None:
        print("No path found to collect all treasures under the given rules and map.")
        return
    print("\n" + "="*40)
    print("      --- Solution Found ---")
    print("="*40)
    print(f"Total energy cost: {total_cost:.2f}")
    print(f"Path length (number of moves): {len(path) - 1}")
    print("\n--- Full Path (q, r) and Cell Type ---")
    for i, (q, r) in enumerate(path):
        step_label = " (Start)" if i == 0 else ""
        step_label += " (End)" if i == len(path) - 1 else ""
        cell = HEX_MAP.get((q, r), 'UNKNOWN')
        print(f"Step {i:<3}: ({q}, {r}) -> {cell}{step_label}")
    print("="*40)
# Reverted hex_to_pixel to original behavior to avoid "slanted" visualization
def hex_to_pixel(q, r):
    x = HEX_SIZE * 3/2 * q + 100
    y = HEX_SIZE * math.sqrt(3) * (r + 0.5 * (q % 2)) + 100
    return x, y

def draw_hex(surface, x, y, color, label=''):
    points = [
        (x + HEX_SIZE * math.cos(math.radians(angle)),
         y + HEX_SIZE * math.sin(math.radians(angle)))
        for angle in range(0, 360, 60)
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, (0, 0, 0), points, 2)
    if label:
        font = pygame.font.SysFont(None, 24)
        img = font.render(label, True, (0, 0, 0))
        # Adjust label position slightly for better centering
        text_rect = img.get_rect(center=(x, y))
        surface.blit(img, text_rect)

def visualize_path(path, total_cost):
    if path is None:
        print("No path found to collect all treasures.")
        return
    textual_path_summary(path, total_cost)


def draw_solution_path(surface, path, color=(255, 0, 0)):
    if not path or len(path) < 2:
        return
    points = [hex_to_pixel(q, r) for (q, r) in path]
    pygame.draw.lines(surface, color, False, points, 5)

def main():
    global WIDTH, HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Hexagon Maze Map")
    clock = pygame.time.Clock()

    solution_path = None
    solution_cost = None
    no_solution = False

    running = True
    while running:
        screen.fill((255, 255, 255))
        for (q, r), cell_type in HEX_MAP.items():
            x, y = hex_to_pixel(q, r)
            color = COLOR_MAP.get(cell_type, (211, 211, 211))
            if cell_type == CELL_TYPE_ENTRY:
                label = 'S'
            elif cell_type == CELL_TYPE_TREASURE:
                label = 'T'
            elif cell_type == CELL_TYPE_OBSTACLE:
                label = '#'
            elif cell_type.startswith('TRAP'):
                label = cell_type[-1] # Displays 1, 2, 3, 4
            elif cell_type.startswith('REWARD'):
                label = cell_type[-1] # Displays 1, 2
            else:
                label = ''
            draw_hex(screen, x, y, color, label)

        if solution_path:
            draw_solution_path(screen, solution_path)

        if no_solution:
            font = pygame.font.SysFont(None, 48)
            text = font.render("No solution found!", True, (255, 0, 0))
            rect = text.get_rect(center=(WIDTH // 2, 40))
            screen.blit(text, rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Solving...")
                    # Set allow_trap4_early_stepping to False based on problem understanding
                    solution_path, solution_cost = solve_treasure_hunt(allow_trap4_early_stepping=False) 
                    if solution_path:
                        no_solution = False
                        print(f"Solution found! Total cost: {solution_cost:.2f}")
                        visualize_path(solution_path, solution_cost)
                    else:
                        no_solution = True
                        print("No solution found.")

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    main()