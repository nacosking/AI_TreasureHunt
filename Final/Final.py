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
        TREASURE_LOCATIONS[f'{CELL_TYPE_TREASURE}_{treasure_count}'] = coord
        treasure_count += 1

TREASURE_NAMES = sorted(TREASURE_LOCATIONS.keys())
TREASURE_BIT_MAP = {name: i for i, name in enumerate(TREASURE_NAMES)}
NUM_TREASURES = len(TREASURE_NAMES)
ALL_TREASURES_COLLECTED_MASK = (1 << NUM_TREASURES) - 1
ENTRY_POINT = (0, 0)

def get_hex_neighbors(q, r):
    neighbors = []
    if q % 2 == 0:
        directions = [
            (0, -1), (0, 1),
            (1, 0), (1, -1),
            (-1, 0), (-1, -1)
        ]
    else:
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
                     self.removed_treasures_mask, round(self.gravity_multiplier, 5),
                     round(self.speed_multiplier, 5), self.last_move_direction))

    def __eq__(self, other):
        return (isinstance(other, State) and
                self.q == other.q and
                self.r == other.r and
                self.collected_treasures_mask == other.collected_treasures_mask and
                self.removed_treasures_mask == other.removed_treasures_mask and
                abs(self.gravity_multiplier - other.gravity_multiplier) < 1e-9 and
                abs(self.speed_multiplier - other.speed_multiplier) < 1e-9 and
                self.last_move_direction == other.last_move_direction)

    def __lt__(self, other):
        return id(self) < id(other)

def heuristic(state):
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
    tile = HEX_MAP.get((state.q, state.r), '')
    trap_penalty = 0
    if tile.startswith('TRAP'):
        trap_penalty += 5.0
    elif tile.startswith('REWARD'):
        trap_penalty -= 1.0
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
            new_collected_treasures_mask = current_state.collected_treasures_mask
            new_removed_treasures_mask = current_state.removed_treasures_mask
            new_gravity_multiplier = current_state.gravity_multiplier
            new_speed_multiplier = current_state.speed_multiplier
            new_last_move_direction = (next_q - current_q, next_r - current_r)
            final_agent_q, final_agent_r = next_q, next_r
            current_move_energy_cost = 1.0 * current_state.speed_multiplier * current_state.gravity_multiplier
            if cell_type_at_next.startswith('TRAP'):
                current_move_energy_cost += 3.0
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

class HexMapVisualizer:
    def __init__(self, hex_map, color_map, hex_size=40, padding=4):
        self.hex_map = hex_map
        self.color_map = color_map
        self.hex_size = hex_size
        self.padding = padding

    def hex_to_pixel(self, q, r):
        x = self.hex_size * 3/2 * q + 100
        y = self.hex_size * math.sqrt(3) * (r + 0.5 * (q % 2)) + 100
        return x, y

    def draw_hex(self, surface, x, y, color, label=''):
        radius = self.hex_size - self.padding
        points = [
            (x + radius * math.cos(math.radians(angle)),
             y + radius * math.sin(math.radians(angle)))
            for angle in range(0, 360, 60)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)
        if label:
            font = pygame.font.SysFont("Segoe UI Symbol", 24)
            img = font.render(label, True, (0, 0, 0))
            text_rect = img.get_rect(center=(x, y))
            surface.blit(img, text_rect)

    def draw_cell(self, screen, q, r, cell_type):
        x, y = self.hex_to_pixel(q, r)
        color = self.color_map.get(cell_type, (211, 211, 211))
        label = ''
        if cell_type == CELL_TYPE_ENTRY:
            label = 'S'
        elif cell_type == CELL_TYPE_TREASURE:
            label = 'T'
        elif cell_type == CELL_TYPE_OBSTACLE:
            label = '#'
        elif cell_type == CELL_TYPE_TRAP1:
            label = '⊖'
        elif cell_type == CELL_TYPE_TRAP2:
            label = '⊕'
        elif cell_type == CELL_TYPE_TRAP3:
            label = '⊗'
        elif cell_type == CELL_TYPE_TRAP4:
            label = '⊘'
        elif cell_type == CELL_TYPE_REWARD1:
            label = '⊞'
        elif cell_type == CELL_TYPE_REWARD2:
            label = '⊠'
        self.draw_hex(screen, x, y, color, label)

    def draw_solution_path(self, surface, path, color=(255, 0, 0), node_radius=8, line_width=5, arrow_size=18):
        if not path or len(path) < 2:
            return
        points = [self.hex_to_pixel(q, r) for (q, r) in path]
        pygame.draw.lines(surface, color, False, points, line_width)
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.draw_arrow(surface, x1, y1, x2, y2, color, arrow_size)
        pygame.draw.circle(surface, (0, 200, 0), (int(points[0][0]), int(points[0][1])), node_radius + 2)
        pygame.draw.circle(surface, (0, 0, 200), (int(points[-1][0]), int(points[-1][1])), node_radius + 2)

    def draw_arrow(self, surface, x1, y1, x2, y2, color, arrow_size=18):
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_tip = (x2, y2)
        left = (x2 - arrow_size * math.cos(angle - math.pi / 6),
                y2 - arrow_size * math.sin(angle - math.pi / 6))
        right = (x2 - arrow_size * math.cos(angle + math.pi / 6),
                 y2 - arrow_size * math.sin(angle + math.pi / 6))
        pygame.draw.polygon(surface, color, [arrow_tip, left, right])

def visualize_path(path, total_cost):
    if path is None:
        print("No path found to collect all treasures.")
        return
    textual_path_summary(path, total_cost)

def main():
    global WIDTH, HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Hexagon Maze Map")
    clock = pygame.time.Clock()

    visualizer = HexMapVisualizer(HEX_MAP, COLOR_MAP, HEX_SIZE, padding=4)
    solution_path = None
    solution_cost = None
    no_solution = False

    running = True
    while running:
        screen.fill((255, 255, 255))
        for (q, r), cell_type in HEX_MAP.items():
            visualizer.draw_cell(screen, q, r, cell_type)

        if solution_path:
            visualizer.draw_solution_path(screen, solution_path)

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