import pygame
import math

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
        surface.blit(img, (x - 10, y - 12))

def main():
    global WIDTH, HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Hexagon Maze Map")
    clock = pygame.time.Clock()

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
                label = cell_type[-1]
            elif cell_type.startswith('REWARD'):
                label = cell_type[-1]
            else:
                label = ''
            draw_hex(screen, x, y, color, label)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    main()
