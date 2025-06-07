import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# --- Cell Type Definitions ---
# These constants represent the different types of cells on your hexagonal map.
# You will use these values in the HEX_MAP dictionary.
CELL_TYPE_EMPTY = 'EMPTY'
CELL_TYPE_OBSTACLE = 'OBSTACLE'
CELL_TYPE_TRAP1 = 'TRAP1' # Symbol: Theta (θ)
CELL_TYPE_TRAP2 = 'TRAP2' # Symbol: Plus in circle (⊕)
CELL_TYPE_TRAP3 = 'TRAP3' # Symbol: Diamond in circle (◈)
CELL_TYPE_TRAP4 = 'TRAP4' # Symbol: Zero in circle (⓪)
CELL_TYPE_REWARD1 = 'REWARD1' # Symbol: Field (田)
CELL_TYPE_REWARD2 = 'REWARD2' # Symbol: Checkmark in box (☑)
CELL_TYPE_TREASURE = 'TREASURE' # Symbol: Orange hexagon
CELL_TYPE_ENTRY = 'ENTRY' # Blue arrow pointing to it (start point)


# --- Hexagonal Map Definition ---
# This is the core dictionary where you will define your map.
# Keys are (q, r) axial coordinates, and values are the CELL_TYPE constants.
#
# How to define your map (guided by your assignment's image):
# 1.  Axial Coordinates (q, r):
#     - 'q' is the column index (horizontal).
#     - 'r' is the row index (vertical).
#     - Pick an arbitrary cell as (0,0) as your reference if you're starting from scratch.
#       (For consistency with your previous map, I've kept the entry point at (0,2) as a reference.)
#     - To move right to the next column:
#         - If you're in an even 'q' column (e.g., q=0, 2), moving right to an odd 'q+1' column:
#           - To move roughly straight right: `r` stays the same. (e.g., (0,2) -> (1,2))
#           - To move slightly down-right: `r` increases by 1. (e.g., (0,2) -> (1,3))
#           - To move slightly up-right: `r` decreases by 1. (e.g., (0,2) -> (1,1))
#         - If you're in an odd 'q' column (e.g., q=1, 3), moving right to an even 'q+1' column:
#           - To move roughly straight right: `r` stays the same. (e.g., (1,2) -> (2,2))
#           - To move slightly down-right: `r` increases by 1. (e.g., (1,2) -> (2,3))
#           - To move slightly up-right: `r` decreases by 1. (e.g., (1,2) -> (2,1))
#     - To move vertically within the same column:
#         - 'q' stays the same.
#         - 'r' changes by `+1` for down, `-1` for up.
#
# 2.  Mapping cells:
#     - Go row by row or column by column in your image.
#     - For each hex, determine its (q, r) coordinates relative to your chosen origin.
#     - Determine its type based on the symbol/color in the legend.
#     - Add an entry to the HEX_MAP dictionary: `(q, r): CELL_TYPE_CONSTANT`
#
# Example (part of the map from your assignment):
# (0, 2): CELL_TYPE_ENTRY,
# (1, 2): CELL_TYPE_TREASURE, # Treasure is orange
# (1, 1): CELL_TYPE_TRAP3,    # Diamond in circle
# (0, 3): CELL_TYPE_OBSTACLE, # Dark grey cell
#
# Current map definition (from your assignment image):
HEX_MAP = {
    # q, r : type
    (0, 0): CELL_TYPE_EMPTY,
    (0, 1): CELL_TYPE_EMPTY,
    (0, 2): CELL_TYPE_ENTRY, # Entry point indicated by blue arrow
    (0, 3): CELL_TYPE_OBSTACLE, # Dark grey cell
    (0, 4): CELL_TYPE_EMPTY,
    (0, 5): CELL_TYPE_EMPTY,

    (1, -1): CELL_TYPE_TRAP1, # Top-left Trap 1 (theta)
    (1, 0): CELL_TYPE_REWARD2, # Below Trap 1 (checkmark in box)
    (1, 1): CELL_TYPE_TRAP3, # Left of Treasure 1 (diamond in circle)
    (1, 2): CELL_TYPE_TREASURE, # Treasure 1 (orange)
    (1, 3): CELL_TYPE_TRAP2, # Below Treasure 1 (plus in circle)
    (1, 4): CELL_TYPE_REWARD2, # Bottom-left Reward 2 (checkmark in box)

    (2, 0): CELL_TYPE_OBSTACLE, # Dark grey cell
    (2, 1): CELL_TYPE_REWARD1, # Left of Treasure 2 (field)
    (2, 2): CELL_TYPE_TREASURE, # Treasure 2 (orange)
    (2, 3): CELL_TYPE_REWARD1, # Below Treasure 2 (field)
    (2, 4): CELL_TYPE_OBSTACLE, # Dark grey cell

    (3, -1): CELL_TYPE_OBSTACLE, # Above first Treasure (dark grey)
    (3, 0): CELL_TYPE_REWARD2,
    (3, 1): CELL_TYPE_TREASURE, # Treasure 3 (orange)
    (3, 2): CELL_TYPE_TRAP3, # Below Treasure 3 (diamond in circle)
    (3, 3): CELL_TYPE_REWARD1, # Below Treasure 4 (field) - Rechecked from image, this is R1
    (3, 4): CELL_TYPE_TREASURE, # Treasure 4 (orange) - This was R1, but the image shows orange hex here
    (3, 5): CELL_TYPE_TRAP2, # Bottom Trap 2 (plus in circle)

    (4, 0): CELL_TYPE_TREASURE, # Treasure 5 (orange)
    (4, 1): CELL_TYPE_OBSTACLE, # Dark grey cell
    (4, 2): CELL_TYPE_TREASURE, # Treasure 6 (orange)
    (4, 3): CELL_TYPE_OBSTACLE, # Dark grey cell
    (4, 4): CELL_TYPE_TREASURE, # Treasure 7 (orange) - Counted 7 treasures in total
    (4, 5): CELL_TYPE_EMPTY, # Added based on visual extent

    (5, -1): CELL_TYPE_EMPTY,
    (5, 0): CELL_TYPE_TRAP4, # Top-right Trap 4 (zero in circle)
    (5, 1): CELL_TYPE_REWARD2, # Checkmark in box
    (5, 2): CELL_TYPE_OBSTACLE, # Dark grey cell
    (5, 3): CELL_TYPE_TREASURE, # Treasure 8 (orange) - If 7 treasures, then this is not T8.
                               # Let's re-verify treasure count and map to avoid confusion.
                               # From image: (1,2), (2,2), (3,1), (3,4), (4,0), (4,2), (4,4), (5,3) -- 8 treasures.
                               # Corrected treasure count based on a careful re-scan of the orange hexagons.
                               # Let's adjust the map for 8 treasures:
    (5, 3): CELL_TYPE_TREASURE, # Treasure 8
    (5, 4): CELL_TYPE_REWARD2, # Checkmark in box
    (5, 5): CELL_TYPE_TRAP1, # Bottom-right Trap 1 (theta)

    (6, 0): CELL_TYPE_REWARD1, # Top-right Reward 1 (field)
    (6, 1): CELL_TYPE_EMPTY,
    (6, 2): CELL_TYPE_OBSTACLE, # Dark grey cell
    (6, 3): CELL_TYPE_TRAP2, # Rightmost Trap 2 (plus in circle)
    (6, 4): CELL_TYPE_TRAP1, # Rightmost Trap 1 (theta)
    (6, 5): CELL_TYPE_EMPTY, # Bottom-right empty cell
}

# The entry point, as indicated by the blue arrow in the screenshot.
ENTRY_POINT = (0, 2)


# --- Hex grid utility functions ---
def hex_to_cartesian(q, r, hex_size=1.0):
    """
    Converts axial hexagonal coordinates (q, r) to Cartesian (x, y) coordinates
    for a pointy-top hexagonal grid.
    """
    # For pointy-top hexes, standard conversion:
    x = hex_size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
    y = hex_size * (3/2 * r)
    return x, y


def plot_hex_grid(highlight_cells=None, highlight_color='blue', title="Hexagonal Map"):
    """
    Visualizes the hexagonal grid using matplotlib.

    Args:
        highlight_cells (list of (q, r) tuples, optional): A list of coordinates
            to highlight on the map (e.g., a path).
        highlight_color (str, optional): The color to use for highlighting. Defaults to 'blue'.
        title (str, optional): The title for the plot.
    """
    fig, ax = plt.subplots(1, figsize=(12, 10)) # Increased figure size for better visibility
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#f0f0f0') # Light gray background for contrast

    # Define colors for different cell types
    colors = {
        CELL_TYPE_EMPTY: '#D3D3D3',  # Light gray
        CELL_TYPE_OBSTACLE: '#555555', # Dark gray
        CELL_TYPE_TRAP1: '#FFCCCC',  # Light red (Trap 1: θ)
        CELL_TYPE_TRAP2: '#FF9999',  # Medium red (Trap 2: ⊕)
        CELL_TYPE_TRAP3: '#FF6666',  # Red (Trap 3: ◈)
        CELL_TYPE_TRAP4: '#FF3333',  # Dark red (Trap 4: ⓪)
        CELL_TYPE_REWARD1: '#CCFFCC', # Light green (Reward 1: 田)
        CELL_TYPE_REWARD2: '#99FF99', # Medium green (Reward 2: ☑)
        CELL_TYPE_TREASURE: '#FFD700', # Gold (Treasure: Orange hexagon)
        CELL_TYPE_ENTRY: '#ADD8E6',   # Light blue (Entry point)
    }

    # Labels for the legend, including symbols for traps/rewards
    labels = {
        CELL_TYPE_EMPTY: 'Empty Cell',
        CELL_TYPE_OBSTACLE: 'Obstacle (#)',
        CELL_TYPE_TRAP1: 'Trap 1 (θ)',
        CELL_TYPE_TRAP2: 'Trap 2 (⊕)',
        CELL_TYPE_TRAP3: 'Trap 3 (◈)',
        CELL_TYPE_TRAP4: 'Trap 4 (⓪)',
        CELL_TYPE_REWARD1: 'Reward 1 (田)',
        CELL_TYPE_REWARD2: 'Reward 2 (☑)',
        CELL_TYPE_TREASURE: 'Treasure (T)',
        CELL_TYPE_ENTRY: 'Entry Point (S)',
        'HIGHLIGHT': f'Highlighted Cells ({highlight_color.capitalize()})'
    }

    hex_size = 1.0 # Standard size for hex drawing
    patch_list = [] # For legend creation

    # Draw all hexagonal cells and color them based on type
    for (q, r), cell_type in HEX_MAP.items():
        x_center, y_center = hex_to_cartesian(q, r, hex_size)

        color = colors.get(cell_type, colors[CELL_TYPE_EMPTY]) # Default to empty if not found

        hex_patch = mpatches.RegularPolygon(
            (x_center, y_center),
            numVertices=6,
            radius=hex_size * 0.95, # Slightly smaller to show gaps between hexes
            orientation=math.radians(30), # For pointy-top hexes (rotates by 30 degrees)
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(hex_patch)

        # Add text labels for specific cell types as per the diagram legend/common sense
        if cell_type == CELL_TYPE_ENTRY:
            ax.text(x_center, y_center, 'S', ha='center', va='center', fontsize=12, weight='bold', color='darkblue')
        elif cell_type == CELL_TYPE_TREASURE:
            ax.text(x_center, y_center, 'T', ha='center', va='center', fontsize=12, weight='bold', color='darkgoldenrod')
        elif cell_type == CELL_TYPE_OBSTACLE:
            ax.text(x_center, y_center, '#', ha='center', va='center', fontsize=12, weight='bold', color='white')
        elif cell_type == CELL_TYPE_TRAP1: # θ
            ax.text(x_center, y_center, 'θ', ha='center', va='center', fontsize=12, color='darkred', weight='bold')
        elif cell_type == CELL_TYPE_TRAP2: # ⊕
            ax.text(x_center, y_center, '⊕', ha='center', va='center', fontsize=12, color='darkred', weight='bold')
        elif cell_type == CELL_TYPE_TRAP3: # ◈
            ax.text(x_center, y_center, '◈', ha='center', va='center', fontsize=12, color='darkred', weight='bold')
        elif cell_type == CELL_TYPE_TRAP4: # ⓪
            ax.text(x_center, y_center, '⓪', ha='center', va='center', fontsize=12, color='darkred', weight='bold')
        elif cell_type == CELL_TYPE_REWARD1: # 田
            ax.text(x_center, y_center, '田', ha='center', va='center', fontsize=12, color='darkgreen', weight='bold')
        elif cell_type == CELL_TYPE_REWARD2: # ☑
            ax.text(x_center, y_center, '☑', ha='center', va='center', fontsize=12, color='darkgreen', weight='bold')

    # Highlight specific cells if provided (e.g., to trace a path later)
    if highlight_cells:
        for q, r in highlight_cells:
            x_center, y_center = hex_to_cartesian(q, r, hex_size)
            highlight_patch = mpatches.RegularPolygon(
                (x_center, y_center),
                numVertices=6,
                radius=hex_size * 0.8, # Slightly smaller to show underlying hex
                orientation=math.radians(30),
                facecolor=highlight_color,
                edgecolor='none', # No border for highlights
                alpha=0.6, # Semi-transparent
                zorder=2 # Ensure highlights are on top of base cells but below text
            )
            ax.add_patch(highlight_patch)

    # Add patches for the legend
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_ENTRY], label=labels[CELL_TYPE_ENTRY]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_OBSTACLE], label=labels[CELL_TYPE_OBSTACLE]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_TREASURE], label=labels[CELL_TYPE_TREASURE]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_TRAP1], label=labels[CELL_TYPE_TRAP1]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_TRAP2], label=labels[CELL_TYPE_TRAP2]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_TRAP3], label=labels[CELL_TYPE_TRAP3]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_TRAP4], label=labels[CELL_TYPE_TRAP4]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_REWARD1], label=labels[CELL_TYPE_REWARD1]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_REWARD2], label=labels[CELL_TYPE_REWARD2]))
    patch_list.append(mpatches.Patch(color=colors[CELL_TYPE_EMPTY], label=labels[CELL_TYPE_EMPTY])) # Empty cell legend
    if highlight_cells: # Only add highlight legend if cells are highlighted
        patch_list.append(mpatches.Patch(color=highlight_color, label=labels['HIGHLIGHT'], alpha=0.6))

    ax.legend(handles=patch_list, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    # Adjust plot limits to fit all hexes plus some padding
    all_x = [hex_to_cartesian(q, r, hex_size)[0] for q, r in HEX_MAP.keys()]
    all_y = [hex_to_cartesian(q, r, hex_size)[1] for q, r in HEX_MAP.keys()]
    ax.set_xlim(min(all_x) - hex_size * 2, max(all_x) + hex_size * 2)
    ax.set_ylim(min(all_y) - hex_size * 2, max(all_y) + hex_size * 2)

    plt.tight_layout() # Adjust layout to prevent labels/legend from overlapping
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Generating Hexagonal Map Visualization...")

    # You can call plot_hex_grid() without any highlight_cells
    # to just see the raw map.
    plot_hex_grid(title="Your Custom Hexagonal Map")

    # Example of how you might highlight some cells (e.g., a test path)
    # test_path = [(0, 2), (1, 2), (2, 2), (3, 2)]
    # plot_hex_grid(highlight_cells=test_path, highlight_color='purple',
    #               title="Hexagonal Map with a Test Path")
