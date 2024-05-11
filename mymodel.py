import numpy as np
import math
import os

# Constants for the display setup
NUM_FRAMES = 25
NUM_ROWS = 7     # Number of LED rows
NUM_COLS = 8     # Number of LEDs per row
RADIUS_CM = 7.5  # Radius of the strip board in cm
HEIGHT_CM = 5.0  # Height of the strip board in cm

# Directory to store the output sketches
output_dir = 'output_sketches6'

template_path = 'template.cpp.in'  # Ensure this path is correct

# declare colour values
black  = 0b000
blue   = 0b001
green  = 0b010
teal   = 0b011
red    = 0b100
pink   = 0b101
yellow = 0b110
white  = 0b111

# Create a function to convert from polar to Cartesian coordinates
def polar_to_cartesian(angle_deg, z, r):
    """
    Convert polar coordinates to Cartesian coordinates.
    - angle_deg: angle in degrees
    - z: vertical position on the strip board
    - r: radial distance from the center
    """
    angle_rad = math.radians(angle_deg)
    x = r * math.cos(angle_rad)  # Horizontal coordinate
    y = r * math.sin(angle_rad)  # Depth coordinate
    return x, y, z

def initialize_image():
    """Initialize a 3D numpy array for the image with all LEDs turned off."""
    return np.zeros((NUM_FRAMES, NUM_ROWS, NUM_COLS), dtype=int)

def plot_point(image, frame, row, col, value):
    """Plot a single point in the image, ensuring it falls within the display bounds."""
    if 0 <= frame < NUM_FRAMES and 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS:
        image[frame][row][col] = value

def draw_sphere(image, center, radius, color):
    """Draw a sphere within the 3D space covered by the LED display."""
    cx, cy, cz = center
    for frame in range(NUM_FRAMES):
        angle_deg = (frame / NUM_FRAMES) * 360
        for row in range(NUM_ROWS):
            z = (row / (NUM_ROWS - 1)) * HEIGHT_CM  # Map row index to height in cm
            for col in range(NUM_COLS):
                r = (col / (NUM_COLS - 1)) * RADIUS_CM  # Map col index to radius in cm
                x, y, _ = polar_to_cartesian(angle_deg, z, r)
                if (x - cx)**2 + (y - cy)**2 + (_ - cz)**2 <= radius**2:
                    plot_point(image, frame, row, col, color)

def draw_line(image, start, end, color):
    """Draw a line between two points in 3D space."""
    # This function will need an implementation of a 3D line drawing algorithm like Bresenham's
    pass

def draw_line(image, start, end, color, thickness=1):
    """Draw a line from start to end in the 3D space with specified thickness."""
    x0, y0, z0 = start
    x1, y1, z1 = end
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    sx, sy, sz = (1 if x0 < x1 else -1, 1 if y0 < y1 else -1, 1 if z0 < z1 else -1)
    if dx >= dy and dx >= dz:
        p1, p2 = 2 * dy - dx, 2 * dz - dx
        while x0 != x1:
            if 0 <= z0 < NUM_ROWS and 0 <= x0 < NUM_COLS:
                plot_point(image, int(y0 / 360 * NUM_FRAMES), z0, x0, color)
            x0 += sx
            if p1 >= 0:
                y0 += sy
                p1 -= 2 * dx
            if p2 >= 0:
                z0 += sz
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    # Additional cases for dy being the largest or dz being the largest go here

def draw_hollow_sphere(image, center, inner_radius, outer_radius, color):
    """Draw a hollow sphere at the center with the specified inner and outer radii."""
    cx, cy, cz = center
    for frame in range(NUM_FRAMES):
        angle_deg = (frame / NUM_FRAMES) * 360
        for row in range(NUM_ROWS):
            z = (row / (NUM_ROWS - 1)) * HEIGHT_CM
            for col in range(NUM_COLS):
                r = (col / (NUM_COLS - 1)) * RADIUS_CM
                x, y, _ = polar_to_cartesian(angle_deg, z, r)
                dist_sq = (x - cx)**2 + (y - cy)**2 + (_ - cz)**2
                if inner_radius**2 < dist_sq <= outer_radius**2:
                    plot_point(image, frame, row, col, color)

def draw_cuboid(image, corner1, corner2, color):
    """Draw a cuboid specified by the diagonal corners."""
    x0, y0, z0 = corner1
    x1, y1, z1 = corner2
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            for z in range(min(z0, z1), max(z0, z1) + 1):
                plot_point(image, int(y / 360 * NUM_FRAMES), z, x, color)

def draw_pyramid(image, base_center, base_size, height, color):
    """Draw a pyramid with a square base."""
    cx, cy, cz = base_center
    half_size = base_size / 2
    # Draw base
    for x in range(cx - half_size, cx + half_size + 1):
        for y in range(cy - half_size, cy + half_size + 1):
            plot_point(image, int(y / 360 * NUM_FRAMES), cz, x, color)
    # Draw sides
    peak = (cx, cy, cz + height)
    corners = [
        (cx - half_size, cy - half_size, cz),
        (cx + half_size, cy - half_size, cz),
        (cx + half_size, cy + half_size, cz),
        (cx - half_size, cy + half_size, cz)
    ]
    for corner in corners:
        draw_line(image, corner, peak, color, 1)  # thickness=1 for lines

# Initialize the image array
def initialize_image():
    return np.zeros((NUM_FRAMES, NUM_ROWS, NUM_COLS), dtype=int)

# Convert voxel value to a string of binary digits
def get_colour_string(voxel):
    return f'0b{voxel:03b}'

# Generate the image array literal in C format
def get_image(image):
    image_data = ""
    for index, ang_slice in enumerate(image):
        slice_data = ',\n'.join(line_for_c(line) for line in ang_slice)
        image_data += f'\t// Slice {index}\n\t{{\n{slice_data}\n\t}},\n'
    return f'const byte image[{NUM_FRAMES}][{NUM_ROWS}][{NUM_COLS}] = {{\n{image_data}}};\n\n'

# Convert a line of voxel data to a C array literal
def line_for_c(line):
    return '\t\t{' + ', '.join(get_colour_string(voxel) for voxel in line) + '}'

# Read the template file and append the generated image array
def get_program(image):
    image_array_literal = get_image(image)
    with open(template_path, 'r') as file:
        template_code = file.read()
    return '#include <SPI.h>\n\n' + image_array_literal + template_code

# Write the final Arduino sketch to a file
def write_sketch(image, name):
    program = get_program(image)
    output_dir = f'output_sketches/{name}'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{name}.ino')
    with open(file_path, 'w') as file:
        file.write(program)
    print(f'Wrote sketch to {file_path}')

# Main function to generate and write an Arduino sketch
def main():
    img = initialize_image()
    # You can call draw functions here to modify the `img` array as needed
    write_sketch(img, '3D_POV_Display')

if __name__ == '__main__':
    main()
