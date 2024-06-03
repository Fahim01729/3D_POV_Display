import numpy as np
import math
from random import randint
import os

# Directory to store the output sketches
output_directory = 'output_files'
template_file = '3D_POV_Display_template.cpp.in'  # Ensure this path is correct

# Color values
COLOR_BLACK  = 0b000
COLOR_BLUE   = 0b001
COLOR_GREEN  = 0b010
COLOR_TEAL   = 0b011
COLOR_RED    = 0b100
COLOR_PINK   = 0b101
COLOR_YELLOW = 0b110
COLOR_WHITE  = 0b111

### Basic Functions ###

def create_empty_image():
    """
    Return an empty numpy 3-D array with the dimensions of the image
    """
    return np.zeros((25, 4, 8), dtype=int)  # Adjusted radius range to 4 (0 to 3)

def generate_random_point():
    """
    Generate a random point in the defined space
    """
    return (randint(0, 24), randint(0, 3), randint(0, 3))  # Adjusted radius range to 3

def pixels_to_mm(pixels, inner_radius_mm=20, gap_mm=12):
    """
    Convert pixels to mm based on a specified pixel density for the given radii.
    
    Parameters:
    pixels (float): The number of pixels to convert.
    inner_radius_mm (float): The inner radius in mm. Default is 20 mm (2.0 cm).
    gap_mm (float): The gap between each pixel in mm. Default is 12 mm (1.2 cm).
    
    Returns:
    float: The converted distance in mm.
    """
    return inner_radius_mm + pixels * gap_mm

def polar_to_cartesian(angle, height, radius, total_angles=25, inner_radius_mm=20, gap_mm=12):
    """
    Convert polar coordinates to Cartesian coordinates.
    Polar: angle [0-(total_angles-1)], height [0-3], radius [0-3]
    Cartesian: x [mm], y [mm], z [mm]
    """
    # Convert the radius to mm
    radius_mm = pixels_to_mm(radius, inner_radius_mm=inner_radius_mm, gap_mm=gap_mm)
    
    # Convert the angle to radians based on the total angles
    angle_rad = math.radians(angle * (360 / total_angles))
    
    x_mm = math.cos(angle_rad) * radius_mm
    y_mm = math.sin(angle_rad) * radius_mm
    
    # Convert height to mm
    height_mm = height * gap_mm  # Each height level is separated by 1.2 cm

    return (x_mm, y_mm, height_mm)

def distance_between_3d_points(p1, p2):
    """
    Calculate the Euclidean distance between two 3D points.
    
    Parameters:
    p1 (tuple): The first point (x1, y1, z1).
    p2 (tuple): The second point (x2, y2, z2).
    
    Returns:
    float: The Euclidean distance between the points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def distance_point_to_line_segment(point, line_start, line_end):
    """
    Calculate the distance from a point to a line segment.
    
    Parameters:
    point (tuple): The point (px, py, pz).
    line_start (tuple): The start point of the line segment (vx, vy, vz).
    line_end (tuple): The end point of the line segment (wx, wy, wz).
    
    Returns:
    float: The distance from the point to the line segment.
    """
    if line_start == line_end:
        return distance_between_3d_points(point, line_start)
    line_length_squared = distance_between_3d_points(line_start, line_end) ** 2
    t = max(0, min(1, np.dot(np.subtract(point, line_start), np.subtract(line_end, line_start)) / line_length_squared))
    projection = tuple(line_start[i] + t * (line_end[i] - line_start[i]) for i in range(3))
    return distance_between_3d_points(point, projection)

### Drawing Functions ###

def draw_line(image, start, end, color, thickness):
    """
    Or-equals all pixels in {image} that are closer to the line (between {start}
    and {end}) than {thickness} with the {color} value (int, 0-7).

    All arguments are in Cartesian coordinates and mm
    """
    plot_boolean_function(image, lambda x, y, z: distance_point_to_line_segment((x, y, z), start, end) <= thickness, color)

def draw_line_polar(image, start, end, color, thickness):
    """
    Converts all arguments from polar to Cartesian and then calls draw_line
    """
    start = polar_to_cartesian(*start)
    end = polar_to_cartesian(*end)
    thickness = pixels_to_mm(thickness)
    draw_line(image, start, end, color, thickness)

def draw_sphere(image, position, color, radius):
    """
    Colors all points that are closer to {position} than {radius},
    resulting in a sphere
    """
    plot_boolean_function(image, lambda x, y, z: distance_between_3d_points(position, (x, y, z)) <= radius, color)

def draw_sphere_polar(image, position, color, radius):
    """
    Converts all arguments from polar to Cartesian and then calls draw_sphere
    """
    position = polar_to_cartesian(*position)
    radius = pixels_to_mm(radius)
    draw_sphere(image, position, color, radius)

def draw_hollow_sphere(image, position, color, inner_radius, outer_radius):
    """
    Plot a sphere that is hollow. Save in {image}, middle of sphere is
    {position}, everything between {inner_radius} and {outer_radius} is colored
    """
    plot_boolean_function(image, lambda x, y, z: inner_radius <= distance_between_3d_points(position, (x, y, z)) <= outer_radius, color)

def draw_hollow_sphere_polar(image, position, color, inner_radius, outer_radius):
    inner_radius, outer_radius = (pixels_to_mm(inner_radius), pixels_to_mm(outer_radius))
    position = polar_to_cartesian(*position)
    draw_hollow_sphere(image, position, color, inner_radius, outer_radius)

def plot_color_function(image, function):
    """
    Plot a function (x, y, z) [mm] -> color [0-7]
    """
    for (angle, height, radius), value in np.ndenumerate(image):
        image[angle][height][radius] |= function(*polar_to_cartesian(angle, height, radius))

def plot_boolean_function(image, function, color):
    """
    Plot a function (x, y, z) [mm] -> {true, false}
    """
    for (angle, height, radius), value in np.ndenumerate(image):
        if function(*polar_to_cartesian(angle, height, radius)):
            image[angle][height][radius] |= color

def plot_real_function(image, func, color):
    """
    Plots a function (x, y) [mm] -> z [pixels]
    """
    for angle in range(25):  # Adjusted for the new angle range
        for radius in range(4):  # Adjusted radius range to 4
            x, y, z = polar_to_cartesian(angle, 0, radius)
            z = int(func(x, y))
            if z in range(4):  # Adjusted height range to 4
                image[angle][z][radius] |= color

def draw_string_pattern(image, radius, interval, twist, color, thickness):
    """
    Plots a 'string pattern', where two circles are connected by a number of lines
    (strings) and are twisted with respect to each other
    """
    for i in range(0, 25, interval):  # Adjusted for the new angle range
        draw_line_polar(image, (i, 0, radius), ((i + twist) % 25, 3, radius), color, thickness)  # Adjusted height to 3

def draw_plane(image, plane_params, color, thickness):
    """
    Plot a plane on {image} defined by {a}x + {b}y + {c}z = d
    where (a, b, c, d) = plane_params
    {color} = color of the plane (0-7)
    {thicknes} = thickness [mm]
    """
    (a, b, c, d) = plane_params
    limit = 0.5 * thickness

    for (angle, height, radius), value in np.ndenumerate(image):
        (x, y, z) = polar_to_cartesian(angle, height, radius)
        if abs((a * x) + (b * y) + (c * z) - d) < limit:
            image[angle][height][radius] |= color

def draw_plane_px(image, plane_params, color, thickness):
    (a, b, c, d) = plane_params
    draw_plane(image, (a, b, c, pixels_to_mm(d)), color, pixels_to_mm(thickness))

def connect_points_in_circle(image, point_list, color, thickness):
    """
    Connect each given point to the next one using draw_line, including
    the last one to the first one.
    image: ndarray of 3d image as given by getImage()
    point_list: List of Cartesian points :: [(int, int, int)]
    color/thickness: Same as draw_line
    """
    for (start, end) in zip(point_list, point_list[1:]):
        draw_line(image, start, end, color, thickness)
    draw_line(image, point_list[-1], point_list[0], color, thickness)

def connect_points_in_circle_polar(image, point_list, color, thickness):
    """
    Same as connect_points_in_circle, but with polar coordinates instead of
    Cartesian ones
    """
    for (start, end) in zip(point_list, point_list[1:]):
        draw_line_polar(image, start, end, color, thickness)
    draw_line_polar(image, point_list[-1], point_list[0], color, thickness)

def connect_all_points(image, point_list, color, thickness):
    """
    Connect every given point with every other one.
    Complexity: O(n^2)
    """
    for (start, end) in combinations(point_list, 2):
        draw_line(image, start, end, color, thickness)

def connect_all_points_polar(image, point_list, color, thickness):
    for (start, end) in combinations(point_list, 2):
        draw_line_polar(image, start, end, color, thickness)

def draw_box(image, point, opposite_point, color, thickness):
    """
    Draw a box, aligned to the Cartesian axes, between {point} and
    {opposite_point}.
    """
    (x0, y0, z0) = point
    (x1, y1, z1) = opposite_point

    bottom = [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]
    top    = [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)]

    connect_points_in_circle(image, bottom, color, thickness)
    connect_points_in_circle(image, top, color, thickness)

    for (b, t) in zip(bottom, top):
        draw_line(image, b, t, color, thickness)

def draw_box_polar(image, point, opposite_point, color, thickness):
    point, opposite_point = polar_to_cartesian(*point), polar_to_cartesian(*opposite_point)
    thickness = pixels_to_mm(thickness)
    draw_box(image, point, opposite_point, color, thickness)

### End Drawing Functions ###

### Meta-programming Functions ###

def get_color_string(voxel):
    """
    Get a fixed-width binary string representation
    """
    assert (voxel <= 7), 'Only the last 3 bits may be set'
    return '{0:03b}'.format(voxel)

def display_image(image):
    """
    Print C representation of image
    """
    print(assemble_image(image))

def assemble_image(image):
    """
    Generate C array literal representing the image
    """
    start_string = 'const byte image[25][4][8] = {\n'  # Adjusted dimensions
    end_string = '\n};\n\n'
    return start_string + '\n'.join(convert_slice_to_c(slice, i) for i, slice in enumerate(image)) + end_string

def assemble_program(image):
    """
    Generates image array literal, appends template file to it, returns that
    """
    with open(template_file, 'r') as infile:
        lines = infile.readlines()

    preprocessor = '#include <SPI.h>\n\n'
    image_array_literal = assemble_image(image)
    program = ''.join(lines)

    return preprocessor + image_array_literal + program

def convert_slice_to_c(angle_slice, index):
    """
    Get a slice of the image as a C array literal
    """
    start_string = '\t// Slice {0:02}\n\t{{\n'.format(index)
    end_string = '\n\t}'
    if index != 24:
        end_string += ',\n'
    return start_string + ',\n'.join(convert_line_to_c(line) for line in angle_slice) + end_string

def convert_line_to_c(line):
    """
    Get a line of the image as a C array literal
    """
    # Get a string representation of the binary data for that LED row
    color_string = ''.join([get_color_string(voxel) for voxel in line])
    assert len(color_string) == 24, "Length of color_string is {}, should be 24".format(len(color_string))

    # Print them in chunks of 8 so that they're one byte in C
    byte_literals = ['0b' + i for i in chunk_string(color_string, 8)]
    return '\t\t{' + ', '.join(byte_literals) + '}'

def chunk_string(string, chunk_size):
    """
    Splits a string in chunks of given length
    """
    return [string[i:i+chunk_size] for i in range(0, len(string), chunk_size)]

def save_sketch(image, name):
    """
    Uses all of the above methods to get the Arduino program with the given image,
    and writes that to ./$name/$name.ino
    """
    directory_name = os.path.join(output_directory, name.replace('.ino', ''))

    try:
        os.makedirs(directory_name)
    except OSError:
        print('Directory already exists. Delete it or choose another name.')
        return

    if not name.endswith('.ino'):
        name += '.ino'

    file_path = os.path.join(directory_name, name)
    with open(file_path, 'w') as outfile:
        outfile.write(assemble_program(image))
        print('Wrote sketch to ./' + file_path)

### End Meta-programming Functions ###

def main():
    img = create_empty_image()
    
    # Example: Draw a sphere
    draw_sphere(img, (12, 2, 1), COLOR_RED, 4)  # Adjusted for 25 frames
    
    # Example: Draw a box
    draw_box(img, (5, 1, 0), (20, 3, 2), COLOR_BLUE, 2)  # Adjusted for 25 frames
    
    # Example: Draw a hollow sphere
    draw_hollow_sphere(img, (12, 2, 1), COLOR_GREEN, 2, 4)  # Adjusted for 25 frames
    
    # Example: String pattern
    draw_string_pattern(img, 1, 3, 10, COLOR_YELLOW, 1)
    
    # Optional: Display the image for debugging
    display_image(img)
    
    save_sketch(img, '3D_POV_Display')

if __name__ == '__main__':
    main()
