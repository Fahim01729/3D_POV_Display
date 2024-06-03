import numpy as np
import math
from random import randint
import os

# Directory to store the output sketches
output_dir = 'output_sketches2'
template_path = '3D_POV_Display.cpp.in'  # Ensure this path is correct

# Declare color values
black  = 0b000
blue   = 0b001
green  = 0b010
teal   = 0b011
red    = 0b100
pink   = 0b101
yellow = 0b110
white  = 0b111

### start basics ###

def newImage():
    """
    Return an empty numpy 3-D Array with the dimensions of the image
    """
    return np.zeros((25, 4, 8), dtype=int)  # Adjusted radius range to 4 (0 to 3)

def randPoint():
    """
    Generate a random point in the defined space
    """
    return (randint(0, 24), randint(0, 3), randint(0, 3))  # Adjusted radius range to 3

def px_to_mm(px, inner_radius_mm=20, gap_mm=12):
    """
    Convert pixels to mm based on a specified pixel density for the given radii.
    
    Parameters:
    px (float): The number of pixels to convert.
    inner_radius_mm (float): The inner radius in mm. Default is 20 mm (2.0 cm).
    gap_mm (float): The gap between each pixel in mm. Default is 12 mm (1.2 cm).
    
    Returns:
    float: The converted distance in mm.
    """
    return inner_radius_mm + px * gap_mm

def cartesian(angle, height, radius, angles=25, inner_radius_mm=20, gap_mm=12):
    """
    Convert polar coordinates to cartesian ones.
    Polar: angle [0-(angles-1)], height [0-3], radius [0-3]
    Cartesian: x [mm], y [mm], z [mm]
    """
    # Convert the radius to mm
    radius_mm = px_to_mm(radius, inner_radius_mm=inner_radius_mm, gap_mm=gap_mm)
    
    # Convert the angle to radians based on the angles range
    angle_rad = math.radians(angle * (360 / angles))
    
    x_mm = math.cos(angle_rad) * radius_mm
    y_mm = math.sin(angle_rad) * radius_mm
    
    # Convert height to mm
    height_mm = height * gap_mm  # Each height level is separated by 1.2 cm

    return (x_mm, y_mm, height_mm)

def point_dst_3d(p1, p2):
    """
    Calculate the Euclidean distance between two 3D points.
    
    Parameters:
    p1 (tuple): The first point (x1, y1, z1).
    p2 (tuple): The second point (x2, y2, z2).
    
    Returns:
    float: The Euclidean distance between the points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def point_line_dst(p, v, w):
    """
    Calculate the distance from point p to the line segment vw.
    
    Parameters:
    p (tuple): The point (px, py, pz).
    v (tuple): The start point of the line segment (vx, vy, vz).
    w (tuple): The end point of the line segment (wx, wy, wz).
    
    Returns:
    float: The distance from the point to the line segment.
    """
    if v == w:
        return point_dst_3d(p, v)
    l2 = point_dst_3d(v, w) ** 2
    t = max(0, min(1, np.dot(np.subtract(p, v), np.subtract(w, v)) / l2))
    projection = tuple(v[i] + t * (w[i] - v[i]) for i in range(3))
    return point_dst_3d(p, projection)

### start drawing ###

def drawLine(image, start, end, colour, thickness):
    """
    Or-equals all pixels in {image} that are closer to the line (between {start}
    and {end}) than {thickness} with the {colour} value (int, 0-7).

    All arguments are in cartesian and mm
    """
    plotBoolFunction(image, lambda x, y, z: point_line_dst((x, y, z), start, end) <= thickness, colour)

def drawLinePolar(image, start, end, colour, thickness):
    """
    Converts all arguments from polar to cartesian and then calls drawLine
    """
    start = cartesian(*start)
    end = cartesian(*end)
    thickness = px_to_mm(thickness)
    drawLine(image, start, end, colour, thickness)

def drawSphere(image, position, colour, radius):
    """
    Colours all points that are closer to {position} than {radius},
    resulting in a sphere
    """
    plotBoolFunction(image, lambda x, y, z: point_dst_3d(position, (x, y, z)) <= radius, colour)

def drawSpherePolar(image, position, colour, radius):
    """
    Converts all arguments from polar to cartesian and then calls drawSphere
    """
    position = cartesian(*position)
    radius = px_to_mm(radius)
    drawSphere(image, position, colour, radius)

def hollowSphere(image, position, colour, inner_r, outer_r):
    """
    Plot a sphere that is hollow. Save in {image}, middle of sphere is
    {position}, everything between {inner_r} and {outer_r} is coloured
    """
    plotBoolFunction(image, lambda x, y, z: inner_r <= point_dst_3d(position, (x, y, z)) <= outer_r, colour)

def hollowSpherePolar(image, position, colour, inner_r, outer_r):
    inner_r, outer_r = (px_to_mm(inner_r), px_to_mm(outer_r))
    position = cartesian(*position)
    hollowSphere(image, position, colour, inner_r, outer_r)

def plotColourFunction(image, function):
    """
    Plot a function (x, y, z) [mm] -> colour [0-7]
    """
    for (angle, height, radius), value in np.ndenumerate(image):
        image[angle][height][radius] |= function(*cartesian(angle, height, radius))

def plotBoolFunction(image, function, colour):
    """
    Plot a function (x, y, z) [mm] -> {true, false}
    """
    for (angle, height, radius), value in np.ndenumerate(image):
        if function(*cartesian(angle, height, radius)):
            image[angle][height][radius] |= colour

def realFunction(image, func, colour):
    """
    Plots a function (x, y) [mm] -> z [pixels]
    """
    for angle in range(25):  # Adjusted for the new angle range
        for radius in range(4):  # Adjusted radius range to 4
            x, y, z = cartesian(angle, 0, radius)
            z = int(func(x, y))
            if z in range(4):  # Adjusted height range to 4
                image[angle][z][radius] |= colour

def fadenbild_bruteforce(image, radius, interval, twist, colour, thickness):
    """
    Plots a 'fadenbild', where two circles are connected by a number of lines
    (strings) and are twisted with respect to each other
    """
    for i in range(0, 25, interval):  # Adjusted for the new angle range
        drawLinePolar(image, (i, 0, radius), ((i + twist) % 25, 3, radius), colour, thickness)  # Adjusted height to 3

def drawSurface(image, surface_params, colour, thickness):
    """
    Plot a surface on {image} defined by {a}x + {b}y + {c}z = d
    where (a, b, c, d) = surface_params
    {colour} = color of the surface (0-7)
    {thicknes} = guess what [mm]
    """
    (a, b, c, d) = surface_params
    limit = 0.5 * thickness

    for (angle, height, radius), value in np.ndenumerate(image):
        (x, y, z) = cartesian(angle, height, radius)
        if abs((a * x) + (b * y) + (c * z) - d) < limit:
            image[angle][height][radius] |= colour

def drawSurfacePx(image, surface_params, colour, thickness):
    (a, b, c, d) = surface_params
    drawSurface(image, (a, b, c, px_to_mm(d)), colour, px_to_mm(thickness))

def connectCircle(image, pointList, colour, thickness):
    """
    Connect each given point to the next one using drawLine, including
    the last one to the first one.
    image: ndarray of 3d image as given by getImage()
    pointList: List of cartesian points :: [(int, int, int)]
    colour/thickness: Same as drawLine
    """
    for (start, end) in zip(pointList, pointList[1:]):
        drawLine(image, start, end, colour, thickness)
    drawLine(image, pointList[-1], pointList[0], colour, thickness)

def connectCirclePolar(image, pointList, colour, thickness):
    """
    Same as connectCircle, but with polar coordinates instead of
    cartesian ones
    """
    for (start, end) in zip(pointList, pointList[1:]):
        drawLinePolar(image, start, end, colour, thickness)
    drawLinePolar(image, pointList[-1], pointList[0], colour, thickness)

def connectAll(image, pointList, colour, thickness):
    """
    Connect every given point with every other one.
    SNAIL ALARM: O(n^2)
    """
    for (start, end) in combinations(pointList, 2):
        drawLine(image, start, end, colour, thickness)

def connectAllPolar(image, pointList, colour, thickness):
    for (start, end) in combinations(pointList, 2):
        drawLinePolar(image, start, end, colour, thickness)

def drawCuboid(image, point, oppositePoint, colour, thickness):
    """
    Draw a cuboid, aligned to the cartesian axes, between {point} and
    {oppositePoint}.
    """
    (x0, y0, z0) = point
    (x1, y1, z1) = oppositePoint

    bottom = [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]
    top    = [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)]

    connectCircle(image, bottom, colour, thickness)
    connectCircle(image, top, colour, thickness)

    for (b, t) in zip(bottom, top):
        drawLine(image, b, t, colour, thickness)

def drawCuboidPolar(image, point, oppositePoint, colour, thickness):
    point, oppositePoint = cartesian(*point), cartesian(*oppositePoint)
    thickness = px_to_mm(thickness)
    drawCuboid(image, point, oppositePoint, colour, thickness)

### end drawing ###

### start metaprogramming ###

def getColourString(voxel):
    """
    Get a fixed-width binary string representation
    """
    assert (voxel <= 7), 'Only the last 3 bits may be set'
    return '{0:03b}'.format(voxel)

def printImage(image):
    """
    Print C representation of image
    """
    print(getImage(image))

def getImage(image):
    """
    Generate C array literal representing the image
    """
    startString = 'const byte image[25][4][8] = {\n'  # Adjusted dimensions
    endString = '\n};\n\n'
    return startString + '\n'.join(sliceForC(sl, i) for i, sl in enumerate(image)) + endString

def getProgram(image):
    """
    Generates image array literal, appends template file to it, returns that
    """
    with open(template_path, 'r') as infile:
        lines = infile.readlines()

    preprocessor = '#include <SPI.h>\n\n'
    imageArrayLiteral = getImage(image)
    program = ''.join(lines)

    return preprocessor + imageArrayLiteral + program

def sliceForC(angSlice, index):
    """
    Get a slice of the image as a C array literal
    """
    startString = '\t// Slice {0:02}\n\t{{\n'.format(index)
    endString = '\n\t}'
    if index != 24:
        endString += ',\n'
    return startString + ',\n'.join(lineForC(line) for line in angSlice) + endString

def lineForC(line):
    """
    Get a line of the image as a C array literal
    """
    # Get a string representation of the binary data for that LED row
    colourString = ''.join([getColourString(voxel) for voxel in line])
    assert len(colourString) == 24, "Length of colourString is {}, should be 24".format(len(colourString))

    # Print them in chunks of 8 so that they're one byte in C
    cByteLiterals = ['0b' + i for i in chunks(colourString, 8)]
    return '\t\t{' + ', '.join(cByteLiterals) + '}'

def chunks(string, chunksize):
    """
    Splits a string in chunks of given length
    """
    return [string[i:i+chunksize] for i in range(0, len(string), chunksize)]

def writeSketch(image, name):
    """
    Uses all of the above methods to get the Arduino program with the given image,
    and writes that to ./$name/$name.ino
    """
    dirname = os.path.join(output_dir, name.replace('.ino', ''))

    try:
        os.makedirs(dirname)
    except OSError:
        print('Directory already exists. Delete it or choose another name.')
        return

    if not name.endswith('.ino'):
        name += '.ino'

    filepath = os.path.join(dirname, name)
    with open(filepath, 'w') as outfile:
        outfile.write(getProgram(image))
        print('Wrote sketch to ./' + filepath)

### end metaprogramming ###

def main():
    img = newImage()
    
    # Example: Draw a sphere
    drawSphere(img, (12, 2, 1), red, 4)  # Adjusted for 25 frames
    
    # Example: Draw a cuboid
    drawCuboid(img, (5, 1, 0), (20, 3, 2), blue, 2)  # Adjusted for 25 frames
    
    # Example: Draw a hollow sphere
    hollowSphere(img, (12, 2, 1), green, 2, 4)  # Adjusted for 25 frames
    
    # Example: Fadenbild pattern
    fadenbild_bruteforce(img, 1, 3, 10, yellow, 1)
    
    # Optional: Visualize the image for debugging
    printImage(img)
    
    writeSketch(img, '3D_POV_Display')

if __name__ == '__main__':
    main()
