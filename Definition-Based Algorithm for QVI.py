import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from shapely.geometry import Point


start_time = time.process_time()
print("start time cpu:", start_time)


def Generate_C():
    triangle_vertices = [(0, 1), (1, 0), (1, 1)]
    Lines_C = [((1, 0), (0, 1)),
               ((1, 0), (1, 1)),
               ((0, 1), (1, 1))]
    # print("Type of Lines_C in Generate_C():", type(Lines_C))
    return Lines_C

def Random_Point_of_C():
    while True:
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        if u + v >= 1:
            print("Random point in C:", [u, v])
            return [u, v]
def Check_points_in_C(x):
    # print("Type x in Check_points_in_C(x):", type(x))
    if isinstance(x, str):
        input_str = x
        if input_str.startswith('[') and input_str.endswith(']'):
            numbers = [float(x.strip()) for x in input_str[1:-1].split(',')]
            if len(numbers) == 2 and all(0 <= num <= 1 for num in numbers) and sum(numbers) >= 1:
                return True
    elif isinstance(x, list):
        numbers = x
        if len(numbers) == 2 and all(0 <= num <= 1 for num in numbers) and sum(numbers) >= 1:
            return True
    else:
        return None


def Generate_Phi_C(x):
    # print("Type x in Generate_Phi_C(x):", type(x))
    if Check_points_in_C(x):
        if isinstance(x, str):
            x = [float(x.strip()) for x in x[1:-1].split(',')]
        # print("Type of x in Generate_Phi_C(x):", type(x))
        print("x_1, x_2,", x[0], x[1])
        k = 1 / (2 ** 6)
        print("k:=", k)
        Square_Vertices_Phi_C = [(k * x[0], k * x[1]), (k * x[0], (k * x[1]) + 1), ((k * x[0]) + 1, k * x[1]),
                                 ((k * x[0]) + 1, (k * x[1]) + 1)]
        print("Square Vertices:", Square_Vertices_Phi_C)

        Lines_Phi_C = [((k * x[0], k * x[1]), (k * x[0], (k * x[1]) + 1)),
                       (((k * x[0]) + 1, k * x[1]), ((k * x[0]) + 1, (k * x[1]) + 1)),
                       ((k * x[0], k * x[1]), (k * x[0] + 1, k * x[1])),
                       ((k * x[0], k * x[1] + 1), ((k * x[0]) + 1, (k * x[1]) + 1))]
        # print("Lines_Phi_C:", Lines_Phi_C)
        return Lines_Phi_C
    else:
        return None

def Random_Point_of_Phi_C(x):
    point = []
    if Check_points_in_C(x):
        if isinstance(x, str):
            x = [float(x.strip()) for x in x[1:-1].split(',')]
        k = 1 / ((2 ** 6) * (math.sqrt(x[0] ** 2 + x[1] ** 2)))
        x_min = k * x[0]
        x_max = (k * x[0]) + 1
        y_min = k * x[1]
        y_max = (k * x[1]) + 1
        point.append(random.uniform(x_min, x_max))
        point.append(random.uniform(y_min, y_max))
        #print("Random point in \Phi(x):", point)
        return point
    else:
        None


def Check_points_in_Phi_C(point1, point2):
    if Check_points_in_C(point1):
        if isinstance(point1, str):
            point1 = [float(x.strip()) for x in point1[1:-1].split(',')]
        if isinstance(point2, str):
            point2 = [float(x.strip()) for x in point2[1:-1].split(',')]
        k = 1 / (2 ** 6)
        if (k * point1[0] <= point2[0] <= (k * point1[0]) + 1) and (k * point1[1] <= point2[1] <= (k * point1[1]) + 1):
            return True


def euclidean_distance(point1, point2):
    if isinstance(point1, (int, float)) or isinstance(point2, (int, float)):
        raise ValueError("Both points must be iterable objects.")
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def closest_point_on_line_segment(point, line_start, line_end):
    p = np.array(point)
    start = np.array(line_start)
    end = np.array(line_end)
    line_vector = end - start
    point_vector = p - start
    line_length_squared = np.dot(line_vector, line_vector)
    if line_length_squared == 0:
        return start.tolist()
    t = np.dot(point_vector, line_vector) / line_length_squared
    t = np.clip(t, 0, 1)
    closest_point = start + t * line_vector
    return closest_point.tolist()


def euclidean_projection_WR_C(point):
    Lines = Generate_C()
    #if not Lines:
        #print("No lines provided for projection.")
        #return None
    closest_point = None
    min_distance = float('inf')
    for line_start, line_end in Lines:
        if Check_points_in_C(point):
            closest_point = point
            break
        closest = closest_point_on_line_segment(point, line_start, line_end)
        distance = euclidean_distance(point, closest)
        if distance < min_distance:
            min_distance = distance
            closest_point = closest
    print("Closest point on the lines to the given point:", closest_point)
    return closest_point


def euclidean_projection_WR_Phi_C(point1, point2):
    #if not Lines:
        #print("No lines provided for projection.")
        #return None
    Lines = Generate_Phi_C(point1)
    closest_point = None
    min_distance = float('inf')
    for line_start, line_end in Lines:
        if Check_points_in_Phi_C(point1, point2):
            closest_point = point2
            break
        closest = closest_point_on_line_segment(point2, line_start, line_end)
        distance = euclidean_distance(point2, closest)
        if distance < min_distance:
            min_distance = distance
            closest_point = closest
    print("Closest point on the lines to the given point:", closest_point)
    return closest_point


def compare_variables(x, point):
    if isinstance(x, str) and isinstance(point, list):
        x = [float(val.strip()) for val in x.strip('[]').split(',')]
    elif isinstance(x, list) and isinstance(point, str):
        point = [float(val.strip()) for val in point.strip('[]').split(',')]
    elif not isinstance(x, (str, list)) or not isinstance(point, (str, list)):
        raise ValueError("Both variables must be either strings or lists.")

    if x == point:
        # print("Variables are equal.")
        return True
    else:
        # print("Variables are not equal.")
        return None


def difference_variables(x, point):
    if isinstance(x, str) and isinstance(point, list):
        x = [float(val.strip()) for val in x.strip('[]').split(',')]
    elif isinstance(x, list) and isinstance(point, str):
        point = [float(val.strip()) for val in point.strip('[]').split(',')]
    elif not isinstance(x, (str, list)) or not isinstance(point, (str, list)):
        raise ValueError("Both variables must be either strings or lists.")

    if all(abs(a - b) <= 10 ** (-8) for a, b in zip(x, point)):
        print("Variables are equal.")
        return True
    else:
        print("Variables are not equal.")
        return None


def Follow_the_algorithm():
    start_time = time.process_time()
    print("start time cpu:", start_time)

    x = input("Enter the number x_0 to check if it is in C:")
    # print("Type of x:", type(x))
    z = input("Enter the number as a z_0 to check if it is in Phi(C):")
    The_Sequence_Of_Equilibrium_Problem_Solution = []
    The_Sequence_Of_Projected_Solution = []
    for i in range(100):
        #if not check_points_in_C(x):
            #print("Please start the algorithm with new starting point x belonging to C.")
            #x = Random_Point_of_C()
        #if not Check_points_in_Phi_C(x, z):
            #print("Please consider Phi_C(x) to find the proper choice as a new z belonging to Phi_C(x).")
            #z = Random_Point_of_Phi_C(x)
        #if Check_points_in_C(x) and Check_points_in_Phi_C(x, z):
        if isinstance(x, str):
            x = [float(u.strip()) for u in x[1:-1].split(',')]
        # Instead of (\gamma * T), where (\gamma = 4), we write T= 4*T, while T = [[0.22, 0], [0, 0.25]]
        T = [[0.88, 0], [0, 1]]
        if isinstance(z, str):
            z = [float(x.strip()) for x in z[1:-1].split(',')]
        u = z - np.dot(T, z)
        print("Type of u = z - np.dot(T, z):", type(u))
        print("u = z - np.dot(T, z):", u)
        if isinstance(u, np.ndarray):
            u = u.tolist()
        print("Type of u_new:", type(u))
        print("u_new:", u)
        #Lines_Phi_C = Generate_Phi_C(x)
        z_new = euclidean_projection_WR_Phi_C(x, u)
        print("z_new:", z_new)
        The_Sequence_Of_Equilibrium_Problem_Solution.append(z_new)
        if difference_variables(z, z_new):
            print("The variables z and z_new are equal.")
            print("The number of the last iteration where the variables z and z_new are equal:",
                  i + 1)
            #Lines_C = Generate_C()
            x_new = euclidean_projection_WR_C(z_new)
            print("The Approximate Projected Solution:", x_new)
            The_Sequence_Of_Projected_Solution.append(x_new)
            if difference_variables(x, x_new):
                print("The variables x and x_new are equal.")
                print("The number of the last iteration where the sequence converges to the projected solution:",
                      i + 1)
                break
            else:
                print("The variables x and x_new are NOT equal.")
                x = x_new
                z = z_new

        else:
            print("The number of iteration we have z_new:", i + 1)
            z = z_new


        if not Check_points_in_Phi_C(x, z):
            print("Please consider Phi_C(x) to find the proper choice as a new z belonging to Phi_C(x).")
            z = Random_Point_of_Phi_C(x)
            print("Random point z in \Phi(x):", z)


    print("The Sequence Of Equilibrium Problem Solution", The_Sequence_Of_Equilibrium_Problem_Solution)
    print("The Sequence Of Projected Solution:", The_Sequence_Of_Projected_Solution)
    end_time = time.process_time()
    cpu_time = end_time - start_time
    print("start time:", start_time, "end time:", end_time)
    print("CPU time:", cpu_time, "seconds")


Follow_the_algorithm()
