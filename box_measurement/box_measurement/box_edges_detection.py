import argparse
import math
import random
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.optimize import minimize

from typing import Tuple


class Line:
    def __init__(self, a: float, b: float, c: float, inlier_points: list = []):

        self.a = a
        self.b = b
        self.c = c
        self.inlier_points = inlier_points

    def __str__(self):
        return f"a:{self.a} b:{self.b} c:{self.b}"

    def __repr__(self):
        return f"a:{self.a} b:{self.b} c:{self.b}"

    def get_line_explicit(self):
        return self.m, self.c

    def get_inlier_count(self):
        return len(self.inlier_points)

    def set_line(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


def sequential_ransac_multi_line_detection(
    data: npt.NDArray[np.float64],
    threshold: float,
    min_points: int,
    max_iterations: int,
    max_lines: int,
    min_inliers: int = 5,
    min_points_cnt: int = 5,
    visualize: bool = False,
    subwindow: int = 1,
) -> npt.NDArray[Line]:

    best_lines = []
    remaining_data = data

    plt.subplot(1, 2, subwindow)
    for i in range(max_lines):

        best_line = ransac_line_detection(
            data=remaining_data,
            threshold=threshold,
            min_points=min_points,
            max_iterations=max_iterations,
        )

        # first stopping condition
        if best_line.get_inlier_count() <= min_inliers:
            break

        # perform PCA to inliers
        mean = np.mean(best_line.inlier_points, axis=0)
        points_centered = best_line.inlier_points - mean

        pca = PCA(n_components=2)
        pca.fit(best_line.inlier_points)

        v = pca.components_[0]

        a = v[1]
        b = -v[0]
        c = -a * mean[0] - b * mean[1]

        best_line.set_line(a, b, c)

        # accumulate the fitted line
        best_lines.append(best_line)

        # remove the inliers
        dtype = np.dtype(
            (np.void, (remaining_data.shape[1] * remaining_data.dtype.itemsize))
        )
        mask = np.in1d(remaining_data.view(dtype), best_line.inlier_points.view(dtype))
        remaining_data = remaining_data[~mask]

        # visualization
        if visualize:
            X = np.array(best_line.inlier_points)[:, 0]
            Y = np.array(best_line.inlier_points)[:, 1]

            Y_hat = (-a * X - c) / b

            # median_point = geometric_median(best_line.inlier_points)
            # plt.scatter(median_point[0], median_point[1], color="b")

            plt.scatter(X, Y)
            plt.plot(X, Y_hat, color="r")

        # second stopping condition
        if len(remaining_data) <= min_points_cnt:
            break

    return np.array(best_lines)


def ransac_line_detection(
    data: npt.NDArray[np.float64],
    threshold: float,
    min_points: int,
    max_iterations: int,
) -> Line:

    best_num_inliers = None
    for i in range(max_iterations):
        # randomly select a subset of data points
        sample = data[np.random.choice(data.shape[0], 2, replace=False), :]

        # fit a line to the subset of data points
        x1, y1 = sample[0]
        x2, y2 = sample[1]

        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1

        distances = np.abs(a * data[:, 0] + b * data[:, 1] + c) / np.sqrt(
            a**2 + b**2
        )

        # count the number of inliers (data points that are within the threshold distance of the line)
        curr_num_inliers = (distances < threshold).sum()

        # If this line has more inliers than any previous line, update the best fit
        if best_num_inliers is None or curr_num_inliers > best_num_inliers:
            inlier_distances = np.abs(a * data[:, 0] + b * data[:, 1] + c) / np.sqrt(
                a**2 + b**2
            )
            inliers = data[inlier_distances < threshold]

            best_num_inliers = curr_num_inliers
            best_line_model = Line(a, b, c, inliers)

    return best_line_model


def calc_dist_to_line_implicit(line, point):

    a = line.a
    b = line.b
    c = line.c
    x0, y0 = point

    distance = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)
    return distance


def calc_dist_to_point(point1: Tuple[float, float], point2: Tuple[float, float]):

    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def polar_to_cartesian(
    polar_points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    cartesian_points = []
    for point in polar_points:
        degrees = point[0]
        radius = point[1]

        radians = math.radians(degrees)
        x_cartesian = radius * math.cos(radians)
        y_cartesian = radius * math.sin(radians)

        cartesian_points.append((x_cartesian, y_cartesian))

    return np.array(cartesian_points)


def load_points_file(filename):

    points = []
    with open(filename, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            x, y, _ = map(float, line.split())
            points.append((x, y))

    return points


def find_intersection(line1, line2):

    det = line1.a * line2.b - line2.a * line1.b

    # Check if the lines are parallel
    if np.abs(det) < 1e-6:
        return None

    x_int = (line1.b * line2.c - line2.b * line1.c) / det
    y_int = (line2.a * line1.c - line1.a * line2.c) / det

    return (x_int, y_int)


def find_connected_line_pair(detected_lines: npt.NDArray[Line]) -> Tuple[Line, Line]:

    best_inside_circle_cnt = 0
    for i in range(len(detected_lines)):
        for j in range(i + 1, len(detected_lines)):

            line1 = detected_lines[i]
            line2 = detected_lines[j]

            radius = 50
            center = find_intersection(line1, line2)

            inside_circle_cnt = 0

            for inlier_point in line1.inlier_points:
                if calc_dist_to_point(inlier_point, center) < radius:
                    inside_circle_cnt += 1

            for inlier_point in line2.inlier_points:
                if calc_dist_to_point(inlier_point, center) < radius:
                    inside_circle_cnt += 1

            if (
                inside_circle_cnt > best_inside_circle_cnt
                and angle_between_lines(line1, line2) > 85
            ):
                best_intersection = (detected_lines[i], detected_lines[j])
                best_inside_circle_cnt = inside_circle_cnt

    return best_intersection


def visualize_lines(
    cartesian_points: npt.NDArray[np.float64], best_intersection: Tuple[Line, ...]
):

    x_coords = [p[0] for p in cartesian_points]
    y_coords = [p[1] for p in cartesian_points]

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(wspace=0)

    ax = fig.add_subplot()

    ax.scatter(x_coords, y_coords)
    ax.set_xlim(-2000, 1000)
    ax.set_ylim(-2000, 1000)

    # draw the lines
    color_choice = ["r", "b"]
    cnt = 0
    for line in best_intersection:

        curr_color = color_choice[cnt]
        x = np.array([-5000, 5000])
        # y = line.m * x + line.c
        y = (-line.a * x - line.c) / line.b

        ax.plot(x, y, color=curr_color)
        print(f"a:{line.a} b:{line.b} c:{line.c} color:{curr_color}")
        cnt+=1

def visualize_points_polar(
    points_polar: npt.NDArray[np.float64], lidar: str, SCAN_RANGE: dict
):

    thetas = [p[0] for p in points_polar]
    rhos = [p[1] for p in points_polar]

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(wspace=0)
    ax = fig.add_subplot(polar=True)

    if lidar == "horizontal":
        ax.set_xlim(
            np.radians(SCAN_RANGE["ANGLE_H"][0]), np.radians(SCAN_RANGE["ANGLE_H"][1])
        )
    elif lidar == "vertical":
        ax.set_xlim(
            np.radians(SCAN_RANGE["ANGLE_V"][0]), np.radians(SCAN_RANGE["ANGLE_V"][1])
        )

    ax.set_ylim(0, SCAN_RANGE["DIST_H"])
    ax.scatter(np.radians(thetas), rhos)


def roundup(x):
    return int(x) if x % 100 == 0 else int(x + 100 - x % 100)


def geometric_median(points):
    def distance_to_candidate(candidate):
        return np.sum(np.sqrt(np.sum((points - candidate) ** 2, axis=1)))

    candidate = np.mean(points, axis=0)
    result = minimize(distance_to_candidate, candidate, method="L-BFGS-B")

    return result.x


def angle_between_lines(line1, line2):

    L1 = [line1.a, line1.b, line1.c]
    L2 = [line2.a, line2.b, line2.c]

    vec1 = L1[:2]
    vec2 = L2[:2]

    dot_product = np.dot(vec1, vec2)

    mag_vec1 = np.linalg.norm(vec1)
    mag_vec2 = np.linalg.norm(vec2)

    # calculate the cosine of the angle between the vectors
    cos_angle = dot_product / (mag_vec1 * mag_vec2)

    # use arccosine to find the angle in radians
    angle_rad = np.arccos(cos_angle)

    # convert the angle to degrees
    angle_deg = math.degrees(
        angle_rad
        if angle_rad >= 0 and angle_rad <= math.pi / 2
        else math.pi - angle_rad
    )

    return angle_deg


def calc_dist_point(v, p):

    c = np.cross(v, p)
    d_norm = np.linalg.norm(c) / np.linalg.norm(v[:2])

    return d_norm


def calc_box_size(line_pair_h, line_pair_v):
    box_thickness = 9.1

    # calibration lines -- currently hardcoded
    line_length_v = Line(0.019756088821827642, -0.9998048294314565, -853.2871131686443)
    line_height_v = Line(-0.9994686266873855, -0.032595463911893476, -1468.209467629716)

    line_width_h = Line(0.7671607356819438, 0.641454913168446, 1012.9073459819676)
    line_length_h = Line(-0.6454222502939428, 0.7638259741757301, -1076.2303851152674)

    line1 = line_pair_h[0]
    line2 = line_pair_h[1]

    median_point1 = geometric_median(line1.inlier_points)
    median_point2 = geometric_median(line2.inlier_points)

    if angle_between_lines(line1, line_width_h) < angle_between_lines(
        line1, line_length_h
    ):
        width = calc_dist_to_line_implicit(line_width_h, median_point1) + box_thickness
        length = (
            calc_dist_to_line_implicit(line_length_h, median_point2) + box_thickness
        )
    else:
        width = calc_dist_to_line_implicit(line_width_h, median_point2) + box_thickness
        length = (
            calc_dist_to_line_implicit(line_length_h, median_point1) + box_thickness
        )

    line3 = line_pair_v[0]
    line4 = line_pair_v[1]

    median_point3 = geometric_median(line3.inlier_points)
    median_point4 = geometric_median(line4.inlier_points)

    if angle_between_lines(line3, line_length_v) < angle_between_lines(
        line4, line_length_v
    ):
        height = (
            calc_dist_to_line_implicit(line_length_v, median_point3) + box_thickness
        )
    else:
        height = (
            calc_dist_to_line_implicit(line_length_v, median_point4) + box_thickness
        )
    return width, length, height


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--hfile", default="2h.txt")
    parser.add_argument("--vfile", default="2v.txt")
    parser.add_argument("--threshold", default=3, type=float)
    parser.add_argument("--iter", default=1000, type=int)
    args = parser.parse_args()

    SCAN_RANGE = {
        "ANGLE_V": (130, 160),
        "ANGLE_H": (170, 210),
        "DIST_V": 2000,
        "DIST_H": 2000,
    }

    points_h = np.array(load_points_file(args.hfile))
    points_v = np.array(load_points_file(args.vfile))

    # points_calib_h = np.array(load_points_file("1h.txt"))
    # points_calib_v = np.array(load_points_file("1h.txt"))

    mask_h = (
        (points_h[:, 0] > SCAN_RANGE["ANGLE_H"][0])
        & (points_h[:, 0] < SCAN_RANGE["ANGLE_H"][1])
        & (np.absolute(points_h[:, 1]) < SCAN_RANGE["DIST_H"])
    )
    mask_v = (
        (points_v[:, 0] > SCAN_RANGE["ANGLE_V"][0])
        & (points_v[:, 0] < SCAN_RANGE["ANGLE_V"][1])
        & (np.absolute(points_v[:, 1]) < SCAN_RANGE["DIST_V"])
    )

    points_filt_h = points_h[mask_h]
    points_filt_v = points_v[mask_v]

    points_cartesian_h = polar_to_cartesian(points_filt_h)
    points_cartesian_v = polar_to_cartesian(points_filt_v)

    # four data points are required to fit two lines
    if points_filt_h.size < 8:
        print(f"The number of input points are too small: {points_filtered.size}")
        return

    detected_lines_h = sequential_ransac_multi_line_detection(
        points_cartesian_h,
        threshold=args.threshold,
        min_points=2,
        max_iterations=args.iter,
        max_lines=3,
        visualize=True,
        subwindow=1,
    )

    detected_lines_v = sequential_ransac_multi_line_detection(
        points_cartesian_v,
        threshold=args.threshold,
        min_points=2,
        max_iterations=args.iter,
        max_lines=3,
        visualize=True,
        subwindow=2,
    )

    # find the line pair denoting the two edges of the box
    line_pair_h = find_connected_line_pair(detected_lines_h)
    line_pair_v = find_connected_line_pair(detected_lines_v)

    # for line in line_pair_v:
    #     median_point = geometric_median(line.inlier_points)
    #     plt.scatter(median_point[0], median_point[1])

    l, w, h = calc_box_size(line_pair_h, line_pair_v)
    print(f"w: {w/10} cm, l: {l/10} cm, h: {h/10} cm")

    # visualization
    # visualize_lines(cartesian_points, detected_lines)
    # visualize_lines(cartesian_points, best_line_pair)
    # visualize_points_polar(points_filtered, args.lidar, SCAN_RANGE)

    plt.show(block=True)


if __name__ == "__main__":
    main()
