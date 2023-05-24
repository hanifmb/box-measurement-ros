import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from example_interfaces.srv import SetBool
from box_measurement_srv.srv import BoxDimensions
import box_measurement.box_edges_detection as ed
import matplotlib.pyplot as plt

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.subscriber1 = self.create_subscription(
            LaserScan,
            'horizontal/scan',
            self.callback1,
            10  # QoS profile depth
        )

        self.subscriber2 = self.create_subscription(
            LaserScan,
            'vertical/scan',
            self.callback2,
            10  # QoS profile depth
        )

        self.service2 = self.create_service(
            BoxDimensions,
            'box_measurement/get_box_size',
            self.get_box_size
        )

        self.service1 = self.create_service(
            SetBool,
            'box_measurement/calibrate',
            self.calibrate
        )

        self.horizontal_ranges = []
        self.vertical_ranges = []

    def callback1(self, msg):
        self.horizontal_ranges = msg.ranges
        self.angle_min_h = msg.angle_min
        self.angle_max_h = msg.angle_max
        self.angle_increment_h = msg.angle_increment

    def callback2(self, msg):
        self.vertical_ranges = msg.ranges
        self.angle_min_v = msg.angle_min
        self.angle_max_v = msg.angle_max
        self.angle_increment_v = msg.angle_increment

    def calibrate(self, request, response):
        self.get_logger().info('Received service request: %s' % request.data)
        # Do something with the request and generate a response
        response.success = True
        response.message = 'Toggle service executed successfully'
        return response

    def get_combined_angle_range(self):

        start = np.degrees(self.angle_min_h)
        spacing = np.degrees(self.angle_increment_h)
        num_elements = np.array(self.horizontal_ranges).size
        stop = start + (num_elements - 1) * spacing

        angle_h = np.linspace(start, stop, num=num_elements) + 180 # turn to 0-360
        horizontal_ranges = np.array(self.horizontal_ranges) * 1000 # turn to mm
        points_h = np.hstack((angle_h.reshape(-1, 1), horizontal_ranges.reshape(-1, 1)))

        start = np.degrees(self.angle_min_v)
        spacing = np.degrees(self.angle_increment_v)
        num_elements = np.array(self.vertical_ranges).size
        stop = start + (num_elements - 1) * spacing

        angle_v = np.linspace(start, stop, num=num_elements) + 180 # turn to 0-360
        vertical_ranges = np.array(self.vertical_ranges) * 1000 # turn to mm
        points_v = np.hstack((angle_v.reshape(-1, 1), vertical_ranges.reshape(-1, 1)))

        return points_h, points_v

    def get_box_size(self, request, response):
        self.get_logger().info('Received service request')

        points_h, points_v = self.get_combined_angle_range()

        # print("DATA: ", np.degrees(self.angle_min_h), np.degrees(self.angle_max_h), np.degrees(self.angle_increment_h))
        if self.horizontal_ranges and self.vertical_ranges:	

            SCAN_RANGE = {
                "ANGLE_V": (180, 225),
                "ANGLE_H": (160, 190),
                "DIST_V": 2000,
                "DIST_H": 2000,
            }

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

            points_cartesian_h = ed.polar_to_cartesian(points_filt_h)
            points_cartesian_v = ed.polar_to_cartesian(points_filt_v)

            if points_filt_h.size < 8:
                print(f"The number of input points are too small: {points_filt_h.size}")
                return

            detected_lines_h = ed.sequential_ransac_multi_line_detection(
                points_cartesian_h,
                threshold=5,
                min_points=2,
                max_iterations=1000,
                max_lines=3,
                # visualize=True,
                # subwindow=1,
            )

            detected_lines_v = ed.sequential_ransac_multi_line_detection(
                points_cartesian_v,
                threshold=5,
                min_points=2,
                max_iterations=1000,
                max_lines=3,
                # visualize=True,
                # subwindow=2,
            )

            ed.visualize_points_polar(points_v, "vertical", SCAN_RANGE)

            # find the line pair denoting the two edges of the box
            line_pair_h = ed.find_connected_line_pair(detected_lines_h)
            line_pair_v = ed.find_connected_line_pair(detected_lines_v)
            
            l, w, h = ed.calc_box_size(line_pair_h, line_pair_v)
            print(f"w: {w/10} cm, l: {l/10} cm, h: {h/10} cm")

            # ed.visualize_lines(points_cartesian_v, line_pair_v)
            # plt.show(block=True)

            response.width = w
            response.height = h
            response.length = l

        # automatically return [width:0.0, height:0.0, length:0.0] if range data empty
        return response

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
