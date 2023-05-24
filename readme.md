## Usage
Run the node:
```
ros2 run box_measurement box_measurement_node
```
Requesting measurement through the service (see [message definition](box_measurement_interface/src/BoxDimensions.srv)):
```
ros2 service call /box_measurement/get_box_size box_measurement_interface/srv/BoxDimensions "{}"
```
Expected response:
```
response:
box_measurement_interface.srv.BoxDimensions_Response(width=194.8324432373047, length=190.75941467285156, height=256.3702697753906)
```

## Notes
`box_measurement_interface` is a separate package for the custom message generation. `box_measurement_node` will return `width=0.0, length=0.0, height=0.0` in case of no lidar topic advertised, otherwise return measured values in mm.

### todo
* Automatic calibration parameters update.
* Multiple threads for topic and service callbacks.
* Error handing when lines undetected.
