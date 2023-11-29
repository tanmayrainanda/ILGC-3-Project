import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import cv2

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)  # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c)  # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ  # set refresh rate
mlx_shape = (24, 32)

# Create a window for displaying the thermal image
cv2.namedWindow('Thermal Image', cv2.WINDOW_NORMAL)

frame = np.zeros((24 * 32,))  # setup array for storing all 768 temperatures
t_array = []

while True:
    t1 = time.monotonic()
    try:
        mlx.getFrame(frame)  # read MLX temperatures into frame var
        data_array = np.reshape(frame, mlx_shape)  # reshape to 24x32
        data_array = cv2.flip(data_array, 1)  # flip left to right

        # Scale the data to 8-bit for display
        scaled_data = cv2.normalize(data_array, None, 0, 255, cv2.NORM_MINMAX)
        scaled_data = np.uint8(scaled_data)

        # Apply a colormap (in this example, 'plasma')
        thermal_image_colored = cv2.applyColorMap(scaled_data, cv2.COLORMAP_PLASMA)

        cv2.imshow('Thermal Image', thermal_image_colored)
        cv2.waitKey(1)  # Required for imshow to work

        t_array.append(time.monotonic() - t1)
        print('Sample Rate: {0:2.1f}fps'.format(len(t_array) / np.sum(t_array)))
    except ValueError:
        continue  # if error, just read again
