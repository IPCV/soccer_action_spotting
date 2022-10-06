import numpy as np


REPRESENTATION_WIDTH = 64
REPRESENTATION_HEIGHT = 32
REPRESENTATION_CHANNEL = 3

DIM_TERRAIN = (68, 105, REPRESENTATION_CHANNEL)
DIM_IMAGE = (REPRESENTATION_HEIGHT, REPRESENTATION_WIDTH, REPRESENTATION_CHANNEL)


def unproject_image_point(homography, point2D):
    pitchpoint = homography @ point2D
    return pitchpoint / pitchpoint[2]


def load_homography(homography):
    homography = np.reshape(homography, (3, 3))
    homography = homography / homography[2, 2]
    return np.linalg.inv(homography)


def meter2radar(point2D, dim_terrain, dim_image):
    return np.array([dim_image[1] * ((0.95 * point2D[0] / dim_terrain[1]) + 0.5 + 0.025),
                     dim_image[0] * ((0.95 * point2D[1] / dim_terrain[0]) + 0.5 + 0.025)])


def calculate_player_position(bbox, homography, use_calibration=True):
    if use_calibration:
        projection_point = np.array([int((bbox[0] + bbox[2]) / 2), bbox[3], 1])
        return unproject_image_point(homography, projection_point)[:2]
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[3]) / 2])


def calculate_radar_position(bbox, homography):
    projected_point = calculate_player_position(bbox, homography)
    return meter2radar(projected_point, DIM_TERRAIN, DIM_IMAGE)
