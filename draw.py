import pygame
import numpy as np

def draw_density(screen, density, obstacles, scale, tracer_pos=None):
    image = (density * 255).clip(0, 255).astype(np.uint8)
    image = np.stack([image]*3, axis=-1)  # RGB
    image[obstacles == 1] = [255, 0, 0]  # Draw obstacles in red
    image = np.kron(image, np.ones((scale, scale, 1))).astype(np.uint8)
    if tracer_pos is not None:
        y, x = tracer_pos
        py = int(y * scale)
        px = int(x * scale)
        # Draw a green dot for the tracer
        if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
            image[py-2:py+3, px-2:px+3] = [0, 255, 0]
    pygame.surfarray.blit_array(screen, np.transpose(image, (1, 0, 2)))