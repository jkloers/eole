import pygame
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from config import N, size, scale, screen_size, dt, visc

# Pygame setup
pygame.init()
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

# Fluid fields
density = np.zeros(size)
u = np.zeros(size)
v = np.zeros(size)

# Obstacle mask (1 = obstacle, 0 = free)
obstacles = np.zeros(size, dtype=np.uint8)

# Draw density field to screen
def draw_density():
    image = (density * 255).clip(0, 255).astype(np.uint8)
    image = np.stack([image]*3, axis=-1)  # RGB
    # Draw obstacles in red
    image[obstacles == 1] = [255, 0, 0]
    image = np.kron(image, np.ones((scale, scale, 1))).astype(np.uint8)
    pygame.surfarray.blit_array(screen, image)

# Apply simple inflow on left side
def apply_inflow():
    # Make inflow wider and stronger
    density[:, 1:4] = 1.0
    u[:, 1:4] = 2.0

# Semi-Lagrangian advection using scipy's map_coordinates
def advect(field, u, v):
    ny, nx = field.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    x_back = (x - dt * u).clip(0, nx - 1)
    y_back = (y - dt * v).clip(0, ny - 1)
    coords = np.array([y_back.ravel(), x_back.ravel()])
    advected = map_coordinates(field, coords, order=1, mode='reflect').reshape(field.shape)
    advected[obstacles == 1] = 0
    return advected

# Basic diffusion using Gaussian blur
def diffuse(field, amount):
    return gaussian_filter(field, sigma=np.sqrt(amount/dt))

# Project step to make the velocity field divergence-free
def project(u, v):
    div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
    p = np.zeros_like(div)
    for _ in range(10):
        p = (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1) - div) / 4
    u -= np.gradient(p, axis=1)
    v -= np.gradient(p, axis=0)
    return u, v

# Obstacle velocity cancellation
def apply_obstacles(u, v):
    u[obstacles == 1] = 0
    v[obstacles == 1] = 0

running = True
mouse_down = False

while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

    if mouse_down:
        mx, my = pygame.mouse.get_pos()
        i, j = mx // scale, my // scale
        # Clamp indices to valid range
        i = np.clip(i, 2, N-3)
        j = np.clip(j, 2, N-3)
        obstacles[j-2:j+3, i-2:i+3] = 1

    apply_inflow()

    density = advect(density, u, v)
    u = advect(u, u, v)
    v = advect(v, u, v)

    # Clamp density to [0, 1]
    density = np.clip(density, 0, 1)

    u = diffuse(u, visc)
    v = diffuse(v, visc)

    u, v = project(u, v)
    apply_obstacles(u, v)

    draw_density()
    pygame.display.flip()

pygame.quit()
