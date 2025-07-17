import pygame
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from config import N, size, scale, screen_size, dt, visc, inflow_speed
from draw import draw_density

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

# Tracer particle (start near the left edge, center vertically)
tracer_pos = [size[0] // 2, 2.0]  # [row (y), col (x)] in grid coordinates (float)

# Apply simple inflow on left side
def apply_inflow():
    # Make inflow wider and stronger
    density[:, 1:4] = 1.0
    u[:, 1:4] = inflow_speed

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
mouse_button = 1  # 1 for left, 3 for right

while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
            mouse_button = event.button  # 1=left, 3=right
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

    if mouse_down:
        mx, my = pygame.mouse.get_pos()
        grid_x = mx // scale  # column index
        grid_y = my // scale  # row index
        # Clamp indices to valid range
        grid_x = np.clip(grid_x, 2, N-3)
        grid_y = np.clip(grid_y, 2, N-3)
        obstacles[grid_y-2:grid_y+3, grid_x-2:grid_x+3] = 1

        # Add positive or negative current
        if mouse_button == 1:  # Left click: positive
            u[grid_y-2:grid_y+3, grid_x-2:grid_x+3] = inflow_speed
        elif mouse_button == 3:  # Right click: negative
            u[grid_y-2:grid_y+3, grid_x-2:grid_x+3] = -inflow_speed

    apply_inflow()

    density = advect(density, u, v)
    u = advect(u, u, v)
    v = advect(v, u, v)

    # Update tracer position using bilinear interpolation of u, v
    y, x = tracer_pos
    if 1 <= y < size[0]-2 and 1 <= x < size[1]-2:
        # Bilinear interpolation for smooth movement
        uy = np.interp([y], np.arange(size[0]), v[:, int(x)])[0]
        ux = np.interp([x], np.arange(size[1]), u[int(y), :])[0]
        tracer_pos[0] += uy * dt
        tracer_pos[1] += ux * dt

    # Clamp density to [0, 1]
    density = np.clip(density, 0, 1)

    u = diffuse(u, visc)
    v = diffuse(v, visc)

    u, v = project(u, v)
    apply_obstacles(u, v)

    draw_density(screen, density, obstacles, scale, tracer_pos)
    pygame.display.flip()

pygame.quit()
