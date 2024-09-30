import pygame
from numba import njit
import numpy as np

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FOV_VERTICAL = np.pi / 4
FOV_HORIZONTAL = FOV_VERTICAL * SCREEN_WIDTH / SCREEN_HEIGHT
MOUSE_SENSITIVITY = 50  # default: 10
EARTH_MOON_DISTANCE = 61.25
IGNORE = 9999
MOVEMENT_SPEED = 25  # default: 1


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    running = True
    clock = pygame.time.Clock()
    surface = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.mouse.set_visible(False)
    pygame.display.set_caption('NASA App Development Challenge - Surface Simulation')

    points, triangles = read_obj('../assets/surface/surface0.obj')

    z_order = np.zeros(len(triangles))
    shade = np.zeros(len(triangles))

    camera = np.asarray([-74.5, 15.6, 20.9, 1, 0])
    camera_meanings = ['X', 'Y', 'Z', 'H ANGLE', 'V ANGLE']

    while running:
        pygame.mouse.set_pos(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        surface.fill([50, 127, 200])
        elapsed_time = clock.tick() * 0.001
        light_direction = np.asarray([np.sin(pygame.time.get_ticks() * 0.001), 1, 1])
        light_direction /= np.linalg.norm(light_direction)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        project_points(points, camera)
        sort_triangles(points, triangles, camera, z_order, light_direction, shade)

        for index in np.argsort(z_order):
            if z_order[index] == IGNORE:
                break
            triangle = [points[triangles[index][0]][3:], points[triangles[index][1]][3:], points[triangles[index][2]][3:]]
            color = shade[index] * np.abs(points[triangles[index][0]][:3]) + 25  # make * 45 + 25 if object is smaller
            pygame.draw.polygon(surface, color, triangle)

        screen.blit(surface, (0, 0))

        render_text(screen, (10, 10), str(round(1/(elapsed_time + 1e-16))), 30, (255, 255, 255))
        camera_information = [str(round(camera[i], 3)) for i in range(5)]
        for i in range(5):
            render_text(screen, (10, 50 + 15 * i), f'{camera_meanings[i]}: {camera_information[i]}', 15, (255, 255, 255))

        pygame.display.update()
        movement(camera, elapsed_time)

    pygame.quit()


def movement(camera, elapsed_time):
    if pygame.mouse.get_focused():
        p_mouse = pygame.mouse.get_pos()
        camera[3] = (camera[3] + MOUSE_SENSITIVITY * elapsed_time * np.clip((p_mouse[0] - SCREEN_WIDTH / 2) / SCREEN_WIDTH, -0.2, 0.2)) % (2 * np.pi)
        camera[4] += MOUSE_SENSITIVITY * elapsed_time * np.clip((p_mouse[1] - SCREEN_HEIGHT / 2) / SCREEN_HEIGHT, -0.2, 0.2)
        camera[4] = np.clip(camera[4], -1.57, 1.57)

    pressed_keys = pygame.key.get_pressed()

    if pressed_keys[ord('e')]:
        camera[1] += elapsed_time * MOVEMENT_SPEED
    elif pressed_keys[ord('q')]:
        camera[1] -= elapsed_time * MOVEMENT_SPEED

    if (pressed_keys[ord('w')] or pressed_keys[ord('s')]) and (pressed_keys[ord('a')] or pressed_keys[ord('d')]):
        elapsed_time *= 0.707

    if pressed_keys[pygame.K_UP] or pressed_keys[ord('w')]:
        camera[0] += elapsed_time * np.cos(camera[3]) * MOVEMENT_SPEED
        camera[2] += elapsed_time * np.sin(camera[3]) * MOVEMENT_SPEED
    elif pressed_keys[pygame.K_DOWN] or pressed_keys[ord('s')]:
        camera[0] -= elapsed_time * np.cos(camera[3]) * MOVEMENT_SPEED
        camera[2] -= elapsed_time * np.sin(camera[3]) * MOVEMENT_SPEED

    if pressed_keys[pygame.K_LEFT] or pressed_keys[ord('a')]:
        camera[0] += elapsed_time * np.sin(camera[3]) * MOVEMENT_SPEED
        camera[2] -= elapsed_time * np.cos(camera[3]) * MOVEMENT_SPEED
    elif pressed_keys[pygame.K_RIGHT] or pressed_keys[ord('d')]:
        camera[0] -= elapsed_time * np.sin(camera[3]) * MOVEMENT_SPEED
        camera[2] += elapsed_time * np.cos(camera[3]) * MOVEMENT_SPEED


@njit()
def project_points(points, camera):
    for point in points:
        h_angle_camera_point = np.arctan((point[2] - camera[2]) / (point[0] - camera[0] + 1e-16))
        if abs(camera[0] + np.cos(h_angle_camera_point) - point[0]) > abs(camera[0] - point[0]):
            h_angle_camera_point = (h_angle_camera_point - np.pi) % (2 * np.pi)
        h_angle = (h_angle_camera_point - camera[3]) % (2 * np.pi)
        if h_angle > np.pi:
            h_angle -= 2 * np.pi
        point[3] = SCREEN_WIDTH * h_angle / FOV_HORIZONTAL + SCREEN_WIDTH / 2
        distance = np.sqrt((point[0] - camera[0]) ** 2 + (point[1] - camera[1]) ** 2 + (point[2] - camera[2]) ** 2)

        v_angle_camera_point = np.arcsin((camera[1] - point[1]) / distance)
        v_angle = (v_angle_camera_point - camera[4]) % (2 * np.pi)
        if v_angle > np.pi:
            v_angle -= 2 * np.pi
        point[4] = SCREEN_HEIGHT * v_angle / FOV_VERTICAL + SCREEN_HEIGHT / 2


@njit()
def sort_triangles(points, triangles, camera, z_order, light_direction, shade):
    for i in range(len(triangles)):
        triangle = triangles[i]

        vector1 = points[triangle[1]][:3] - points[triangle[0]][:3]
        vector2 = points[triangle[2]][:3] - points[triangle[0]][:3]

        normal = np.cross(vector1, vector2)
        normal /= np.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)

        camera_ray = points[triangle[0]][:3] - camera[:3]
        distance = np.sqrt(camera_ray[0] ** 2 + camera_ray[1] ** 2 + camera_ray[2] ** 2)
        camera_ray /= distance

        xxs = np.asarray([points[triangle[0]][3:5][0], points[triangle[1]][3:5][0], points[triangle[2]][3:5][0]])
        yys = np.asarray([points[triangle[0]][3:5][1], points[triangle[1]][3:5][1], points[triangle[2]][3:5][1]])

        if dot_3d(normal, camera_ray) < 0 and np.min(xxs) > -SCREEN_WIDTH and np.max(xxs) < 2 * SCREEN_WIDTH\
                                          and np.min(yys) > -SCREEN_HEIGHT and np.max(yys) < 2 * SCREEN_HEIGHT:
            z_order[i] = -distance
            shade[i] = 0.5 * dot_3d(light_direction, normal) + 0.5
        else:
            z_order[i] = IGNORE


@njit()
def dot_3d(arr1, arr2):
    return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2]


def render_text(screen, location, text, size, color):
    font = pygame.font.Font('../assets/system/font.ttf', size)
    screen.blit(font.render(text, True, color), location)


def read_obj(filename):
    vertices = []
    triangles = []
    with open(filename) as file:
        for line in file:
            index1 = line.find(' ') + 1
            index2 = line.find(' ', index1 + 1)
            index3 = line.find(' ', index2 + 1)
            if line[:2] == 'v ':
                vertices.append([float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]), 1, 1])
            elif line[0] == 'f':
                triangles.append([int(line[index1:index2]) - 1, int(line[index2:index3]) - 1, int(line[index3:-1]) - 1])
    return np.asarray(vertices), np.asarray(triangles)


if __name__ == '__main__':
    main()
