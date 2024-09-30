import pygame
from numba import njit
import numpy as np
from time import time

# CONSTANTS
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FOV_VERTICAL = np.pi / 4
FOV_HORIZONTAL = FOV_VERTICAL * SCREEN_WIDTH / SCREEN_HEIGHT
MOUSE_SENSITIVITY = 50  # default: 50
EARTH_MOON_DISTANCE = 61.25
MOVEMENT_SPEED = 1
IGNORE = 9999
m = dict()


def main(title, camera, pre, actions, skybox_file=None):
    """
    Creates, updates, and renders scene.

    Args:
        title (str): The title of the scene.
        camera (list): The initial position and direction of the camera.
        pre (func): Creation of models at beginning.
        actions (func): Updates for models every game tick.
        skybox_file (str): The file path to the skybox image (optional).
    """

    # window initialization
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    running = True
    clock = pygame.time.Clock()
    frame = np.ones((SCREEN_WIDTH, SCREEN_HEIGHT, 3)).astype('uint8')
    z_buffer = np.ones((SCREEN_WIDTH, SCREEN_HEIGHT))
    shadow_map = np.ones((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.mouse.set_visible(False)
    pygame.display.set_caption(f'SILVER - {title} Simulation')

    # clears data from previous runs
    Model.registry.clear()
    m.clear()

    # display loading screen before engine initialization
    loading = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    draw_loading(loading)
    screen.blit(loading, (0, 0))
    pygame.display.update()

    # creation of skybox
    try:
        if skybox_file is not None:
            skybox = pygame.image.load(skybox_file).convert()
            skybox = pygame.transform.smoothscale(skybox, (
                SCREEN_WIDTH * (np.pi * 2 / FOV_HORIZONTAL + 1), SCREEN_HEIGHT * (np.pi / FOV_VERTICAL + 1)))
            skybox = pygame.surfarray.array3d(skybox)
    except FileNotFoundError:
        log('error', 'Skybox file not found. Replaced with default.', [skybox_file])
        skybox_file = None

    # model initialization (EXTERNAL)
    pre()

    # camera and lighting initialization
    camera = np.asarray(camera)
    camera_meanings = ['X', 'Y', 'Z', 'H ANGLE', 'V ANGLE']
    light_camera = np.asarray([-500000.0, 1000000.0, -500000.0, 0.8, 1.0])

    # miscellaneous
    paused = False  # pause settings for simulation
    opacity = 255  # opacity of loading screen for fade effect

    log('log', f'{title} simulation started.')
    start_time = time()
    while running:
        # update camera, mouse, and clock
        pygame.mouse.set_pos(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        elapsed_time = clock.tick() * 0.001

        # update models (EXTERNAL)
        if not paused:
            actions(elapsed_time)

        # update lighting
        light_camera[0] = light_camera[1] * np.sin(pygame.time.get_ticks() / 4500)
        light_camera[2] = light_camera[1] * np.cos(pygame.time.get_ticks() / 5000)

        # pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # quit on window close
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False  # quit on window escape
                if event.key == pygame.K_DELETE:
                    camera = np.asarray(camera)  # reset camera on delete
                if event.key == pygame.K_p:
                    paused = not paused  # pause/resume simulation

        # update skybox (if present)
        h1 = int(SCREEN_WIDTH * camera[3] / FOV_HORIZONTAL)
        v1 = int(SCREEN_HEIGHT * camera[4] / FOV_VERTICAL + 2 * SCREEN_HEIGHT)
        if skybox_file is None:
            frame[:, :, :] = np.asarray([0, 0, 0]).astype('uint8')
        else:
            frame[:, :, :] = skybox[h1:h1 + SCREEN_WIDTH, v1:v1 + SCREEN_HEIGHT, :]

        # update shadows
        z_buffer[:, :] = 1e32
        shadow_map[:, :] = 1e32

        # render all updates
        render_frame(camera, frame, z_buffer, shadow_map, light_camera)

        # blit render to screen
        surface = pygame.surfarray.make_surface(frame)
        surface = pygame.transform.scale(surface, screen.get_size())
        screen.blit(surface, (0, 0))

        # display FPS and camera information
        render_text(screen, (10, 10), str(round(1 / (elapsed_time + 1e-16))), 30, (255, 255, 255))
        camera_information = [str(round(float(camera[i]), 3)) for i in range(5)]
        for i in range(5):
            render_text(screen, (10, 50 + 15 * i), f'{camera_meanings[i]}: {camera_information[i]}', 15,
                        (255, 255, 255))

        # update loading screen if applicable
        if opacity:
            loading.set_alpha(opacity)
            draw_loading(loading)
            screen.blit(loading, (0, 0))
            if elapsed_time < 2:
                opacity -= elapsed_time * 100

        # refresh window
        pygame.display.update()
        if elapsed_time < 2:
            movement(camera, min(elapsed_time * 10, 1))

    pygame.quit()  # quits after game loop ends
    log('log', f'{title} simulation ended after {round(time() - start_time, 1)} seconds.')
    log('break', '')


def read_obj(filename):
    """
    Interprets OBJ to find vertices and triangles and creates UV maps and textures for them.

    Args:
        filename (str): The file path to the OBJ.

    Returns:
        vertices (list): The vertices of the model.
        triangles (list): The triangles of the model.
        uv_map (list): The UV map of the model.
        diffuse_map (list): The diffuse map of the model.
        textured (bool): Whether the model is textured or not.
    """

    vertices, triangles, uv_map, diffuse_map = [], [], [], []
    try:
        with open(filename) as file:
            for line in file.readlines():
                line = line.split()
                if len(line) == 0:
                    continue

                if line[0] == "v":
                    # vertex location in 3D space (x, y, z)
                    vertices.append(line[1:4] + [1, 1, 1])
                elif line[0] == "vt":
                    # vertex location on diffuse map in 2D space (x, y)
                    uv_map.append(line[1:3])
                elif line[0] == "f":  # face
                    if len(line[1].split("/")) == 1:  # if not textured
                        # if the face is made of 4 vertices (quadrilateral), cuts into 2 triangles and only appends 2nd one
                        triangles.append([line[1], line[3], line[4]] if len(line) > 4 else [line[1], line[2], line[3]])
                    else:
                        p1, p2, p3 = line[1].split("/"), line[2].split("/"), line[3].split("/")
                        triangles.append([p1[0], p2[0], p3[0]])  # connects first set of vertices
                        diffuse_map.append([p1[1], p2[1], p3[1]])
                        if len(line) > 4:
                            p4 = line[4].split("/")
                            triangles.append([p1[0], p3[0], p4[0]])  # connects 2nd set of vertices, if applicable
                            diffuse_map.append([p1[1], p3[1], p4[1]])

        # converts vertices and triangles to numpy arrays
        vertices = np.asarray(vertices).astype(float)
        triangles = np.asarray(triangles).astype(int) - 1

        # converts uv_map and diffuse_map to numpy arrays
        textured = len(uv_map) and len(diffuse_map)  # if textured
        if textured:
            uv_map = np.asarray(uv_map).astype(float)
            uv_map[:, 1] = 1 - uv_map[:, 1]
            diffuse_map = np.asarray(diffuse_map).astype(int) - 1
        else:
            uv_map, diffuse_map = np.asarray(uv_map), np.asarray(diffuse_map)
            log('warning', 'Textures not applied to model.', [filename])

    except FileNotFoundError:
        log('error', 'FATAL. Model file not found.', [filename])
        quit()
    except Exception as e:
        log('error', 'FATAL. During the parsing of the file.',
            [filename, 'Error ' + str(e), 'Ensure your OBJ file is correctly formatted.'])
        quit()
    log('log', 'Successfully loaded model.', [filename])

    return vertices, triangles, uv_map, diffuse_map, textured


def movement(camera, elapsed_time):
    """
    Updates the camera's position and direction in the scene.

    Args:
        camera (np.ndarray): The scene's camera object.
        elapsed_time (float): The time since the previous tick.
    """

    # detects movement in mouse and adjusts horizontal and vertical angles accordingly
    if pygame.mouse.get_focused():
        p_mouse = pygame.mouse.get_pos()
        camera[3] = (camera[3] + MOUSE_SENSITIVITY * elapsed_time * np.clip(
            (p_mouse[0] - SCREEN_WIDTH / 2) / SCREEN_WIDTH, -0.2, 0.2)) % (2 * np.pi)
        camera[4] = np.clip(
            camera[4] + MOUSE_SENSITIVITY * elapsed_time * np.clip((p_mouse[1] - SCREEN_HEIGHT / 2) / SCREEN_HEIGHT,
                                                                   -0.2, 0.2), -1.57, 1.57)

    pressed_keys = pygame.key.get_pressed()
    distance = elapsed_time * MOVEMENT_SPEED
    cos_angle, sin_angle = np.cos(camera[3]), np.sin(camera[3])

    # vertical movement
    camera[1] += distance if pressed_keys[ord('e')] else -distance if pressed_keys[ord('q')] else 0

    # diagonal movement adjustment
    if (pressed_keys[ord('w')] or pressed_keys[ord('s')]) and (pressed_keys[ord('a')] or pressed_keys[ord('d')]):
        elapsed_time *= 0.707

    # forward/backward movement
    if pressed_keys[pygame.K_UP] or pressed_keys[ord('w')]:
        camera[0] += distance * cos_angle
        camera[2] += distance * sin_angle
    elif pressed_keys[pygame.K_DOWN] or pressed_keys[ord('s')]:
        camera[0] -= distance * cos_angle
        camera[2] -= distance * sin_angle

    # left/right movement
    if pressed_keys[pygame.K_LEFT] or pressed_keys[ord('a')]:
        camera[0] += distance * sin_angle
        camera[2] -= distance * cos_angle
    elif pressed_keys[pygame.K_RIGHT] or pressed_keys[ord('d')]:
        camera[0] -= distance * sin_angle
        camera[2] += distance * cos_angle


class Model:
    registry = []

    def __init__(self, path_obj, path_texture=''):
        self.registry.append(self)
        self.points_og, self.triangles, self.texture_uv, self.texture_map, self.textured = read_obj(path_obj)
        self.position = np.asarray([0, 0, 0, 0, 0, 0, 1])
        self.points = self.points_og.copy()
        self.shadow_points = self.points.copy()

        if self.textured and path_texture != '':
            self.texture = pygame.surfarray.array3d(pygame.image.load(path_texture))
        else:
            self.textured = False
            self.texture_uv, self.texture_map = np.ones((2, 2)), np.random.randint(1, 2, (2, 3))
            self.texture = np.random.randint(0, 255, (10, 10, 3))

    def change_position(self, x=0, y=0, z=0, rot_x=0, rot_y=0, rot_z=0, scale=1, reset=0):
        self.position = self.position * (1 - reset) + np.asarray([x, y, z, rot_x, rot_y, rot_z, scale])
        self.points = self.points_og.copy()

        if self.position[6] != 1:
            self.points *= scale
        if self.position[3] != 0:
            temp_points = self.points.copy()
            self.points[:, 1] = temp_points[:, 1] * np.cos(self.position[3]) - temp_points[:, 2] * np.sin(
                self.position[3])
            self.points[:, 2] = temp_points[:, 1] * np.sin(self.position[3]) + temp_points[:, 2] * np.cos(
                self.position[3])
        if self.position[4] != 0:
            temp_points = self.points.copy()
            self.points[:, 0] = temp_points[:, 0] * np.cos(self.position[4]) - temp_points[:, 2] * np.sin(
                self.position[4])
            self.points[:, 2] = temp_points[:, 0] * np.sin(self.position[4]) + temp_points[:, 2] * np.cos(
                self.position[4])
        if self.position[5] != 0:
            temp_points = self.points.copy()
            self.points[:, 0] = temp_points[:, 0] * np.cos(self.position[5]) - temp_points[:, 1] * np.sin(
                self.position[5])
            self.points[:, 1] = temp_points[:, 0] * np.sin(self.position[5]) + temp_points[:, 1] * np.cos(
                self.position[5])

        if self.position[0] != 0:
            self.points[:, 0] += self.position[0]
        if self.position[1] != 0:
            self.points[:, 1] += self.position[1]
        if self.position[2] != 0:
            self.points[:, 2] += self.position[2]
        self.shadow_points = self.points.copy()


def render_frame(camera, frame, z_buffer, shadow_map, light_camera):
    """
    Renders the current frame by projecting 3D models onto a 2D frame and updating the shadow map.

    Args:
        camera (np.ndarray): The position and orientation of the camera in the scene.
        frame (np.ndarray): The 2D array representing the screen where the frame is drawn.
        z_buffer (np.ndarray): A buffer to keep track of depth information.
        shadow_map (np.ndarray): A 2D array representing the shadow map.
        light_camera (np.ndarray): The position and orientation of the light source.
    """

    # calculate light vector
    light_vector = np.asarray([
        camera[0] + 30 * np.cos(camera[3]) - light_camera[0],
        -light_camera[1],
        camera[2] + 30 * np.sin(camera[3]) - light_camera[2]
    ])
    length = np.linalg.norm(light_vector)

    # calculate horizontal and vertical light camera angles
    h_vector = light_vector[[0, 2]] / np.linalg.norm(light_vector[[0, 2]])
    light_camera[3] = np.arccos(dot_product(h_vector, np.asarray([1, 0])))
    light_camera[4] = np.arcsin(dot_product(light_vector / length, np.asarray([0, -1, 0])))

    # correct light camera horizontal angle sign if necessary
    if np.sign(np.sin(light_camera[3])) != np.sign(light_vector[2]):
        light_camera[3] *= -1

    for model in Model.registry:
        # project shadow points and render shadow map
        project_points(model.shadow_points, light_camera, shadow_mod=0.01 * length)
        render_shadow_map(model.shadow_points, model.triangles, light_camera, shadow_map)

        # project points and render model
        project_points(model.points, camera)
        draw_model(frame, model.points, model.triangles, camera, light_camera, z_buffer, model.textured,
                   model.texture_uv, model.texture_map, model.texture, model.shadow_points, shadow_map)


@njit()
def render_shadow_map(points, triangles, light_camera, shadow_map):
    """
    Renders the shadow map by projecting 3D points of each triangle onto the 2D screen and updating the shadow map.

    Args:
        points (np.ndarray): An array of 3D points for the object being rendered.
        triangles (np.ndarray): An array of indices that represent the vertices of each triangle.
        light_camera (np.ndarray): The position and orientation of the light source.
        shadow_map (np.ndarray): A 2D array representing the shadow map.
    """

    for index in range(len(triangles)):
        triangle = triangles[index]

        # calculate the vectors and normal for the current triangle
        vector1 = points[triangle[1]][:3] - points[triangle[0]][:3]
        vector2 = points[triangle[2]][:3] - points[triangle[0]][:3]
        normal = np.cross(vector1, vector2)
        normal /= np.linalg.norm(normal)

        # compute camera ray and light shading
        camera_ray = (points[triangle[0]][:3] - light_camera[:3]) / points[triangle[0]][5]
        xxs = [points[triangle[0]][3], points[triangle[1]][3], points[triangle[2]][3]]
        yys = [points[triangle[0]][4], points[triangle[1]][4], points[triangle[2]][4]]
        z_min = min([points[triangle[0]][5], points[triangle[1]][5], points[triangle[2]][5]])

        # filter out triangles that are not visible
        if not filter_triangles(z_min, normal, -camera_ray, xxs, yys):
            continue

        # sort points by their y-coordinate
        proj_points = points[triangle][:, 3:]
        sorted_y = proj_points[:, 1].argsort()
        start, middle, stop = proj_points[sorted_y[0]], proj_points[sorted_y[1]], proj_points[sorted_y[2]]

        # precompute x and z slopes
        x_slopes = get_slopes(start[0], middle[0], stop[0], start[1], middle[1], stop[1])
        z_slopes = get_slopes(start[2], middle[2], stop[2], start[1], middle[1], stop[1])

        # iterate over the y-axis range
        y_start = max(1, int(start[1]))
        y_end = min(SCREEN_HEIGHT - 1, int(stop[1]) + 1)

        for y in range(y_start, y_end):
            delta_y = y - start[1]
            x1 = start[0] + int(delta_y * x_slopes[0])
            z1 = start[2] + delta_y * z_slopes[0]

            # switch between upper and lower segments of the triangle
            if y < middle[1]:
                x2 = start[0] + int(delta_y * x_slopes[1])
                z2 = start[2] + delta_y * z_slopes[1]
            else:
                delta_y = y - middle[1]
                x2 = middle[0] + int(delta_y * x_slopes[2])
                z2 = middle[2] + delta_y * z_slopes[2]

            # ensure x1 <= x2 for proper iteration
            if x1 > x2:
                x1, x2 = x2, x1
                z1, z2 = z2, z1

            # bound x1 and x2 within the screen width and iterate over them
            xx1 = max(1, min(SCREEN_WIDTH - 1, int(x1)))
            xx2 = max(1, min(SCREEN_WIDTH - 1, int(x2 + 1)))

            if xx1 == xx2:
                continue

            z_slope = (z2 - z1) / (x2 - x1 + 1e-32)

            # update the shadow map
            if np.min(shadow_map[xx1:xx2, y]) == 1e32:
                shadow_map[xx1:xx2, y] = (np.arange(xx1, xx2) - x1) * z_slope + z1
            else:
                for x in range(xx1, xx2):
                    z = z1 + (x - x1) * z_slope + 1e-32
                    if z < shadow_map[x, y]:
                        shadow_map[x, y] = z


@njit()
def project_points(points, camera, shadow_mod=1.0):
    """
    Projects 3D points onto a 2D plane using the camera's position and orientation.

    Args:
        points (np.ndarray): An array of 3D points where each point contains its position in space and screen coordinates.
        camera (np.ndarray): The position and orientation of the camera in the scene.
        shadow_mod (float): A modifier for adjusting the field of view when projecting shadows. Defaults to 1.0.
    """

    cos_hor = np.cos(-camera[3] + np.pi / 2)
    sin_hor = np.sin(-camera[3] + np.pi / 2)
    cos_ver = np.cos(-camera[4])
    sin_ver = np.sin(-camera[4])

    # precompute FOV adjustments
    hor_fov_adjust = 0.5 * SCREEN_WIDTH / np.tan(FOV_HORIZONTAL * 0.5 / shadow_mod)
    ver_fov_adjust = 0.5 * SCREEN_HEIGHT / np.tan(FOV_VERTICAL * 0.5 / shadow_mod)

    # translate points to camera space
    points[:, 3:6] = points[:, :3] - camera[:3]

    # creates copy for intermediate transformations
    points2 = points.copy()

    # horizontal and vertical rotations
    points2[:, 3] = points[:, 3] * cos_hor - points[:, 5] * sin_hor
    points2[:, 5] = points[:, 3] * sin_hor + points[:, 5] * cos_hor
    points[:, 4] = points2[:, 4] * cos_ver - points2[:, 5] * sin_ver
    points[:, 5] = points2[:, 4] * sin_ver + points2[:, 5] * cos_ver
    # prevent near-zero division in prospective projection
    points[:, 5][(points[:, 5] < 0.001) & (points[:, 5] > -0.001)] = -0.001

    # perspective projection to screen coordinates
    points[:, 3] = (-hor_fov_adjust * points2[:, 3] / points[:, 5] + 0.5 * SCREEN_WIDTH).astype(np.int32)
    points[:, 4] = (-ver_fov_adjust * points[:, 4] / points[:, 5] + 0.5 * SCREEN_HEIGHT).astype(np.int32)


@njit()
def dot_product(arr1, arr2):
    """
    Calculates the dot product between two vectors of length 2 or 3.

    Args:
        arr1 (np.ndarray): The first vector, either of length 2 or 3.
        arr2 (np.ndarray): The second vector, either of length 2 or 3.

    Returns:
        float: The dot product of the two input vectors. It returns the sum of the products of the respective components.
    """

    if len(arr1) == len(arr2) == 2:
        return arr1[0] * arr2[0] + arr1[1] * arr2[1]
    elif len(arr1) == len(arr2) == 3:
        return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2]


@njit()
def draw_model(frame, points, triangles, camera, light_camera, z_buffer, textured, texture_uv, texture_map, texture,
               shadow_points, shadow_map):
    """
    Draws a 3D model on the frame by projecting its triangles onto the screen and applying textures and shading.

    Args:
        frame (np.ndarray): A 2D array representing the screen where the frame is drawn.
        points (np.ndarray): An array of 3D points representing the vertices of the model.
        triangles (np.ndarray): An array of indices representing the vertices of each triangle.
        camera (np.ndarray): The position and orientation of the camera.
        light_camera (np.ndarray): The position and orientation of the light source.
        z_buffer (np.ndarray): A buffer to keep track of depth information.
        textured (bool): Whether the model is textured or not.
        texture_uv (np.ndarray): The texture UV coordinates for each triangle.
        texture_map (np.ndarray): A map that associates each triangle with its corresponding UV coordinates.
        texture (np.ndarray): A 2D array representing the texture.
        shadow_points (np.ndarray): An array of points used for projecting shadows.
        shadow_map (np.ndarray): A 2D array representing the shadow map.
    """

    text_size = [len(texture) - 1, len(texture[0]) - 1]
    color_scale = 230 / np.max(np.abs(points[:, :3]))

    for index in range(len(triangles)):
        triangle = triangles[index]

        # calculate the vectors and normal for the current triangle
        vector1 = points[triangle[1]][:3] - points[triangle[0]][:3]
        vector2 = points[triangle[2]][:3] - points[triangle[0]][:3]
        normal = np.cross(vector1, vector2)
        normal /= np.linalg.norm(normal)

        # compute camera ray and light shading
        camera_ray = (points[triangle[0]][:3] - camera[:3]) / points[triangle[0]][5]
        xxs = [points[triangle[0]][3], points[triangle[1]][3], points[triangle[2]][3]]
        yys = [points[triangle[0]][4], points[triangle[1]][4], points[triangle[2]][4]]
        z_min = min([points[triangle[0]][5], points[triangle[1]][5], points[triangle[2]][5]])

        if not filter_triangles(z_min, normal, camera_ray, xxs, yys):
            continue

        light_ray = (points[triangle[0]][:3] - light_camera[:3]) / points[triangle[0]][5]
        light_ray /= np.linalg.norm(light_ray)
        shade1 = 0.2 + 0.8 * (- 0.5 * dot_product(light_ray, normal) + 0.5)

        # project points and shadows
        proj_points = points[triangle][:, 3:]
        proj_shadows = shadow_points[triangle][:, 3:]
        sorted_y = proj_points[:, 1].argsort()
        start, middle, stop = proj_points[sorted_y[0]], proj_points[sorted_y[1]], proj_points[sorted_y[2]]

        # precompute x and z slopes and value, respectively
        x_slopes = get_slopes(start[0], middle[0], stop[0], start[1], middle[1], stop[1])
        min_z = min(proj_points[0][2], proj_points[1][2], proj_points[2][2])
        z0, z1, z2 = 1 / proj_points[0][2], 1 / proj_points[1][2], 1 / proj_points[2][2]

        # texture or vertex colors
        if textured:
            uv_points = texture_uv[texture_map[index]]
            uv_points[0] *= z0
            uv_points[1] *= z1
            uv_points[2] *= z2
        else:
            color0 = (np.abs(points[triangles[index][0]][:3]) * color_scale + 25) * z0
            color1 = (np.abs(points[triangles[index][1]][:3]) * color_scale + 25) * z1
            color2 = (np.abs(points[triangles[index][2]][:3]) * color_scale + 25) * z2

        denominator = 1 / ((proj_points[1][1] - proj_points[2][1]) * (proj_points[0][0] - proj_points[2][0]) +
                           (proj_points[2][0] - proj_points[1][0]) * (proj_points[0][1] - proj_points[2][1]) + 1e-32)

        proj_shadows[0] *= z0
        proj_shadows[1] *= z1
        proj_shadows[2] *= z2

        # rasterization loop to fill the triangle
        for y in range(max(0, int(start[1])), min(SCREEN_HEIGHT, int(stop[1] + 1))):
            x1 = start[0] + int((y - start[1]) * x_slopes[0])
            x2 = start[0] + int((y - start[1]) * x_slopes[1]) if y < middle[1] else middle[0] + int((y - middle[1]) * x_slopes[2])
            min_x, max_x = max(0, min(x1, x2, SCREEN_WIDTH)), min(SCREEN_WIDTH, max(0, x1 + 1, x2 + 1))

            for x in range(int(min_x), int(max_x)):
                w0 = ((proj_points[1][1] - proj_points[2][1]) * (x - proj_points[2][0]) + (
                        proj_points[2][0] - proj_points[1][0]) * (y - proj_points[2][1])) * denominator
                w1 = ((proj_points[2][1] - proj_points[0][1]) * (x - proj_points[2][0]) + (
                        proj_points[0][0] - proj_points[2][0]) * (y - proj_points[2][1])) * denominator
                w2 = 1 - w0 - w1
                z = 1 / (w0 * z0 + w1 * z1 + w2 * z2 + 1e-32)

                if z_buffer[x, y] <= z or z < min_z:
                    continue

                z_buffer[x, y] = z
                shade2 = 1
                if shade1 < 0.6:
                    shade2 = shade1
                else:
                    point = (w0 * proj_shadows[0] + w1 * proj_shadows[1] + w2 * proj_shadows[2]) * z
                    lx, ly = max(0, min(SCREEN_WIDTH - 1, int(point[0]))), max(0, min(SCREEN_HEIGHT - 1, int(point[1])))

                    if point[2] > shadow_map[lx][ly]:
                        shade2 = min(0.9, 2.5 / np.sum(point[2] > shadow_map[lx - 1:lx + 1, ly - 1:ly + 1]))

                # apply texture or color
                if textured:
                    u = int((w0 * uv_points[0][0] + w1 * uv_points[1][0] + w2 * uv_points[2][0]) * z * text_size[0])
                    v = int((w0 * uv_points[0][1] + w1 * uv_points[1][1] + w2 * uv_points[2][1]) * z * text_size[1])
                    if 0 <= u < text_size[0] and 0 <= v < text_size[1]:
                        frame[x, y] = shade1 * shade2 * texture[u][v]
                else:
                    color = (w0 * color0 + w1 * color1 + w2 * color2) * z
                    frame[x, y] = shade1 * shade2 * color


@njit()
def get_slopes(num_start, num_middle, num_stop, den_start, den_middle, den_stop):
    """
    Calculates the slopes between three points using their numerators and denominators.

    Args:
        num_start (float): Numerator of the starting point.
        num_middle (float): Numerator of the middle point.
        num_stop (float): Numerator of the stopping point.
        den_start (float): Denominator of the starting point.
        den_middle (float): Denominator of the middle point.
        den_stop (float): Denominator of the stopping point.

    Returns:
        np.ndarray: Array of three slopes calculated between start-middle, start-stop, and middle-stop points.
    """
    return np.asarray([(num_stop - num_start) / (den_stop - den_start + 1e-32),
                       (num_middle - num_start) / (den_middle - den_start + 1e-32),
                       (num_stop - num_middle) / (den_stop - den_middle + 1e-32)])


@njit()
def filter_triangles(z_min, normal, camera_ray, xxs, yys, scale=1):
    """
    Filters triangles based on their visibility and position relative to the screen and camera.

    Args:
        z_min (float): The minimum z-depth of the triangle.
        normal (np.ndarray): Normal vector of the triangle.
        camera_ray (np.ndarray): Ray from the camera to the triangle.
        xxs (list): List of x-coordinates of the triangle's vertices.
        yys (list): List of y-coordinates of the triangle's vertices.
        scale (float, optional): Scaling factor for screen dimensions. Default is 1.

    Returns:
        bool: Whether the triangle should be rendered or not.
    """

    within_x_bounds = max(xxs) >= 0 and min(xxs) < SCREEN_WIDTH * scale
    within_y_bounds = max(yys) >= 0 and min(yys) < SCREEN_HEIGHT * scale
    return z_min > 0 > dot_product(normal, camera_ray) and within_x_bounds and within_y_bounds


def render_text(screen, location, text, size, color, bold=False):
    font = pygame.font.Font('assets/system/font.ttf', size)
    font.set_bold(bold)
    screen.blit(font.render(text, True, color), location)


def draw_loading(surface):
    surface.fill((0, 0, 0))
    render_text(surface, (50, (SCREEN_HEIGHT - 120) / 2), 'Loading', 30, (255, 255, 255))
    render_text(surface, (50, (SCREEN_HEIGHT - 120) / 2 + 50), 'SILVER', 70, (255, 255, 255), True)


def log(prefix, message, data=None):
    if data is None:
        data = []
    if prefix == 'break':
        print()
        return

    color_codes = {'warning': '\033[1;33m', 'error': '\033[0;31m', 'log': '\033[0;34m'}
    print(end=color_codes.get(prefix, ''))

    print(f'{prefix.upper()}: {message}')
    for item in data:
        print(f'{" " * (len(prefix) + 2)}- {item}')
    print(end='\033[0m')
