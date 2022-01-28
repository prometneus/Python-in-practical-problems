from vispy import app, scene
from functions import *
from scipy.spatial.distance import cdist
from numba import prange
# from scipy.spatial import KDTree
import ffmpeg

app.use_app('pyglet')

# %% Globals and other variables implementation


aspect = 16 / 9  # To preserve the proportions of the window, no matter how we change the sizes
window_height = 720  # Window size corresponding HD
window_width = int(window_height * aspect)

boids_number = 10000  # Number of agents
dt = 0.05  # Time step
velocity_range = np.array([0.05, 0.1])
acceleration_range = np.array([0.0, 1.0])
threshold = 0.05

# Set 1: Small circles
# coefficients = np.array([0.0,  # Alignment
#                          50.0,  # Cohesion
#                          0.0,  # Separation
#                          0.005,  # Walls of area
#                          0.02])  # Noise

# Set 2: Pulsating dots or a structure
# coefficients = np.array([2.0,  # Alignment
#                          2000.0,  # Cohesion
#                          2.5,  # Separation
#                          0.015,  # Walls of area
#                          0.01])  # Noise

# Set 3: Two something with tails
coefficients = np.array([1.25,  # Alignment
                         1.0,  # Cohesion
                         0.0,  # Separation
                         0.02,  # Walls of area
                         0.01])  # Noise

boids = np.zeros((boids_number, 6), dtype=float)  # 2 coords, 2 speeds, 2 accelerations
boids_initialization(boids, aspect, velocity_range)
boids[:, 2:4] = clip_magnitude(boids[:, 2:4], velocity_range[0], velocity_range[1])
# %% Scene formatting


canvas = scene.SceneCanvas(keys='interactive', show=True, size=(window_width, window_height))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=(0, 0, aspect, 1), aspect=1)

agents = scene.Arrow(arrows=directions(boids), arrow_color=(1, 1, 1, 1),  # Agent's shape
                     arrow_size=5, connect='segments', parent=view.scene)

scene.Line(pos=np.array([[0, 0], [aspect, 0], [aspect, 1], [0, 1], [0, 0]]),  # Zone
           color=(0, 0, 0, 1), connect='strip', method='gl', parent=view.scene)

scene.visuals.Text(text='Number of boids: ' + str(boids_number),
                   pos=[0.15, 0.07], face='Ubuntu', parent=view.scene, color=(0.954, 0.71, 0, 1))
scene.visuals.Text(text=f'a={coefficients[0]}, b={coefficients[1]}, '
                        f'c={coefficients[2]}, d={coefficients[3]}, e={coefficients[4]}',
                   pos=[0.2, 0.03], face='Ubuntu', parent=view.scene, color=(0.954, 0.71, 0, 1))
scene_fps = scene.visuals.Text(text='FPS: ' + str(canvas.fps),
                               pos=[0.15, 0.05], face='Ubuntu', parent=view.scene, color=(0.954, 0.71, 0, 1))
# %% Video parameters


video = False
if video:
    filename = f"/home/artem/Programming Python/aa_{boids_number}.mp4"

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{window_width}x{window_height}', r=60)
            .output(filename, pix_fmt='yuv420p', preset='slower', r=60)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
# %% Like main function. No less significant than main


def update(event):
    """
    Updates all the data while simulating agents' motion. Allows to record a video.

    Parameters
    ----------
    event: event.
    Class describing events that occur and can be reacted to with callbacks.

    Returns
    -------
    None.
    """
    global process
    scene_fps.text = 'FPS: ' + str(canvas.fps)
    distance_matrix = cdist(boids[:, :2], boids[:, :2])
    neighbours_coordinates = neighbours_detection(distance_matrix, threshold)
    walls_influence = avoid_walls(boids, aspect)
    noises = noise(boids)
    for i in prange(boids_number):
        neighbours_indexes = np.where(neighbours_coordinates[i])[0]
        accelerations = np.zeros((5, 2), dtype=np.float64)
        if neighbours_indexes.size > 0:
            # print(tree.query(boids[i, :2]))
            accelerations[0] = alignment(boids, i, neighbours_indexes)
            accelerations[1] = cohesion(boids, i, neighbours_indexes)
            accelerations[2] = separation(boids, i, neighbours_indexes, distance_matrix)
        accelerations[3] = walls_influence[i]
        accelerations[4] = noises[i]
        clip_magnitude(accelerations, acceleration_range[0], acceleration_range[1])
        boids[i, 4:6] = np.sum(accelerations * coefficients[:, None], axis=0)
    movement(boids, dt, velocity_range)  # Calculates movement
    jail(boids, aspect)  # Imprison agents
    agents.set_data(arrows=directions(boids))
    if video:
        frame = canvas.render(alpha=False)
        process.stdin.write(frame.tobytes())
        if event.count > 1800:  # To stop the process and save a video
            app.quit()
            exit()
    else:
        canvas.update(event)


# %%


update_timer = app.Timer(interval=0, start=True, connect=update)

if __name__ == '__main__':
    canvas.measure_fps()
    app.run()
