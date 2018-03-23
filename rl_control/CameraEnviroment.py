import cv2
import numpy as np
import gym
from gym import spaces
import pickle


class CameraControlEnv(gym.Env):

    def __init__(self, dataset_pickle_path='data/dataset.pickle', testing=False, interactive=False):
        """
        Initializes the camera control environment
        :param dataset_pickle_path: the dataset pickle
        :param testing: True -> evaluation on test set, False -> evaluation on train set
        :param interactive: if set to True, then waits for user input before proceeding to the next action
        """

        self.metadata = {'render.modes': ['human'], }

        # Load data
        with open(dataset_pickle_path, 'rb') as f:
            self.persons = pickle.load(f)

        # List of possible tilts and pans
        self.tilts = [-30, -15, 0, +15, +30]
        self.pans = [-45, -30, -15, 0, +15, +30, +45]

        self.testing = testing

        # Size of the frame on which we are working on
        self.frame_size = 512

        # Margin around the corners
        self.margin = 20
        # Zoom limits
        self.zoom_range = (64, 192)

        # Size of the actual input to the CNN
        self.observation_size = 64
        self.movement_step = 5
        self.zoom_step = 10

        # Create state
        self.current_tilt = 0
        self.current_pan = 0
        self.current_center = [0, 0]
        self.current_window_zoom = 128

        # Center range for initializing the environment
        self.min_range = self.current_window_zoom + self.margin
        self.max_range = self.frame_size - self.current_window_zoom - self.margin

        # Targets for calculating the error
        self.target_zoom = 96
        self.face_position = [0, 0, 0, 0]

        if self.testing:
            self.current_person = np.random.randint(20, 30)
        else:
            self.current_person = np.random.randint(0, 20)

        # Mean RGB vector for normalizing input frames
        self.mean_vector = np.asarray([109, 114, 131])

        # Create environment state
        self.observation = np.zeros((self.observation_size, self.observation_size, 3,))
        self.render_observation = np.zeros((self.frame_size, self.frame_size, 3,))
        self.action_space = spaces.Discrete(7)  # no action, left, right, up, down, zoom_in, zoom_out
        self.observation_space = spaces.Box(-127 * np.ones(self.observation.shape),
                                            127 * np.ones(self.observation.shape), dtype='float32')

        # Keep some statistics
        self.error_memory = []

        # Statistics for testing
        self.epoch_counter = -1
        self.iteration_counter = 0
        self.init_zoom_error = []
        self.init_position_error = []
        self.final_zoom_error = []
        self.final_position_error = []

        self.init_zoom_error_pixels = []
        self.init_position_error_pixels = []
        self.final_zoom_error_pixels = []
        self.final_position_error_pixels = []

        self.current_state_text = ''
        self.action_text = 'None'
        self.interactive = interactive

    def reset(self):

        # Select zoom range
        self.current_window_zoom = np.random.random_integers(self.zoom_range[0], self.zoom_range[1])
        self.min_range = self.current_window_zoom + self.margin
        self.max_range = self.frame_size - self.current_window_zoom - self.margin

        # Select a random person
        if self.testing:
            self.current_person = np.random.randint(20, 30)
        else:
            self.current_person = np.random.randint(0, 20)

        # Select a random tilt and pan to init the environment
        self.current_tilt = np.random.randint(0, len(self.tilts))
        self.current_pan = np.random.randint(0, len(self.pans))

        # Reset statistics
        self.error_memory = []

        # Select a random tilt and pan to init the environment
        self.current_center = [np.random.randint(self.min_range, self.max_range),
                               np.random.randint(self.min_range, self.max_range)]

        # Retrieve observation
        tilt, pan = self.tilts[self.current_tilt], self.pans[self.current_pan]

        _, render_observation, face_position = self.persons[self.current_person][(tilt, pan)]
        self.render_observation = np.zeros((self.frame_size, self.frame_size, 3), dtype='uint8')

        # Fill the space around the image
        for i in range(3):
            # Sample the color from the upper left corner
            current_color = np.mean(render_observation[5:10, 5:10, i])

            self.render_observation[:, :, i] = current_color

            # Remove the black strip at left and bottom
            render_observation[:, 0:5, i] = current_color
            render_observation[-2:, :, i] = current_color

        n_upper_limit = self.frame_size - 256
        shift_x = np.random.random_integers(0, n_upper_limit)
        shift_y = np.random.random_integers(0, n_upper_limit)

        self.render_observation[shift_y:shift_y + 256, shift_x:shift_x + 256, :] = render_observation

        self.face_position[0] = face_position[0] + shift_x
        self.face_position[1] = face_position[1] + shift_y
        self.face_position[2] = face_position[2] + shift_x
        self.face_position[3] = face_position[3] + shift_y

        # Create the actual observation
        self.observation = self.render_observation.copy()
        self.observation = self.observation[self.current_center[1] -
                                            self.current_window_zoom:self.current_center[1] + self.current_window_zoom,
                           self.current_center[0] -
                           self.current_window_zoom:self.current_center[0] + self.current_window_zoom, :]
        self.observation = cv2.resize(self.observation, (self.observation_size, self.observation_size))
        self.observation = self.observation - self.mean_vector

        # Update statistics
        self.epoch_counter += 1
        self.iteration_counter = 0

        face_center = (
        (self.face_position[0] + self.face_position[2]) // 2, (self.face_position[1] + self.face_position[3]) // 2)

        zoom_error = ((self.current_window_zoom - self.target_zoom) / (self.zoom_range[1] - self.zoom_range[0])) ** 2
        position_error = ((self.current_center[0] - face_center[0]) / (0.5 * self.frame_size)) ** 2 + \
                         ((self.current_center[1] - face_center[1]) / (0.5 * self.frame_size)) ** 2

        self.error_memory.append(zoom_error + position_error)
        self.init_zoom_error.append(zoom_error)
        self.init_position_error.append(position_error)
        self.final_zoom_error.append(zoom_error)
        self.final_position_error.append(position_error)

        pixel_zoom_error = np.abs(self.current_window_zoom - self.target_zoom)
        pixel_position_error = np.abs(self.current_center[0] - face_center[0]) + np.abs(
            self.current_center[1] - face_center[1])

        self.init_zoom_error_pixels.append(pixel_zoom_error)
        self.init_position_error_pixels.append(pixel_position_error)
        self.final_zoom_error_pixels.append(pixel_zoom_error)
        self.final_position_error_pixels.append(pixel_position_error)

        # Reset text
        self.current_state_text = '{0:.4f}'.format(zoom_error) + '/' + '{0:.4f}'.format(position_error)
        self.action_text = 'None'

        return self.observation

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):

        # Perform an action
        if action == 0:
            pass
        elif action == 1:  # down
            self.current_center[1] += self.movement_step
        elif action == 2:  # up
            self.current_center[1] -= self.movement_step
        elif action == 3:  # right
            self.current_center[0] += self.movement_step
        elif action == 4:  # left
            self.current_center[0] -= self.movement_step
        elif action == 5:  # zoom in
            self.current_window_zoom -= self.zoom_step
        elif action == 6:  # zoom out
            self.current_window_zoom += self.zoom_step
        else:
            assert False

        if self.current_window_zoom < self.zoom_range[0]:
            self.current_window_zoom = self.zoom_range[0]

        if self.current_window_zoom > self.zoom_range[1]:
            self.current_window_zoom = self.zoom_range[1]

        # Recalculate the margins (needed if the zoom is changed)
        self.min_range = self.current_window_zoom + self.margin
        self.max_range = self.frame_size - self.current_window_zoom - self.margin

        # Ensure that we remain within the limits
        for i in range(2):
            if self.current_center[i] < self.min_range:
                self.current_center[i] = self.min_range
            if self.current_center[i] > self.max_range:
                self.current_center[i] = self.max_range

        # Create the actual observation
        self.observation = self.render_observation.copy()
        self.observation = self.observation[self.current_center[1] -
                                            self.current_window_zoom:self.current_center[1] + self.current_window_zoom,
                           self.current_center[0] - self.current_window_zoom:self.current_center[0]
                                                                             + self.current_window_zoom, :]
        self.observation = cv2.resize(self.observation, (self.observation_size, self.observation_size))
        self.observation = self.observation - self.mean_vector

        # Prepare the text for the observation
        if action == 0:
            action_text = 'Stay'
        elif action == 1:
            action_text = 'Down'
        elif action == 2:
            action_text = 'Up'
        elif action == 3:
            action_text = 'Right'
        elif action == 4:
            action_text = 'Left'
        elif action == 5:
            action_text = 'Zoom in'
        elif action == 6:
            action_text = 'Zoom out'

        face_center = ((self.face_position[0] + self.face_position[2]) // 2,
                       (self.face_position[1] + self.face_position[3]) // 2)

        zoom_error = ((self.current_window_zoom - self.target_zoom) / (self.zoom_range[1] - self.zoom_range[0])) ** 2

        position_error = ((self.current_center[0] - face_center[0]) / (0.5 * self.frame_size)) ** 2 + \
                         ((self.current_center[1] - face_center[1]) / (0.5 * self.frame_size)) ** 2

        current_error = zoom_error + position_error
        self.error_memory.append(current_error)
        self.final_zoom_error[-1] = zoom_error
        self.final_position_error[-1] = position_error

        pixel_zoom_error = np.abs(self.current_window_zoom - self.target_zoom)
        pixel_position_error = np.abs(self.current_center[0] - face_center[0]) \
                               + np.abs(self.current_center[1] - face_center[1])
        self.final_zoom_error_pixels[-1] = pixel_zoom_error
        self.final_position_error_pixels[-1] = pixel_position_error

        # Reset text
        self.current_state_text = '{0:.4f}'.format(zoom_error) + '/' + '{0:.4f}'.format(position_error)
        self.action_text = action_text

        # Calculate and normalize reward
        reward = (3 - current_error) / 3.0

        # Clip and scale the reward
        if reward < 0.95:
            reward = 0
        else:
            reward = (reward - 0.95) / 0.05

        # Add reward/punishment for correct/wrong movements
        if self.error_memory[-2] > self.error_memory[-1]:
            reward += 0.1
        elif self.error_memory[-2] < self.error_memory[-1]:
            reward += -0.15

        self.iteration_counter += 1
        done = False
        return self.observation, reward, done, {}

    def render(self, mode='human', close=False):

        img = self.render_observation.copy()
        cv2.putText(img, "Errors: " + self.current_state_text, (10, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0),
                    1)
        cv2.putText(img, "Sel. Action: " + self.action_text, (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0), 1)

        # Annotate current view
        cv2.rectangle(img,
                      (self.current_center[0] - self.current_window_zoom,
                       self.current_center[1] - self.current_window_zoom),
                      (self.current_center[0] + self.current_window_zoom,
                       self.current_center[1] + self.current_window_zoom),
                      (255, 0, 0), 3)

        cv2.imshow('View', img)
        cv2.imshow('CNN Input', np.uint8(self.observation + self.mean_vector))

        if self.interactive:
            print("Press any key to proceed")
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

    def __str__(self):
        return 'CameraControlEnviroment'
