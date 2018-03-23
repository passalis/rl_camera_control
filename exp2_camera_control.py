import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import numpy as np
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl_control.models import define_model
from rl_control.CameraEnviroment import CameraControlEnv

def train_model(seed=1):
    np.random.seed(seed)

    env = CameraControlEnv()
    env.seed(seed)

    model = define_model(actions=7)

    memory = SequentialMemory(limit=500, window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0.1, value_test=0.05,
                                  nb_steps=1900000)

    dqn = DQNAgent(model=model, nb_actions=7, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1, target_model_update=500, batch_size=32)

    dqn.compile(RMSprop(lr=.00005), metrics=['mae'])

    log_filename = 'results/drone_camera_control_log.json'
    model_checkpoint_filename = 'results/drone_camera_cnn_weights_{step}.model'
    callbacks = [ModelIntervalCheckpoint(model_checkpoint_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=1)]

    dqn.fit(env, nb_steps=2000000, nb_max_episode_steps=100, verbose=2, visualize=False, log_interval=1,
            callbacks=callbacks)

    # After training is done, save the final weights.
    model_filename = 'models/drone_camera_cnn.model'
    dqn.save_weights(model_filename, overwrite=True)




def evaluate_model(model_path=None, interactive=False, seed=12345):

    np.random.seed(seed)
    model = define_model(actions=7)
    memory = SequentialMemory(limit=500, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05, nb_steps=50000)
    dqn = DQNAgent(model=model, nb_actions=7, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1.0, target_model_update=500, batch_size=32)
    dqn.compile(RMSprop(lr=.00005), metrics=['mae'])
    if model_path is not None:
        dqn.load_weights(model_path)

    # Train Evaluation
    env = CameraControlEnv(dataset_pickle_path='data/dataset.pickle', testing=False, interactive=interactive)
    env.seed(seed)
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=100, verbose=0, visualize=True)
    train_mean_reward = np.mean(res.history['episode_reward'])
    before_train_position_error = np.mean(np.abs(env.init_position_error_pixels))
    before_train_zoom_error = np.mean(np.abs(env.init_zoom_error_pixels))
    after_train_position_error = np.mean(np.abs(env.final_position_error_pixels))
    after_train_zoom_error = np.mean(np.abs(env.final_zoom_error_pixels))
    print("Training evaluation: ")
    print("Mean reward: ", train_mean_reward)
    print("Position: ", before_train_position_error, " -> ", after_train_position_error)
    print("Zoom: ", before_train_zoom_error, " -> ", after_train_zoom_error)

    # Test Evaluation
    env = CameraControlEnv(dataset_pickle_path='data/dataset.pickle', testing=True, interactive=interactive)
    env.seed(seed)
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=100, verbose=0, visualize=True)
    train_mean_reward = np.mean(res.history['episode_reward'])
    before_train_position_error = np.mean(np.abs(env.init_position_error_pixels))
    before_train_zoom_error = np.mean(np.abs(env.init_zoom_error_pixels))
    after_train_position_error = np.mean(np.abs(env.final_position_error_pixels))
    after_train_zoom_error = np.mean(np.abs(env.final_zoom_error_pixels))
    print("Testing evaluation: ")
    print("Mean reward: ", train_mean_reward)
    print("Position: ", before_train_position_error, " -> ", after_train_position_error)
    print("Zoom: ", before_train_zoom_error, " -> ", after_train_zoom_error)


if __name__ == '__main__':

    # train_model()
    evaluate_model(model_path='models/camera_control.model')

