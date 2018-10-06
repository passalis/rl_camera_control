import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl_control.models import define_model
from rl_control.CameraEnviroment import CameraControlEnv


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def train_model(seed=1, setup=0):
    np.random.seed(seed)

    if setup == 0:
        env = CameraControlEnv(a_p=0, a_r=0, e_thres=0)
    elif setup == 1:
        env = CameraControlEnv(a_p=0, a_r=0, e_thres=0.8)
    else:
        env = CameraControlEnv(a_p=0.5, a_r=0.2, e_thres=0.8)

    env.seed(seed)

    model = define_model(actions=7)

    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0.1, value_test=0.05,
                                  nb_steps=95000)
    dqn = DQNAgent(model=model, nb_actions=7, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1, target_model_update=0.001, batch_size=32)
    dqn.compile(RMSprop(lr=.0001), metrics=['mae'])

    log_filename = 'results/drone_camera_control_log_' + str(setup) + '.json'
    model_checkpoint_filename = 'results/drone_camera_cnn_weights_' + str(setup) + '_{step}.model'
    callbacks = [ModelIntervalCheckpoint(model_checkpoint_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=1)]

    dqn.fit(env, nb_steps=100000, nb_max_episode_steps=100, verbose=2, visualize=False, log_interval=1,
            callbacks=callbacks)

    # After training is done, save the final weights.
    model_filename = 'models/drone_camera_cnn_' + str(setup) + '.model'
    dqn.save_weights(model_filename, overwrite=True)


def evaluate_model(model_path=None, interactive=False, seed=12345):
    np.random.seed(seed)
    model = define_model(actions=7)
    memory = SequentialMemory(limit=1000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0,
                                  value_min=.0, value_test=0, nb_steps=50000)
    dqn = DQNAgent(model=model, nb_actions=7, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1.0, target_model_update=0.001, batch_size=32)
    dqn.compile(RMSprop(lr=.00005), metrics=['mae'])
    if model_path is not None:
        dqn.load_weights(model_path)

    # Train Evaluation
    env = CameraControlEnv(dataset_pickle_path='data/dataset.pickle', testing=False, interactive=interactive)
    env.seed(seed)
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=100, verbose=0, visualize=False)
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
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=100, verbose=0, visualize=False)
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
    # train_model(setup=0)
    # train_model(setup=1)
    # train_model(setup=2)

    evaluate_model(model_path='models/drone_camera_cnn_weights_0.model')
    evaluate_model(model_path='models/drone_camera_cnn_weights_1.model')
    evaluate_model(model_path='models/drone_camera_cnn_weights_2.model')
