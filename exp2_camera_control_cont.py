import tensorflow as tf
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
import numpy as np
from keras.optimizers import RMSprop, Adam
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl_control.models import define_actor_critic_models
from rl_control.CameraEnviromentCont import CameraControlEnvCont


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def train_model(seed=1):
    np.random.seed(seed)
    env = CameraControlEnvCont()
    env.seed(seed)

    actor, critic, action_input = define_actor_critic_models(actions=3)

    memory = SequentialMemory(limit=10000, window_length=1)

    random_process = GaussianWhiteNoiseProcess(mu=0, sigma=0.1, sigma_min=0.01, n_steps_annealing=49000, size=3)

    agent = DDPGAgent(nb_actions=3, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=500,
                      random_process=random_process, gamma=.1, target_model_update=1e-3, batch_size=32)
    agent.compile([RMSprop(lr=.0001), RMSprop(lr=.01)], metrics=['mae'])

    log_filename = 'results/drone_camera_cont_control_log.json'
    model_checkpoint_filename = 'results/drone_camera_cont_cnn_weights_{step}.model'
    callbacks = [ModelIntervalCheckpoint(model_checkpoint_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=1)]

    agent.fit(env, nb_steps=50000, nb_max_episode_steps=100, verbose=2, visualize=False, log_interval=1,
              callbacks=callbacks)


def evaluate_model(model_path=None, interactive=False, seed=12345):
    np.random.seed(seed)

    actor, critic, action_input = define_actor_critic_models(actions=3)
    memory = SequentialMemory(limit=10000, window_length=1)
    random_process = GaussianWhiteNoiseProcess(mu=0, sigma=0, sigma_min=0, n_steps_annealing=1)

    agent = DDPGAgent(nb_actions=3, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.95, target_model_update=0.0001, batch_size=32)
    agent.compile([RMSprop(lr=.0001), RMSprop(lr=.01)], metrics=['mae'])

    if model_path is not None:
        agent.load_weights(model_path)

    # Train Evaluation
    env = CameraControlEnvCont(dataset_pickle_path='data/dataset.pickle', testing=False, interactive=interactive)
    env.seed(seed)
    res = agent.test(env, nb_episodes=500, nb_max_episode_steps=100, verbose=0, visualize=False)
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
    env = CameraControlEnvCont(dataset_pickle_path='data/dataset.pickle', testing=True, interactive=interactive)
    env.seed(seed)
    res = agent.test(env, nb_episodes=500, nb_max_episode_steps=100, verbose=0, visualize=False)
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
    evaluate_model(model_path='models/drone_camera_cont_cnn_weights.model')
