import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl_control.models import define_model
from rl_control.DroneControlEnv import DroneControlEnv

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def train_model(seed=1, setup=0):
    np.random.seed(seed)

    if setup == 0:
        env = DroneControlEnv(a_p=0, a_r=0, e_thres=0)
    elif setup == 1:
        env = DroneControlEnv(a_p=0, a_r=0, e_thres=0.8)
    else:
        env = DroneControlEnv(a_p=0.5, a_r=0.2, e_thres=0.8)

    env.seed(seed)

    model = define_model()

    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0.1, value_test=0.05,
                                  nb_steps=95000)
    dqn = DQNAgent(model=model, nb_actions=5, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1, target_model_update=0.001, batch_size=32)
    dqn.compile(RMSprop(lr=.0001), metrics=['mae'])

    log_filename = 'results/drone_control_' + str(setup) + '_log_.json'
    model_checkpoint_filename = 'results/drone_cnn_weights_' + str(setup) + '_{step}.model'
    callbacks = [ModelIntervalCheckpoint(model_checkpoint_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=1)]

    dqn.fit(env, nb_steps=100000, nb_max_episode_steps=50, verbose=2, visualize=False, log_interval=1,
            callbacks=callbacks)

    model_filename = 'models/drone_cnn_' + str(setup) + '.model'
    dqn.save_weights(model_filename, overwrite=True)


def evaluate_model(model_path=None, interactive=False, seed=12345):
    np.random.seed(seed)

    model = define_model(5)
    memory = SequentialMemory(limit=10000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0, value_min=0, value_test=0, nb_steps=1)
    dqn = DQNAgent(model=model, nb_actions=5, policy=policy, memory=memory, processor=None,
                   nb_steps_warmup=500, gamma=0.95, delta_clip=1, target_model_update=0.001, batch_size=32)
    dqn.compile(RMSprop(lr=.0001), metrics=['mae'])
    dqn.load_weights(model_path)

    # Train Eval
    env = DroneControlEnv(testing=False, interactive=interactive)
    env.seed(seed)
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=50, verbose=0, visualize=True)
    train_mean_reward = np.mean(res.history['episode_reward'])
    before_train_tilt_error = np.mean(np.abs(env.init_tilt_error))
    before_train_pan_error = np.mean(np.abs(env.init_pan_error))
    after_train_tilt_error = np.mean(np.abs(env.final_tilt_error))
    after_train_pan_error = np.mean(np.abs(env.final_pan_error))
    print("Training evaluation: ")
    print("Mean reward: ", train_mean_reward)
    print("Tilt: ", before_train_tilt_error, " -> ", after_train_tilt_error)
    print("Pan: ", before_train_pan_error, " -> ", after_train_pan_error)

    # Test Eval
    env = DroneControlEnv(testing=True, interactive=interactive)
    env.seed(seed)
    res = dqn.test(env, nb_episodes=500, nb_max_episode_steps=50, verbose=0, visualize=True)
    train_mean_reward = np.mean(res.history['episode_reward'])
    before_train_tilt_error = np.mean(np.abs(env.init_tilt_error))
    before_train_pan_error = np.mean(np.abs(env.init_pan_error))
    after_train_tilt_error = np.mean(np.abs(env.final_tilt_error))
    after_train_pan_error = np.mean(np.abs(env.final_pan_error))
    print("Testing evaluation: ")
    print("Mean reward: ", train_mean_reward)
    print("Tilt: ", before_train_tilt_error, " -> ", after_train_tilt_error)
    print("Pan: ", before_train_pan_error, " -> ", after_train_pan_error)


if __name__ == '__main__':
    # train_model(setup=0)
    # train_model(setup=1)
    # train_model(setup=2)

    evaluate_model('models/drone_cnn_weights_0.model')
    evaluate_model('models/drone_cnn_weights_1.model')
    evaluate_model('models/drone_cnn_weights_2.model')
