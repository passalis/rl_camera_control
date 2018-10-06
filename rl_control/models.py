from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Reshape, Conv2D, BatchNormalization, MaxPooling2D, \
    Concatenate
from keras.models import Model
from keras.initializers import RandomNormal, Zeros


def define_model(actions=5):
    model = Sequential()
    model.add(Reshape((64, 64, 3), input_shape=(1, 64, 64, 3)))

    ## Block 1
    model.add(Conv2D(16, (5, 5), name='conv_layer_1', strides=2))
    model.add(BatchNormalization(name='batch_norm_layer_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ## Block 2
    model.add(Conv2D(32, (3, 3), name='conv_layer_2', strides=1))
    model.add(BatchNormalization(name='batch_norm_layer_2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ## Block 3
    model.add(Conv2D(64, (3, 3), name='conv_layer_3', strides=1))
    model.add(BatchNormalization(name='batch_norm_layer_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ## FC Layers
    model.add(Flatten())
    model.add(Dense(128, name='regression_fc_1'))
    model.add(Activation('relu'))
    model.add(Dense(64, name='rl_fc_1'))
    model.add(Activation('relu'))
    model.add(Dense(actions, name='rl_fc_2'))
    model.add(Activation('linear'))

    return model


def build_common(state):
    ## Block 1
    b1 = Conv2D(16, (5, 5), input_shape=(64, 64, 3), name='conv_layer_1', strides=2)(state)
    b1 = BatchNormalization(name='batch_norm_layer_1')(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling2D(pool_size=(2, 2))(b1)

    ## Block 2
    b2 = Conv2D(32, (3, 3), name='conv_layer_2', strides=1)(b1)
    b2 = BatchNormalization(name='batch_norm_layer_2')(b2)
    b2 = Activation('relu')(b2)
    b2 = MaxPooling2D(pool_size=(2, 2))(b2)

    ## Block 3
    b3 = Conv2D(64, (3, 3), name='conv_layer_3', strides=1)(b2)
    b3 = BatchNormalization(name='batch_norm_layer_3')(b3)
    b3 = Activation('relu')(b3)
    b3 = MaxPooling2D(pool_size=(2, 2))(b3)

    ## FC Layers
    shared = Flatten()(b3)
    shared = Dense(128, name='regression_fc_1')(shared)
    shared = Activation('relu')(shared)
    return shared


def define_actor_critic_models(actions=3):
    state_in = Input(batch_shape=(None, 1, 64, 64, 3))
    state = Reshape((64, 64, 3), input_shape=(1, 64, 64, 3))(state_in)
    action_input = Input(shape=(actions,), name='action_input')

    # Actor
    shared = build_common(state)
    hidden_actor = Dense(64, name='rl_fc_1', activation='relu')(shared)
    actor_output = Dense(actions, name='rl_fc_2_actor', activation='tanh',
                         kernel_initializer=RandomNormal(stddev=0.001), bias_initializer=Zeros())(hidden_actor)
    actor = Model(inputs=state_in, outputs=actor_output)

    # Critic
    # shared = build_common(state)

    critic_input = Concatenate()([action_input, shared])
    hidden_critic = Dense(64, name='rl_fc_1_critic', activation='relu')(critic_input)
    critic_output = Dense(1, name='rl_fc_2_critic', activation='linear', kernel_initializer=RandomNormal(stddev=0.001),
                          bias_initializer=Zeros())(hidden_critic)
    critic = Model(inputs=[action_input, state_in], outputs=critic_output)

    return actor, critic, action_input
