import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Activation, SeparableConv2D, MaxPool2D, AveragePooling2D, concatenate

# generic model design
def model_fn(actions):
    B = len(actions) // 4
    action_list = np.split(np.array(actions), len(actions) // 2)

    ip = Input(shape=(32, 32, 3))
    x = build_cell(ip, action_list, B, stride=(2, 2))
    x = build_cell(x, action_list, B, stride=(2, 2))
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    return model

def parse_action(ip, action, strides=(1, 1)):

    if action == '3x3 dconv':
        return SeparableConv2D(24, (3, 3), strides=strides, padding='same', activation='relu')(ip)

    if action == '5x5 dconv':
        return SeparableConv2D(24, (5, 5), strides=strides, padding='same', activation='relu')(ip)

    if action == '7x7 dconv':
        return SeparableConv2D(24, (5, 5), strides=strides, padding='same', activation='relu')(ip)

    if action == '1x7-7x1 conv':
        x = Conv2D(24, (1, 7), strides=strides, padding='same', activation='relu')(ip)
        x = Conv2D(24, (7, 1), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    if action == '3x3 conv':
        return Conv2D(24, (3, 3), strides=strides, padding='same', activation='relu')(ip)

    if action == '3x3 maxpool':
        return MaxPool2D((2, 2), strides=strides)(ip)

    if action == '3x3 avgpool':
        return AveragePooling2D((2, 2), strides=strides)(ip)

    if strides == (2, 2):
        return Conv2D(24, (1, 1), strides=strides, padding='same', activation='relu')(ip)
    else:
        return Activation('linear')(ip)


def build_cell(ip, action_list, B, stride):
    if B == 1:
        left = parse_action(ip, action_list[0][1], strides=stride)
        right = parse_action(ip, action_list[1][1], strides=stride)
        return concatenate([left, right], axis=-1)

    actions = []
    for i in range(B):
        left_action = parse_action(ip, action_list[i * 2][1], strides=stride)
        right_action = parse_action(ip, action_list[i * 2 + 1][1], strides=stride)
        action = concatenate([left_action, right_action], axis=-1)
        actions.append(action)

    op = concatenate(actions, axis=-1)
    return op




