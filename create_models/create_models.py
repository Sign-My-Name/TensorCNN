from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Flatten
from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


def create_resnet50(input_shape, lr, n_classes):
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # for l in base_model.layers:
    #     l.trainable = False
    input = Input(shape=input_shape)
    x = resnet50.preprocess_input(input)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=lr), metrics=['accuracy'])
    return model

def create_resnet50_with_conv(input_shape, lr, n_classes):
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for l in base_model.layers[:-4]:
        l.trainable = False
    input = Input(shape=input_shape)
    x = resnet50.preprocess_input(input)
    x = base_model(x)
    x = Conv2D(filters=128,kernel_size=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=lr), metrics=['accuracy'])
    return model

def create_small_resnet50_with_conv(input_shape, lr, n_classes):
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for l in base_model.layers:
        l.trainable = False
    base_model = Model(base_model.input, base_model.get_layer('conv4_block6_out').output)
    input = Input(shape=input_shape)
    x = resnet50.preprocess_input(input)
    x = base_model(x)
    x = Conv2D(filters=16,kernel_size=(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=lr), metrics=['accuracy'])
    return model