# Preprocessing
import numpy as np

# Visualization
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.transform import resize
import matplotlib.image

# Deep Learning
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Dense, Reshape, Input, Flatten


def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_channel, a_channel, b_channel = tf.unstack(lab, axis=2)
        return [L_channel / 50 - 1, a_channel / 110, b_channel / 110]


def deprocess_lab(L_channel, a_channel, b_channel):
    with tf.name_scope('deprocess_lab'):
        return tf.stack([(L_channel + 1) / 2 * 100, a_channel * 110, b_channel * 110], axis=2)


def check_image(image):
    ass = tf.assert_equal(tf.shape(image)[-1], 3, message='зображення повинно мати 3 кольорові канали')
    with tf.control_dependencies([ass]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('зображення повинно бути 3-х або 4-х мірним')

    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def rgb_to_lab(im_rgb):
    with tf.name_scope('rgb_to_lab'):
        im_rgb = check_image(im_rgb)
        im_rgb_pixels = tf.reshape(im_rgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            lin_mask = tf.cast(im_rgb_pixels <= 0.04045, dtype=tf.float32)
            exp_mask = tf.cast(im_rgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (im_rgb_pixels / 12.92 * lin_mask) + (
                    ((im_rgb_pixels + 0.055) / 1.055) ** 2.4) * exp_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        with tf.name_scope('xyz_to_cielab'):

            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            lin_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exp_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * lin_mask + (
                    xyz_normalized_pixels ** (1 / 3)) * exp_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(im_rgb))


def lab_to_rgb(lab_img):
    with tf.name_scope('lab_to_rgb'):
        lab_img = check_image(lab_img)
        lab_img_pixels = tf.reshape(lab_img, [-1, 3])
        with tf.name_scope('cielab_to_xyz'):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_img_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)
            # convert to xyz
            epsilon = 6 / 29
            lin_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exp_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * lin_mask + (
                    fxfyfz_pixels ** 3) * exp_mask

            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope('xyz_to_srgb'):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)

            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            lin_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exp_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            rgb_im_pixels = (rgb_pixels * 12.92 * lin_mask) + (
                    (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exp_mask

        return tf.reshape(rgb_im_pixels, tf.shape(lab_img))


def get_height_width(image_path):
    img = matplotlib.image.imread(image_path)  # replace with your image file name and extension
    height, width, channels = img.shape
    return height, width

img_height = 256
img_width = 256


@tf.function
def get_im_lab(img_path, height=img_height, width=img_width):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width])

    image_lab = rgb_to_lab(img / 255.0)
    return image_lab


@tf.function
def get_l_ab_chan(img_path, height=img_height, width=img_width):
    image_lab = get_im_lab(img_path, height, width)

    image_l = tf.expand_dims(image_lab[:, :, 0], -1)
    image_ab = image_lab[:, :, 1:]

    return image_l, image_ab


def preview_lab_image(img_path, height=img_height, width=img_width):
    fig, ax = plt.subplots(1, 6, figsize=(18, 30))

    image_lab = get_im_lab(img_path, height, width)

    original_img = lab_to_rgb(image_lab)
    original_img = tf.image.convert_image_dtype(original_img, dtype=tf.uint8, saturate=True)

    ax[0].imshow(original_img)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image_lab.numpy())
    ax[1].axis('off')
    ax[1].set_title('Lab')

    ax[2].imshow(image_lab[:, :, 0].numpy(), cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('L')

    ax[3].imshow(image_lab[:, :, 1].numpy(), cmap='RdYlGn_r')
    ax[3].axis('off')
    ax[3].set_title('a')

    ax[4].imshow(image_lab[:, :, 2].numpy(), cmap='YlGnBu_r')
    ax[4].axis('off')
    ax[4].set_title('b')

    ax[5].imshow(np.concatenate((np.zeros((img_height, img_width, 1)), image_lab[:, :, 1:].numpy()), axis=2))
    ax[5].axis('off')
    ax[5].set_title('ab')

    plt.show()


def cnn_model():
    model = tf.keras.Sequential([
        # CONV 1
        Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
               input_shape=(img_height, img_width, 1)),
        Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV2
        Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV3
        Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV4
        Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV5 (padding=2)
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1, 1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV6 (padding=2)
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1, 1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV7
        Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV8
        Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        Conv2D(filters=313, kernel_size=1, strides=(1, 1), padding='valid'),

        # OUTPUT
        Conv2D(filters=2, kernel_size=1, padding='valid', dilation_rate=1, strides=(1, 1), use_bias=False),
        UpSampling2D(size=4, interpolation='bilinear'),
    ])

    # Show model summary
    model.build()
    # print(model.summary())

    return model


def plot_result(image_path, best_model):
    image_to_predict_lab = get_l_ab_chan(image_path)

    # Use only L channel (grayscale) to predict
    image_to_predict = tf.expand_dims(image_to_predict_lab[0], 0)

    # Predict
    prediction = best_model.predict(image_to_predict, verbose=1)[0]

    original_img = np.concatenate((image_to_predict_lab[0], image_to_predict_lab[1]), axis=2)
    original_img = lab_to_rgb(original_img).numpy()

    predicted_img = np.concatenate((image_to_predict[0], prediction), axis=2)
    predicted_img = lab_to_rgb(predicted_img).numpy()

    input_height, input_width, *args = plt.imread(image_path).shape

    predicted_img = resize(predicted_img, (input_height, input_width), anti_aliasing=True)
    p = image_path.split('/')
    full_name = p[-1]
    name_of_image = full_name.split('.')[:-1]
    name_of_extension = full_name.split('.')[-1]
    full_new_path = f"{'/'.join(p[:-1])}/{'.'.join(name_of_image)}-colorized.{name_of_extension}"
    matplotlib.image.imsave(full_new_path, predicted_img)
    return full_new_path

