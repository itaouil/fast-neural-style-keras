import nets

# Python modules
import time
import argparse
import numpy as np

# Scipy modules
from scipy import ndimage
from scipy.misc import imsave

# Keras modules
from keras import backend as K
from keras.layers import Input, merge
from keras.callbacks import TensorBoard
from keras.models import Model,Sequential
from keras.optimizers import Adam, SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer


def display_img(i,x,style,is_val=False):
    # save current generated image
    img = x #deprocess_image(x)
    if is_val:
        #img = ndimage.median_filter(img, 3)

        fname = 'images/output/%s_%d_val.png' % (style,i)
    else:
        fname = 'images/output/%s_%d.png' % (style,i)
    imsave(fname, img)
    print('Image saved as', fname)


def get_style_img_path(style):
    """
        Relative path of where
        the style image to train
        on is stored.
    """
    return f"images/style/{style}.jpg"


def main(args):
    """
        Main.
    """
    # Extract CLA values
    style = args.style
    tv_weight = args.tv_weight
    style_weight = args.style_weight
    content_weight = args.content_weight
    img_width = img_height =  args.image_size

    # Get relative path of style image
    style_image_path = get_style_img_path(style)

    # Create image transform network model
    net = nets.image_transform_net(img_width, img_height, tv_weight)
    model = nets.loss_net(net.output,
                          net.input,
                          img_width,
                          img_height,
                          style_image_path,
                          content_weight,
                          style_weight)
    model.summary()

    # Epochs
    nb_epoch = 82785 * 2
    train_batchsize =  1
    train_image_path = "images/train/"

    learning_rate = 1e-3 #1e-3
    optimizer = Adam() # Adam(lr=learning_rate,beta_1=0.99)

    model.compile(optimizer,  dummy_loss)  # Dummy loss since we are learning from regularizes

    datagen = ImageDataGenerator()

    dummy_y = np.zeros((train_batchsize, img_width, img_height, 3)) # Dummy output, not used since we use regularizers to train

    #model.load_weights(style+'_weights.h5',by_name=False)

    skip_to = 0

    i=0
    t1 = time.time()
    for x in datagen.flow_from_directory(train_image_path, class_mode=None, batch_size=train_batchsize,
        target_size=(img_width, img_height), shuffle=False):
        if i > nb_epoch:
            break

        if i < skip_to:
            i += train_batchsize
            if i % 1000 ==0:
                print("skip to: %d" % i)

            continue


        hist = model.train_on_batch(x, dummy_y)

        if i % 50 == 0:
            print(hist,(time.time() -t1))
            t1 = time.time()

        if i % 500 == 0:
            print("epoc: ", i)
            val_x = net.predict(x)

            display_img(i, x[0], style)
            display_img(i, val_x[0],style, True)
            model.save_weights(style+'_weights.h5')

        i+=train_batchsize


if __name__ == "__main__":
    """
        Calls main function
        with CLA.
    """
    # CLA description
    parser = argparse.ArgumentParser(description='Real-time style transfer')

    # Style to train with
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')

    # Output name for training result
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file path without extension')

    # Total Variation weight
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')

    # Content weight
    parser.add_argument('--content_weight', default=1.0, type=float)

    # Style weight
    parser.add_argument('--style_weight', default=4.0, type=float)

    # Image size to be expected
    parser.add_argument('--image_size', default=256, type=int)

    # Parse arguments
    args = parser.parse_args()

    # Call main with CLA
    main(args)
