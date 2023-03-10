import os

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
from mxnet import init
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
import logging
logging.basicConfig()
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

# Helpers
class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNet."""

    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def forward(self, x):
        return np.clip(x, 0, 6)


# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(RELU6() if relu6 else nn.Activation('relu'))


def _add_conv_dw(out, dw_channels, channels, stride, relu6=False):
    print(int(dw_channels/16))
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride, pad=1, num_group=int(dw_channels/16), relu6=relu6)
    #_add_conv(out, channels=dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels, relu6=relu6)

    _add_conv(out, channels=channels,relu6=relu6)


class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
    Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    """

    def __init__(self, in_channels, channels, t, stride, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        self.out = nn.HybridSequential()

        _add_conv(self.out, in_channels * t, relu6=True)
        _add_conv(self.out, in_channels * t, kernel=3, stride=stride,pad=1, num_group=in_channels *t, relu6=True)
        _add_conv(self.out, channels, active=False, relu6=True)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out = np.add(out, x)
        return out

class mx_MobileNet(nn.HybridBlock):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(mx_MobileNet, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        _add_conv(self.features, channels=int(32 * multiplier), kernel=3, pad=1, stride=2)
        dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024]]
        channels = [int(x * multiplier) for x in [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        for dwc, c, s in zip(dw_channels, channels, strides):
            _add_conv_dw(self.features, dw_channels=dwc, channels=c, stride=s)
        self.features.add(nn.GlobalAvgPool2D())
        self.features.add(nn.Flatten())

        self.output = nn.Dense(classes)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def train_model(model,file_name="mx_mobilenetG",mod=1):
    # MNIST images are 28x28. Total pixels in input layer is 28x28 = 784
    num_inputs = 1024
    # Clasify the images into one of the 10 digits
    num_outputs = 10
    # 64 images in a batch
    batch_size = 128
    
    transform_train = transforms.Compose([
    # Randomly crop an area and resize it to be 32x32, then pad it to be 40x40
        gcv_transforms.RandomCrop(32, pad=4),
    # Randomly flip the image horizontally
        transforms.RandomFlipLeftRight(),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
        transforms.ToTensor(),
    # Normalize the image with mean and standard deviation calculated across all images
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_data = gluon.data.DataLoader(
                gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
                    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=1)
    # Initialize the parameters with Xavier initializer
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    #model.hybridize()
    # Use cross entropy loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    # Use Adam optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': .001})

    # Train for one epoch
    for epoch in range(1):
        # Iterate through the images and labels in the training data
        for batch_num, (data, label) in enumerate(train_data):
            # get the images and labels
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Ask autograd to record the forward pass
            with mx.autograd.record():
                # Run the forward pass
                output = model(data)
                # Compute the loss
                loss = softmax_cross_entropy(output, label)
            # Compute gradients
            loss.backward()
            # Update parameters
            trainer.step(data.shape[0])

            # Print loss once in a while
            if batch_num % 50 == 0:
                curr_loss = nd.mean(loss).asscalar()
                print("Epoch: %d; Batch %d; Loss %f" % (epoch, batch_num, curr_loss))

    #model.forward()
    if mod == 2:
        model.export("mx_mobilenetG", epoch=1)
    elif mod == 1:
        model.save_parameters('./param/' + file_name +'.params')
    
    return load_mobilenet(file_name=file_name,mod=1)


def load_mobilenet(class_num=1000, width_multi=1,pretrained = True,file_name="mx_mobilenetG",mod=1):
    if mod == 1:
        net = mx_MobileNet(1,1000, prefix="")
        net.hybridize()
        net.load_parameters('./param/'+ file_name+".params", ctx=mx.cpu())
    else:
        net = gluon.nn.SymbolBlock.imports(file_name, ['data'], "mx_mobilenetG-0001.params", ctx=mx.cpu())
    return net


model = mx_MobileNet(1,1000, prefix="")
model_ = train_model(model,mod=1)

