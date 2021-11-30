from abstractions import ModelBuilderBase

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


class UNetBaselineBuilder(ModelBuilderBase):
    """Unet V2 implementation from CAMUS paper.

    from the paper:

        - model input: (256, 256, 1)
        - density normalization after resizing
        - no augmentation
        - weight init: glorot_uniform
        - max_pool: (2, 2)
        - lowest resolution: (16, 16)
        - activation: relu
        - final activation: softmax (we used sigmoid because of binary output)
        - optimizer: Adam(lr=1e-4)
        - loss: crossentropy and weight decay(L2 regularization of the weights). we will not apply weight decay, we will
          try regularization if the model over-fits
        - epochs: 30
        - params: ~18M
        - batch size: 10

    """

    def get_compiled_model(self):
        model = self._get_model_graph()
        optimizer = self._get_optimizer()
        metrics = self._get_metrics()
        loss = self._get_loss()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def __init__(self, config):
        super().__init__(config)

        self.conv_kernel_size = (3, 3)
        self.conv_padding = 'same'

        self.conv_trans_kernel_size = (2, 2)
        self.conv_trans_strides = (2, 2)
        self.conv_trans_padding = 'same'

        self.max_pool_size = (2, 2)
        self.max_pool_strides = (2, 2)

        self.activation = 'relu'
        self.final_activation = 'sigmoid'

        self.kernel_initializer = 'glorot_uniform'

    def _load_params(self, config):
        self.optimizer_type = config.model_builder.optimizer.type
        self.learning_rate = config.model_builder.optimizer.initial_lr
        self.loss_type = config.model_builder.loss_type
        self.metrics = config.model_builder.metrics
        self.input_h = config.input_height
        self.input_w = config.input_width
        self.n_chanels = config.n_channels
        self.inference_threshold = config.model_builder.inference_threshold

    def _set_defaults(self):
        self.optimizer_type = 'adam'
        self.learning_rate = 0.001
        self.loss_type = 'binary_crossentropy'
        self.metrics = ['acc']
        self.input_h = 128
        self.input_w = 128
        self.n_channels = 1
        self.inference_threshold = 0.5

    def _get_model_graph(self):
        input_tensor = tfk.Input((self.input_h, self.input_w, self.n_channels), name='input_tensor')

        # Encoder
        connection1, x = self._encoder_block(48, 1)(input_tensor)
        connection2, x = self._encoder_block(96, 2)(x)
        connection3, x = self._encoder_block(192, 3)(x)
        connection4, x = self._encoder_block(384, 4)(x)

        # Middle
        x = self._conv2d_bn_relu(768, 'middle_block1')(x)
        x = self._conv2d_bn_relu(768, 'middle_block2')(x)

        # Decoder
        x = self._decoder_transpose_block(384, 384, 1, connection=connection4)(x)
        x = self._decoder_transpose_block(192, 192, 2, connection=connection3)(x)
        x = self._decoder_transpose_block(96, 96, 3, connection=connection2)(x)
        x = self._decoder_transpose_block(48, 48, 4, connection=connection1)(x)

        # Output
        n_classes = 1
        x = tfkl.Conv2D(filters=n_classes,
                        kernel_size=(1, 1),
                        padding='same',
                        use_bias=True,
                        kernel_initializer=self.kernel_initializer,
                        name='final_conv')(x)
        x = tfkl.Activation(self.final_activation, name='output_tensor')(x)

        model = tfk.Model(input_tensor, x)
        return model

    @staticmethod
    def _get_metrics():
        metrics = ['acc']
        return metrics

    def _get_optimizer(self):
        return tfk.optimizers.Adam(learning_rate=self.learning_rate)

    @staticmethod
    def _get_loss():
        return 'binary_crossentropy'

    def _conv2d_bn_relu(self, filters, block_name):
        """Extension of Conv2D layer with BatchNormalization and ReLU activation"""

        conv_name = block_name + '_conv'
        act_name = block_name + '_' + self.activation
        bn_name = block_name + '_bn'

        def wrapper(input_tensor):
            x = tfkl.Conv2D(
                filters=filters,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding,
                activation=None,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                name=conv_name,
            )(input_tensor)

            x = tfkl.BatchNormalization(name=bn_name)(x)
            x = tfkl.Activation(self.activation, name=act_name)(x)

            return x

        return wrapper

    def _convtranspose_bn_relu(self, filters, block_name):
        """Extension of Conv2DTranspose layer with BatchNormalization and ReLU activation"""

        transconv_name = block_name + '_conv2dtrans'
        act_name = block_name + '_' + self.activation
        bn_name = block_name + '_bn'

        def wrapper(input_tensor):
            x = tfkl.Conv2DTranspose(
                filters,
                kernel_size=self.conv_trans_kernel_size,
                strides=self.conv_trans_strides,
                padding=self.conv_trans_padding,
                activation=None,
                name=transconv_name,
                use_bias=False,
                kernel_initializer=self.kernel_initializer)(input_tensor)

            x = tfkl.BatchNormalization(name=bn_name)(x)
            x = tfkl.Activation(self.activation, name=act_name)(x)

            return x

        return wrapper

    def _decoder_transpose_block(self, transpose_filters, conv_filters, block, connection=None):
        """Generates a decoder block.

        ---------- convtranse_bn_relu + concat + conv2d_bn_relu + conv2d_bn_relu -----------
        """

        transpose_name = f'decoder_block{block}a'

        conv1_name = f'decoder_block{block}b'
        conv2_name = f'decoder_block{block}c'

        concat_name = f'decoder_block{block}_concat'

        def wrapper(input_tensor):
            x = self._convtranspose_bn_relu(transpose_filters, transpose_name)(input_tensor)
            if connection is not None:
                x = tfkl.Concatenate(axis=3, name=concat_name)([x, connection])

            x = self._conv2d_bn_relu(conv_filters, conv1_name)(x)
            x = self._conv2d_bn_relu(conv_filters, conv2_name)(x)

            return x

        return wrapper

    def _encoder_block(self, filters, block):
        """Generates an encoder block.

        ---------- conv2d_bn_relu + conv2d_bn_relu + max-pooling2d -----------
        """

        conv1_name = f'encoder_block{block}a'
        conv2_name = f'encoder_block{block}b'
        max_pool_name = f'encoder_block{block}_mp'

        def wrapper(input_tensor):
            x = self._conv2d_bn_relu(filters, conv1_name)(input_tensor)
            connection = self._conv2d_bn_relu(filters, conv2_name)(x)
            x = tfkl.MaxPool2D(self.max_pool_size, strides=self.max_pool_strides, name=max_pool_name)(connection)

            return connection, x

        return wrapper
