from keras.layers import Layer
from keras import initializers,regularizers
from keras import backend as K

class Mil_Attention(Layer):
    def __init__(self,L_dim,output_dim,kernel_initializer='glorot_uniform',kernel_regularizer=None,use_bias=True,use_gated=False,**kwargs):
        self.L_dim=L_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.use_gated=use_gated
        
        self.v_init=initializers.get(kernel_initializer)
        self.w_init=initializers.get(kernel_initializer)
        self.u_init=initializers.get(kernel_initializer)
        
        self.v_regularizer=regularizers.get(kernel_regularizer)
        self.w_regularizer=regularizers.get(kernel_regularizer)
        self.u_regularizer=regularizers.get(kernel_regularizer)
        
        super(Mil_Attention,self).__init__(**kwargs)
        
    def build(self,input_shape):
        assert len(input_shape) == 2
        input_dim=input_shape[1]
        #create a trainable wight for the layer
        #self.V=self.add_weight(shape=(input_dim,self.L_dim),initializer=self.v_init,name='v',regularizer=self.v_regularizer,trainable=True)
        #self.W=self.add_weight(shape=(self.L_dim,1),initializer=self.w_init,name='w',regularizer=self.w_regularizer,trainable=True)
        
        self.V = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)


        self.W = self.add_weight(shape=(self.L_dim, self.output_dim),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)
        
        if self.use_gated:
            self.U=self.add_weight(shape=(input_dim,self.L_dim),initializer=self.u_init,name='u',regularizer=self.u_regularizer,trainable=True)
        else:
            self.U=None
        self.input_built=True
        #super(Mil_Attention, self).build(input_shape)
        
    def call(self,x,mask=None):
        n,d=x.shape
        ori_x=x
        x=K.tanh(K.dot(x,self.V))
        if self.use_gated:
            gate_x=K.sigmoid(K.dot(ori_x,self.U))
            ac_x=x*gate_x
        else:
            ac_x=x
        soft_x=K.dot(ac_x,self.W)
        alpha=K.softmax(K.transpose(soft_x))
        alpha=K.transpose(alpha)
        return alpha
        
    def compute_output_shape(self,input_shape):
        shape=list(input_shape)
        assert len(shape) == 2
        shape[1]=self.output_dim
        return tuple(shape)
        
    
    def get_config(self):
        config={
            'output_dim':self.output_dim,
            'L_dim':self.L_dim,
            #'v_initializer':initializers.serialize(self.v_init),
            #'w_initializer':initializers.serialize(self.w_init),
            #'v_regularizer':regularizers.serialize(self.v_regularizer),
            #'w_regularizer':regularizers.serialize(self.w_regularizer),
            #'use_bias':self.use_bias
        }
        base_config=super(Mil_Attention,self).get_config()
        return dict(list(base_config.items())+list(config.items()))
        
        
class Bag_Pooling(Layer):
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Bag_Pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True
        #super(Bag_Pooling, self).build(input_shape)

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.sigmoid(x)


        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    
    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Bag_Pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
