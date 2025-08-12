
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.utils import register_keras_serializable # type: ignore

@register_keras_serializable()
def radbas(x):
    return K.exp(-K.square(x))

model = load_model('H:\Vehicles_Simulation\work_space\gymnasium_env\envs\\turbine01_model.h5', custom_objects={'radbas': radbas})

class NLARXModelWrapper:
    def __init__(self, keras_model,
                 input_xoffset, input_gain, input_ymin,
                 output_xoffset, output_gain, output_ymin,
                  y_init=None, u_init=None):  
        self.model = keras_model


        self.input_xoffset = np.array(input_xoffset, dtype=np.float32)
        self.input_gain    = np.array(input_gain, dtype=np.float32)
        self.input_ymin    = float(input_ymin)

        self.output_xoffset = float(output_xoffset)
        self.output_gain    = float(output_gain)
        self.output_ymin    = float(output_ymin)
            
        self.reset()

        if y_init is not None and u_init is not None:
            self.initialize_with_real_data(y_init, u_init)

    def reset(self):
        self.y_hist = [0.0] * 5  # y(t-1) ~ y(t-5)
        self.u_last = 0.0        # u(t-1)

    def mapminmax_apply(self, x):
        return (x - self.input_xoffset) * self.input_gain + self.input_ymin

    def mapminmax_reverse(self, y_norm):
        return (y_norm - self.output_ymin) / self.output_gain + self.output_xoffset

    def step(self, u_t):

        input_vec = np.array(self.y_hist + [self.u_last], dtype=np.float32)


        input_norm = self.mapminmax_apply(input_vec).reshape(1, -1)


        y_norm = self.model.predict(input_norm, verbose=0)[0, 0]


        y_pred = self.mapminmax_reverse(y_norm)


        self.y_hist = [y_pred] + self.y_hist[:-1]
        self.u_last = u_t

        return y_pred
    
    def __call__(self, u_t):
        return self.step(u_t)
    

    def initialize_with_real_data(self, y_list, u_last):
        self.y_hist = list(y_list)[::-1]  # y(t-1)~y(t-5)
        self.u_last = u_last




input_xoffset = [3251, 3251, 3251, 3251, 3251, 0]  
input_gain    = [0.000312842171124668, 0.000312842171124668, 0.000312842171124668,
                 0.000312842171124668, 0.000312842171124668, 0.020000000000000]
input_ymin    = -1

output_xoffset = 3251
output_gain    = 3.128421711246676e-04
output_ymin    = -1



turbine01_init_y = [3276, 3276, 3276, 3275, 3275]  
turbine01_init_u = 0  

turbine01_dynamics = NLARXModelWrapper(model,
                                   input_xoffset, input_gain, input_ymin,
                                   output_xoffset, output_gain, output_ymin,
                                   y_init=turbine01_init_y,
                                   u_init=turbine01_init_u)  
turbine01_dynamics.reset()

