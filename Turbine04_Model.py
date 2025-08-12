
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.utils import register_keras_serializable # type: ignore
from tqdm import tqdm

@register_keras_serializable()
def radbas(x):
    return K.exp(-K.square(x))

model = load_model('H:\Vehicles_Simulation\work_space\gymnasium_env\envs\\turbine04_model.h5', custom_objects={'radbas': radbas})


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
        self.y_hist = [0.0] * 10  # y(t-1) ~ y(t-10)
        self.u_hist = [0.0] * 5   # u(t-1) ~ u(t-5)

    def mapminmax_apply(self, x):
        return (x - self.input_xoffset) * self.input_gain + self.input_ymin

    def mapminmax_reverse(self, y_norm):
        return (y_norm - self.output_ymin) / self.output_gain + self.output_xoffset

    def step(self, u_t):

        input_vec = np.array(self.y_hist + self.u_hist, dtype=np.float32)


        input_norm = self.mapminmax_apply(input_vec).reshape(1, -1)


        y_norm = self.model.predict(input_norm, verbose=0)[0, 0]


        y_pred = self.mapminmax_reverse(y_norm)


        self.y_hist = [y_pred] + self.y_hist[:-1]
        self.u_hist = [u_t] + self.u_hist[:-1]

        return y_pred

    def __call__(self, u_t):
        return self.step(u_t)

    def initialize_with_real_data(self, y_list, u_list):
        assert len(y_list) == 10 and len(u_list) == 5
        self.y_hist = list(y_list)[::-1]
        self.u_hist = list(u_list)[::-1]



input_xoffset = [0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]  
input_gain    = [2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2]  
input_ymin    = -1

output_xoffset = 0
output_gain    = 2
output_ymin    = -1

u_min = 0
u_max = 100
y_min = 3528
y_max = 9715

turbine04_init_y = [3564, 3564, 3564, 3564, 3564,
                    3564, 3564, 3563, 3563, 3563]  
turbine04_init_y = np.array(turbine04_init_y, dtype=np.float32)
turbine04_init_u = [0, 0, 0, 0, 0]  
turbine04_init_u = np.array(turbine04_init_u, dtype=np.float32)


turbine04_init_y = (turbine04_init_y - y_min) / (y_max - y_min + np.finfo(np.float32).eps)
turbine04_init_u = (turbine04_init_u - u_min) / (u_max - u_min + np.finfo(np.float32).eps)

turbine04_dynamics = NLARXModelWrapper(model,
    input_xoffset, input_gain, input_ymin,
    output_xoffset, output_gain, output_ymin,
        y_init=turbine04_init_y,
    u_init=turbine04_init_u  
)

turbine04_dynamics.reset()
