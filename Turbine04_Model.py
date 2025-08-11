
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

# ====== 2. 封装 NLARX 包装器类 ======
class NLARXModelWrapper:
    def __init__(self, keras_model,
                 input_xoffset, input_gain, input_ymin,
                 output_xoffset, output_gain, output_ymin,
                 y_init=None, u_init=None):  # 加入可选初始化数据
        self.model = keras_model

        # 保存 mapminmax 自动归一化参数
        self.input_xoffset = np.array(input_xoffset, dtype=np.float32)
        self.input_gain    = np.array(input_gain, dtype=np.float32)
        self.input_ymin    = float(input_ymin)

        self.output_xoffset = float(output_xoffset)
        self.output_gain    = float(output_gain)
        self.output_ymin    = float(output_ymin)

        self.reset()
        # 如果有初始值就初始化
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
        # 构造输入向量：[y(t-1) ~ y(t-10), u(t-1) ~ u(t-5)]
        input_vec = np.array(self.y_hist + self.u_hist, dtype=np.float32)

        # 归一化
        input_norm = self.mapminmax_apply(input_vec).reshape(1, -1)

        # 模型预测（归一化输出）
        y_norm = self.model.predict(input_norm, verbose=0)[0, 0]

        # 反归一化
        y_pred = self.mapminmax_reverse(y_norm)

        # 更新历史记录
        self.y_hist = [y_pred] + self.y_hist[:-1]
        self.u_hist = [u_t] + self.u_hist[:-1]

        return y_pred

    def __call__(self, u_t):
        return self.step(u_t)

    def initialize_with_real_data(self, y_list, u_list):
        assert len(y_list) == 10 and len(u_list) == 5
        self.y_hist = list(y_list)[::-1]
        self.u_hist = list(u_list)[::-1]


# 替换为你的真实归一化参数值（15 个输入参数 + 1 个输出参数）
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
# ====== 2. 初始化模型包装器 ======
turbine04_init_y = [3564, 3564, 3564, 3564, 3564,
                    3564, 3564, 3563, 3563, 3563]  # 初始输出值
turbine04_init_y = np.array(turbine04_init_y, dtype=np.float32)
turbine04_init_u = [0, 0, 0, 0, 0]  # 初始输入值
turbine04_init_u = np.array(turbine04_init_u, dtype=np.float32)

# 手动归一化处理
turbine04_init_y = (turbine04_init_y - y_min) / (y_max - y_min + np.finfo(np.float32).eps)
turbine04_init_u = (turbine04_init_u - u_min) / (u_max - u_min + np.finfo(np.float32).eps)
# 创建包装类
turbine04_dynamics = NLARXModelWrapper(model,
    input_xoffset, input_gain, input_ymin,
    output_xoffset, output_gain, output_ymin,
        y_init=turbine04_init_y,
    u_init=turbine04_init_u  # 初始值可以根据需要调整
)

turbine04_dynamics.reset()
# 在调用时记得输入归一化 输出反归一化


# # ====== 3. 加载数据集 ======
# df = pd.read_csv('Turbine04_Data01_final.csv')  # 替换为你的真实数据集路径

# # 提取输入 u(t) 和输出 y(t)
# u_series = df['V4ECU1_Throttle'].to_numpy(dtype=np.float32)
# y_series = df['V4ECU1_RPM'].to_numpy(dtype=np.float32)

# # ====== 4. 裁剪数据集 ======
# u_series = u_series[:52334]
# y_series = y_series[:52334]
# # 手动归一化处理
# u_series_norm = (u_series - u_min) / (u_max - u_min + np.finfo(np.float32).eps)
# y_series_norm = (y_series - y_min) / (y_max - y_min + np.finfo(np.float32).eps)
# # ====== 5. 初始化模型（使用真实历史数据）======
# turbine03_dynamics.initialize_with_real_data(y_series_norm[:10], u_series_norm[5:10])

# # ====== 6. 逐步预测 ======
# y_preds = []

# bar = tqdm(range(10, len(u_series_norm)), desc="Turbine Simulation", dynamic_ncols=True)
# for i in bar:
#     u_t = u_series_norm[i]  # 使用归一化后的油门信号
#     y_pred = turbine03_dynamics(u_t)  # 进行一步预测
#     y_pred = y_pred * (y_max - y_min + np.finfo(np.float32).eps) + y_min # 反归一化预测结果
#     bar.set_postfix({"Speed": f"{y_pred:.1f}", "Throttle": f"{u_series[i]:.1f}"})
#     y_preds.append(y_pred)

# # ====== 7. 画图比较 ======
# plt.figure(figsize=(10, 4))
# plt.plot(y_series[10:], label='True RPM')
# plt.plot(y_preds, label='Predicted RPM')
# plt.xlabel('Sample Index')
# plt.ylabel('Engine Speed')
# plt.legend()
# plt.title('Turbine03 Prediction vs True Output')
# plt.grid(True)
# plt.show()


# === 小注释 ===
# 由于python中的逐步预测和matlab中的sim/compare命令有些不同
# 这里的预测是从头开始的，前5个点是初始化的历史值。而Matlab的命令会直接传入这几个值初始化
# 这可能导致预测结果有些偏差，尤其是最开始。但这跟我们simulink中的效果很像，最开始也是大起然后回落平稳
# 如果需要更精确的预测，可以在初始化时传入真实数据 或者先运行5-10s，收集足够的历史数据再进行预测
# 当然，在真实情况中，我推荐使用第二种方法，先以10油门行5-10s，最后把这段时间裁剪掉即可
# 在使用数据集拟合时则选择第一种方法，传入五个真实点 这会让拟合变得非常完美