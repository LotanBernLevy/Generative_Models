from datetime import datetime
import numpy as np
import torch
import lightning.pytorch as pl
import math

DEBUG_MESSAGES = True

dprint = lambda toprint: print(toprint) if(DEBUG_MESSAGES) else None




def get_unique_vecs(sampler_func, vector_length, data_size):
    data_str_array = []
    data_array = []

    while len(data_array) < data_size:
        new_vec = sampler_func(vector_length)
        str_vec = str(list(new_vec))
        if str_vec not in data_str_array:
            data_str_array.append(str_vec)
            data_array.append(new_vec)
    return np.array(data_array)


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((x.shape[0], *self.shape))


def conv2d_output_size(N, C_in, H_in, W_in, C_out, kernel_size, padding=(0,0), dilation=(1,1), stride=(1,1)):
    H_out = int(math.floor((H_in + 2 * padding[0] - dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1))
    W_out = int(math.floor((W_in + 2 * padding[1] - dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1))
    return N, C_out, H_out, W_out

def convTranspose2d_output_size(N, C_in, H_in, W_in, C_out, kernel_size, padding=(0,0), dilation=(1,1), stride=(1,1)):
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size [0] - 1) + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size [1] - 1) + 1
    return N, C_out, H_out, W_out

      