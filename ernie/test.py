import numpy as np
import paddle
from paddle.distributed import init_parallel_env

paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()
tensor_list = []
if paddle.distributed.ParallelEnv().local_rank == 0:
    np_data1 = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
    data1 = paddle.to_tensor(np_data1)
    data2 = paddle.to_tensor(np_data2)
    paddle.distributed.all_gather(tensor_list, data1)
    print(tensor_list)
else:
    np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
    np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
    data1 = paddle.to_tensor(np_data1)
    data2 = paddle.to_tensor(np_data2)
    paddle.distributed.all_gather(tensor_list, data2)