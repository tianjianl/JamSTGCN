import os
import sys
import time
import argparse
import numpy as np

import paddle
import tqdm
import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl.utils.logger import log
from models.cands import Cs, Lp
from data_loader.data_utils import data_gen_mydata, gen_batch
from data_loader.graph import GraphFactory
from models.model import STGCNModel
from models.tester import model_inference, model_test

def main(args):
    
    input_file = 'dataset/' + args.dataset + '/input_oct.csv'
    input_prev = 'dataset/' + args.dataset + '/input_sept.csv'
    interval = int(args.dataset[0:2])
    data = data_gen_mydata(input_file, input_prev, args.n_route, args.n_his, args.n_pred, interval, args.dataset, (args.n_val, args.n_test))
    log.info(data.get_stats())
    log.info(data.get_len('train'))
   
    gf = GraphFactory(args)
    model = STGCNModel(args)
#   train_loss = fl.reduce_sum((y - label) * (y - label))
#   lr = fl.exponential_decay(
#           learning_rate=args.lr,
#           #decay_steps=5 * epoch_step,
#           decay_rate=0.7,
#           staircase=True)
#    lr = paddle.optimizer.lr.ExponentialDecay(learning_rate=args.lr, gamma=0.7, verbose=False)          
    
    lr = args.lr
    if args.opt == 'RMSProp':
        optim = paddle.optimizer.RMSProp(learning_rate = lr, parameters = model.parameters())
    elif args.opt == 'ADAM':
        optim = paddle.optimizer.Adam(learning_rate = lr, parameters = model.parameters())
     
    if args.inf_mode == 'sep':
        step_idx = args.n_pred - 1
        tmp_idx = [step_idx]
        min_val = min_va_val = np.array([4e1, 1e5, 1e5])
    elif args.inf_mode == 'merge':
        step_idx = tmp_idx = np.arange(3, args.n_pred + 1, 3) - 1
        min_val = min_va_val = np.array([4e1, 1e5, 1e5]) * len(step_idx)
    else:
        raise ValueError(f'Error: test mode "{args.inf_mode}" is not defined')
    
    step = 0
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        for idx, x_batch in enumerate(
                gen_batch(
                    data.get_data('train'),
                    args.batch_size,
                    dynamic_batch=True,
                    shuffle=True)):
           
            x = np.array(x_batch[:, 0:args.n_his, :, :], dtype = np.int32)
            x_input = np.array(x_batch[:, 0:2*args.n_his+1, :, :], dtype = np.int32)
            
            graph = gf.build_graph(x)
            graph.tensor()
           
            pred, loss = model(graph, x_input)
            loss.backward()
                
            optim.step()
            optim.clear_grad()
            if idx % 5 == 0:
                print(epoch, idx, loss)

        min_va_val, min_val = model_inference(gf, model, pred, data, args, step_idx, min_va_val, min_val)
        va, te = min_va_val, min_val
        print(f'ACC {va[0]:7.3%}, {te[0]:7.3%}; '
            f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
            f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
        if epoch % 1 == 0:
            model_test(gf, model, pred, data, args)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_route', type=int, default=330)
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=1)  #equal to num_layers
    parser.add_argument('--Kt', type=int, default=3)  #filter size 
    parser.add_argument('--stride', type=int, default=1) 
    parser.add_argument('--use_his', type=bool, default=False)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--opt', type=str, default='ADAM')
    parser.add_argument('--inf_mode', type=str, default='sep')
    parser.add_argument('--adj_mat_file', type=str, default='dataset/W.csv')
    parser.add_argument('--output_path', type=str, default='./outputs/')
    parser.add_argument('--n_val', type=int, default=4)
    parser.add_argument('--n_test', type=int, default=4)
    parser.add_argument('--graph_operation', type=str, default='GCN', help = 'graph operation in spatio block, default = GCN, possbile values:GCN, GAT, GraphSAGE')
    parser.add_argument('--act_func', type=str, default = 'GLU')
    parser.add_argument('--dataset', type=str, default='15min_mean')
    args = parser.parse_args()
    blocks = [[16, 16, 16], [16, 16, 64]]
    args.blocks = blocks
    log.info(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
