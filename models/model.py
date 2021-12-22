import numpy as np
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl.nn as p


class STGCNModel(paddle.nn.Layer):
    """Implementation of Spatio-Temporal Graph Convolutional Networks"""
    def __init__(self, args):
        super(STGCNModel, self).__init__()
        self.args = args
        blocks = self.args.blocks
        if self.args.act_func == 'GLU':
            self.multiplier = 2
        else:
            self.multiplier = 1
        #inputshape [B, T, n, C]
        
        #st-block-1
        name = 'st-block-1'
        self.emb = nn.Embedding(5, blocks[0][0], weight_attr = paddle.ParamAttr(trainable = True))  
        self.conv2d1_align = nn.Conv2D(blocks[0][0], blocks[0][1], 1, padding = "SAME", weight_attr = paddle.ParamAttr(name = name + 'conv2d1', trainable = True), data_format = 'NHWC')               
        self.conv2d1_1 = nn.Conv2D(blocks[0][0], blocks[0][1]*self.multiplier, self.args.Kt, weight_attr = paddle.ParamAttr(name = name + 'conv2d1_1', trainable = True), 
                            bias_attr = paddle.ParamAttr(trainable = True), data_format = 'NHWC', padding = "SAME")
        
        if self.args.graph_operation == 'GCN':
            self.graphconv1 = p.GCNConv(blocks[0][1], blocks[0][1])
            self.graphconvw = p.GCNConv(blocks[0][0], blocks[0][1])
        elif self.args.graph_operation == 'GAT':
            self.graphconv1 = p.GATConv(blocks[0][1], blocks[0][1])
            self.graphconvw = p.GATConv(blocks[0][0], blocks[0][1])
        elif self.args.graph_operation == 'GraphSAGE':
            self.graphconv1 = p.GraphSageConv(blocks[0][1], blocks[0][1], "max")
            self.graphconvw = p.GraphSageConv(blocks[0][0], blocks[0][1], "mean")
        self.conv2d1_2 = nn.Conv2D(blocks[0][1], blocks[0][2], self.args.Kt, weight_attr = paddle.ParamAttr(name = name + 'conv2d1_2', trainable = True),
                            bias_attr = paddle.ParamAttr(trainable = True), data_format = 'NHWC', padding = "SAME")
        #shape = [-1, T, N, blocks[0][2]]
        self.ln1 = nn.LayerNorm((args.n_his, args.n_route, blocks[0][2]), weight_attr = paddle.ParamAttr(name = name + 'ln1', trainable = True),
                            bias_attr = paddle.ParamAttr(trainable = True))

        #shape = [-1, T, N, blocks[0][2] = blocks[1][0]]
        #st-block-2
        name = 'st-block-2'
        self.conv2d2_align = nn.Conv2D(blocks[0][2], blocks[1][1], 1, padding = "SAME", weight_attr = paddle.ParamAttr(name = name + 'conv2d2', trainable = True), data_format = 'NHWC')
        self.conv2d2_1 = nn.Conv2D(blocks[0][2], blocks[1][1]*self.multiplier, self.args.Kt, weight_attr = paddle.ParamAttr(name = name + 'conv2d2_1', trainable = True), 
                            bias_attr = paddle.ParamAttr(trainable = True), data_format = 'NHWC', padding = "SAME", stride = self.args.stride)
        
        if self.args.graph_operation == 'GCN':
            self.graphconv2 = p.GCNConv(blocks[1][1], blocks[1][1])
        elif self.args.graph_operation == 'GAT':
            self.graphconv2 = p.GATConv(blocks[1][1], blocks[1][1])
        elif self.args.graph_operation == 'GraphSAGE':
            self.graphconv2 = p.GraphSageConv(blocks[1][1], blocks[1][1], "max")
        
        self.conv2d2_2 = nn.Conv2D(blocks[1][1], blocks[1][2], self.args.Kt, weight_attr = paddle.ParamAttr(name = name + 'conv2d2_2', trainable = True),
                            bias_attr = paddle.ParamAttr(trainable = True), data_format = 'NHWC', padding = "SAME", stride = self.args.stride)
        #shape = [-1, T, N, blocks[1][2]]
        self.ln2 = nn.LayerNorm((args.n_his, args.n_route, blocks[1][2]), weight_attr = paddle.ParamAttr(name = name + 'ln1', trainable = True),
                            bias_attr = paddle.ParamAttr(trainable = True))
        #output layer
        name = 'outputlayer'
        outdim = blocks[1][2] + blocks[0][1]
        self.conv2do_1 = nn.Conv2D(outdim, outdim*self.multiplier, self.args.n_his, weight_attr = paddle.ParamAttr(name = name + 'conv2d', trainable = True), data_format = 'NHWC', padding = "SAME", bias_attr = paddle.ParamAttr(trainable = True))
        self.conv2do_2 = nn.Conv2D(outdim, outdim, 1, weight_attr = paddle.ParamAttr(name = name + 'conv2d2', trainable = True), data_format = 'NHWC', bias_attr = paddle.ParamAttr(trainable = True)) 
        self.ln_o = nn.LayerNorm((self.args.n_his, args.n_route, outdim), weight_attr = paddle.ParamAttr(name = name + 'ln', trainable = True), bias_attr = paddle.ParamAttr(trainable = True))
        self.fc = nn.Linear(outdim, 5, weight_attr = paddle.ParamAttr(name = name + 'linear', trainable = True), bias_attr = paddle.ParamAttr(trainable = True))
        self.mapfc = nn.Linear(self.args.n_his * 5, 5, weight_attr = paddle.ParamAttr(name = 'map_w'), bias_attr = paddle.ParamAttr(name = 'map_b'))

    def forward(self, graph, x_input):
        """forward"""
        x_input = paddle.to_tensor(x_input)
        
        w = x_input[:, 0:12, :, :]
        x = x_input[:, 12:self.args.n_his+12, :, :]
        
        #shape = [B, T, N, 1]
        #emb dim = self.args.blocks[0][0]
        x = self.emb(x)
        w = self.emb(w)
        #two st conv blocks
        x = self.st_conv_block(
                graph,
                x,
                self.args.keep_prob,
                1,
                act_func = self.args.act_func)
        x = self.st_conv_block(
                graph,
                x,
                self.args.keep_prob,
                2,
                act_func = self.args.act_func)

        w = self.spatio_conv_layer(graph, 0, w)
        
        # output layer
        if self.args.n_his > 1:
          y_pred = self.output_block(x, w)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, \
                    but received "{n_his}".')

        y = x_input[:, self.args.n_his:self.args.n_his + 1, :, :]
        
        y = paddle.to_tensor(y) # B 1 N 1
        y = paddle.cast(y, 'int64')

        #shape pf y_pred [B, T, N, C_out]
        #y_pred = y_pred[:, 0:1, :, :] 
       
        y_pred = paddle.transpose(y_pred, [0, 2, 1, 3]) # B N T C
        
        y_pred = paddle.reshape(y_pred, [y_pred.shape[0], self.args.n_route, -1]) #B N T*C
        y_pred = self.mapfc(y_pred) #B N C
        y_pred = nn.functional.relu(y_pred)
        y = paddle.flatten(y, start_axis = 1, stop_axis = 2)
        #shape of y = [B, N, 1]
        loss, pred_softmax = nn.functional.softmax_with_cross_entropy(
                logits=y_pred, label=y, return_softmax=True)
        loss = paddle.mean(loss, axis = 1)
        single_pred = paddle.argmax(y_pred, axis = -1)
        return single_pred, loss

    def getconv(self, x, y):
        if x == 1 and y == 0:
            return self.conv2d1_align
        elif x == 2 and y == 0:
            return self.conv2d2_align
        elif x == 1 and y == 1:
            return self.conv2d1_1
        elif x == 1 and y == 2:
            return self.conv2d1_2
        elif x == 2 and y == 1:
            return self.conv2d2_1
        elif x == 2 and y == 2:
            return self.conv2d2_2
        elif x == 3 and y == 1:
            return self.conv2do_1
        elif x == 3 and y == 2:
            return self.conv2do_2
    
    def st_conv_block(self,
                      graph, 
                      x,
                      keep_prob,
                      num,
                      act_func='GLU'):
        """Spatio-Temporal convolution block"""
        
        x_s = self.temporal_conv_layer(x, num, 1, act_func=act_func)
        x_t = self.spatio_conv_layer(graph, num, x_s)
        x_o = self.temporal_conv_layer(x_t, num, 2,  act_func = act_func)
        
        if num == 1:
            x_ln = self.ln1(x_o)
        elif num == 2:
            x_ln = self.ln2(x_o)
        else:
            raise ValueError(f'ERROR: not enough layers')
        
        return nn.Dropout(p=1.0 - keep_prob)(x_ln)

    def temporal_conv_layer(self, x, num_st, num_temp, act_func='GLU'):
        """Temporal convolution layer"""
        if len(x.shape) == 5:
            x = fl.reshape(x,[x.shape[0], x.shape[1], x.shape[2], x.shape[4]])
        _, T, n, c_in = x.shape
        """padding to the same length"""
        
        if num_st != 3:
            c_out = self.args.blocks[num_st - 1][num_temp]
        else:
            c_out = self.args.blocks[1][2] + self.args.blocks[0][1]
        
        if c_in > c_out:
            conv = self.getconv(num_st, 0)
            x_input = conv(x)
        
        elif c_in < c_out:
            pad = fl.fill_constant_batch_size_like(
                input=x,
                shape=[-1, T, n, c_out - c_in],
                dtype="float32",
                value=0.0)
            x_input = fl.concat([x, pad], axis=3)
        else:
            x_input = x
        
        """GLU"""
        
        conv = self.getconv(num_st, num_temp)
        if act_func == 'GLU':
            x_conv = conv(x)
            return (x_conv[:, :, :, 0:c_out] + x_input)*fl.sigmoid(x_conv[:, :, :, -c_out:])
        else:
            x_conv = conv(x)
            if act_func == "linear":
                return x_conv
            elif act_func == "sigmoid":
                return nn.Sigmoid()(x)
            elif act_func == "relu":
                return nn.ReLU()(x_conv + x_input)
            else:
                raise ValueError(
                    f'ERROR: activation function "{act_func}" is not defined.')

    def spatio_conv_layer(self, graph, num, x):
        """Spatio convolution layer"""
        if len(x.shape) == 5:
            x = fl.reshape(x,[x.shape[0], x.shape[1], x.shape[2], x.shape[4]])
        
        _, T, n, c_out = x.shape
        
        x_input = x
        x_input = fl.reshape(x_input, [-1, c_out])
        graph.tensor()
        
        if num == 1:
            x_input = self.graphconv1(graph, x_input)
        elif num == 2:
            x_input = self.graphconv2(graph, x_input)
        elif num == 0:
            x_input = self.graphconvw(graph, x_input)
            
        x_input = fl.reshape(x_input, [-1, T, n, c_out])
        x_input = x_input + x
        x_input = nn.functional.relu(x_input)
        return x_input

    def output_block(self, x, w, act_func='GLU'):
        """Output layer"""
        _, _, n, channel = x.shape
        
        
        x = paddle.concat((x, w), axis = -1)
        #now x.shape = B, T, N, blocks[1][2] + blocks[0][1]
        
        # maps multi-steps to one.
        x = self.temporal_conv_layer(
            x,
            3,
            1,
            act_func=act_func)
           
        x = self.ln_o(x)
         
        x = self.temporal_conv_layer(
            x,
            3,
            2,
            act_func='sigmoid')
        # maps multi-channels to one.
        x = self.fc(x)
        return x
