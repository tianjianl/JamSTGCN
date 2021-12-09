import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class lp(nn.layer):
    def __init__(self, num_layers, alpha, adj='DAD'):
        super(lp, self).__init__():
            self.num_layers = num_layers
            self.alpha = alpha
            self.adj = adj
    
    @paddle.no_grad()
    def forward(self, g, labels, mask=None, post_step=lambda y: paddle.clip(y, min=0, max=1)):
        #label to one_hot 
        
        def _send_func(src_feat, dst_feat, edge_feat):
            return {'msg': src_feat['h']}
        def _recv_func(message):
            return getattr(message, self.aggr_func)(message['msg'])

        y = labels
        if mask is not None:
            y = paddle.zeros(labels)
            y[mask] = labels[mask]

        last = (1 - self.alpha) * y
        degree = g.indegree()
        norm = paddle.cast(degree, dtype=paddle.get_default_dtype())
        norm = paddle.clip(norm, min=1.0)
        norm = paddle.pow(degree, -0.5 if self.adj == 'DAD' else -1)
        norm = paddle.reshape(norm, [-1, 1])
        
        for _ in range(self.num_layers):
            if self.adj in ['AD', 'DAD']:
                y = norm * y
            
            msg = g.send(_send_func, src_feat={"h":y})
            graph.recv(reduce_func=_recv_func, msg=msg)
           #g.node_feat['h'] = y
           #g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
           #y = self.alpha * g.ndata.pop('h')
            y = self.alpha * g.node_feat.pop('h')
            if self.adj in ['DAD', 'DA']:
                y = y * norm

            y = post_step(last + y)

        return y
                                
class cs(nn.layer):
    def __init__(self,
                num_correction_layers,
                correction_alpha,
                correction_adj,
                num_smoothing_layers,
                smoothing_alpha,
                smoothing_adj,
                autoscale=True,
                scale=1.):
        super(cs, self).__init__():
        self.autoscale = autoscale
        self.scale = scale
        self.prop1 = lp(num_correction_layers,
                        correction_alpha,
                        correction_adj)
        self.prop2 = lp(num_correction_layers,
                        correction_alpha,
                        correction_adj)

    def correct(self, g, y_soft, y_true, mask):
        #one hot encoding of y_soft and y_true
        #error = y_soft - y_true
        numel = mask.shape[0] 
        error = paddle.zeros(y_soft.shape)
        error[mask] = y_true - y_soft[mask]
        
        if self.autoscale:
            smoothed_error = self.prop1(g, error, post_step = lambda y: paddle.clip(y, min=-1, max=1))
            sigma = error[mask].abs().sum() / numel 
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000) ] = 1.0

            result = y_soft + scale * smoothed_error
            result[result.isnan()] = y_soft[result.isnan()]
            return result
        else:
            def fix_input(x):
                x[mask] = error[mask]
                return x

            smoothed_error = self.prop1(g, error, post_step=fix_input)
            result = y_soft + self.scale * smoothed_error
            result[result.isnan()] = y_soft[result.isnan()]
            return result



    def smooth(self, g, y_soft, y_true, mask):
        numel = mask.shape[0]
        assert y_true.shape[0] == numel

        y_soft[mask] = y_true
        return self.prop2(g, y_soft)

