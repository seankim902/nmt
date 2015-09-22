# -*- coding: utf-8 -*-
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from keras import activations, initializations

import cPickle as pkl
import numpy as np

class BiRNN_EncDec:
    def __init__(self, n_vocab, dim_word, dimctx, dim):
        self.n_vocab = n_vocab  # 30000
        self.dim_word = dim_word # 384
        self.dimctx = dimctx  # 1024
        self.dim = dim  # 512
        
        ### Word Embedding ###        
        self.W_enc_emb = initializations.uniform((self.n_vocab, self.dim_word))
        self.W_dec_emb = initializations.uniform((self.n_vocab, self.dim_word))

        ### enc forward GRU ###
        self.W_enc_f_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_enc_f_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_enc_f_gru = initializations.zero((self.dim * 2))
        self.W_enc_f_gru_cdd = initializations.uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_enc_f_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_enc_f_gru_cdd = initializations.zero((self.dim))
        
        ### enc backward GRU ###
        self.W_enc_b_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_enc_b_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_enc_b_gru = initializations.zero((self.dim * 2))
        self.W_enc_b_gru_cdd = initializations.uniform((self.dim_word, self.dim))
        self.U_enc_b_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_enc_b_gru_cdd = initializations.zero((self.dim))
        
        ### context to decoder init state (s0)
        self.W_dec_init = initializations.uniform((self.dimctx, dim))
        self.b_dec_init = initializations.zero((dim))
        
        ### dec GRU ###
        self.W_dec_gru = initializations.uniform((self.dim_word, self.dim * 2))
        self.U_dec_gru = initializations.uniform((self.dim, self.dim * 2))
        self.b_dec_gru = initializations.zero((self.dim * 2))
        self.W_dec_gru_cdd = initializations.uniform((self.dim_word, self.dim))
        self.U_dec_gru_cdd = initializations.uniform((self.dim, self.dim))
        self.b_dec_gru_cdd = initializations.zero((self.dim))      
        self.W_dec_gru_ctx = initializations.uniform((self.dimctx, self.dim * 2))
        self.W_dec_gru_ctx_cdd = initializations.uniform((self.dimctx, self.dim))

        ### enc-dec attention ###
        self.W_att_y2c = initializations.uniform((self.dim_word, self.dimctx))
        self.W_att_h2c = initializations.uniform((self.dimctx, self.dimctx))
        self.W_att_s2c = initializations.uniform((self.dim, self.dimctx))
        self.b_att = initializations.zero((self.dimctx))

        self.U_att_energy = initializations.uniform((self.dimctx, 1))
        self.b_att_energy = initializations.zero((1,))

        ### enc-dec prediction ###
        self.W_dec_pred_s2y = initializations.uniform((self.dim, self.dim_word))
        self.b_dec_pred_s2y = initializations.zero((self.dim_word))
        self.W_dec_pred_y2y = initializations.uniform((self.dim_word, self.dim_word))
        self.b_dec_pred_y2y = initializations.zero((self.dim_word))
        self.W_dec_pred_c2y = initializations.uniform((self.dim * 2, self.dim_word))
        self.b_dec_pred_c2y = initializations.zero((self.dim_word))
        self.W_dec_pred = initializations.uniform((self.dim_word, self.n_vocab))
        self.b_dec_pred = initializations.zero((self.n_vocab))


        self.params = [self.W_enc_emb, self.W_dec_emb,
                       self.W_enc_f_gru, self.U_enc_f_gru, self.b_enc_f_gru, self.W_enc_f_gru_cdd, self.U_enc_f_gru_cdd, self.b_enc_f_gru_cdd,
                       self.W_enc_b_gru, self.U_enc_b_gru, self.b_enc_b_gru, self.W_enc_b_gru_cdd, self.U_enc_b_gru_cdd, self.b_enc_b_gru_cdd,
                       self.W_dec_init, self.b_dec_init,
                       self.W_dec_gru, self.U_dec_gru, self.b_dec_gru, self.W_dec_gru_cdd, self.U_dec_gru_cdd, self.b_dec_gru_cdd,
                       self.W_dec_gru_ctx, self.W_dec_gru_ctx_cdd,
                       self.W_att_y2c, self.W_att_h2c, self.W_att_s2c, self.b_att,
                       self.U_att_energy, self.b_att_energy,
                       self.W_dec_pred_s2y, self.b_dec_pred_s2y,
                       self.W_dec_pred_y2y, self.b_dec_pred_y2y,
                       self.W_dec_pred_c2y, self.b_dec_pred_c2y,
                       self.W_dec_pred, self.b_dec_pred]


    def gru_enc_f_layer(self, state_below, mask=None, **kwargs):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.U_enc_f_gru_cdd.shape[1]
    
        if mask == None:
            mask = T.alloc(1., state_below.shape[0], 1)
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_enc_f_gru) + self.b_enc_f_gru
        state_belowx = T.dot(state_below, self.W_enc_f_gru_cdd) + self.b_enc_f_gru_cdd

        def _step_slice(m_, x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, xx_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_ # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return h#, r, u, preact, preactx
    
        seqs = [mask, state_below_, state_belowx]
        _step = _step_slice
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [T.alloc(0., n_samples, dim)],
                                                    #None, None, None, None],
                                    non_sequences = [self.U_enc_f_gru, 
                                                     self.U_enc_f_gru_cdd],
                                    name='enc_gru_f_layer', 
                                    n_steps=nsteps)
        rval = [rval]
        return rval

    def gru_enc_b_layer(self, state_below, mask=None, **kwargs):
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.U_enc_b_gru_cdd.shape[1]
    
        if mask == None:
            mask = T.alloc(1., state_below.shape[0], 1)
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_enc_b_gru) + self.b_enc_b_gru
        state_belowx = T.dot(state_below, self.W_enc_b_gru_cdd) + self.b_enc_b_gru_cdd

        def _step_slice(m_, x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, xx_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim))
            u = T.nnet.sigmoid(_slice(preact, 1, dim))
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_ # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_  # m_[:,None] : samples * 1
    
            return h#, r, u, preact, preactx
    
        seqs = [mask, state_below_, state_belowx]
        _step = _step_slice
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [T.alloc(0., n_samples, dim)],
                                                    #None, None, None, None],
                                    non_sequences = [self.U_enc_b_gru, 
                                                     self.U_enc_b_gru_cdd],
                                    name='enc_gru_b_layer',
                                    n_steps=nsteps)
        rval = [rval]
        return rval
      
    def gru_dec_layer(self, state_below, mask=None, context=None, one_step=False, 
                      init_memory=None, init_state=None,  context_mask=None, **kwargs):
    
        assert context, 'Context must be provided'
    
        if one_step:
            assert init_state, 'previous state must be provided'
    
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
    
        # mask
        if mask == None:
            mask = T.alloc(1., state_below.shape[0], 1)
    
        dim = self.U_dec_gru_cdd.shape[1]
    
        # initial/previous state
        if init_state == None:
            init_state = T.alloc(0., n_samples, dim)
    
        # projected context 
        assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
        pctx_ = T.dot(context, self.W_att_h2c) + self.b_att
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        # projected x
        state_below_ = T.dot(state_below, self.W_dec_gru) + self.b_dec_gru
        state_belowx = T.dot(state_below, self.W_dec_gru_cdd) + self.b_dec_gru_cdd
        state_belowc = T.dot(state_below, self.W_att_y2c)

        def _step_slice(m_, x_, xx_, xc_, s_, ctx_, alpha_, pctx_, cc_,
                        U, Wc, Wd_att, U_att, c_tt, Ux, Wcx):
                            # ctx_ : samples * 1024
                            # alpha_ : samples * steps
                            # pctx_ : enc_step * sample * 1024
                            # cc_ : ctx : enc_steps*samples*1024
            # attention
            pstate_ = T.dot(s_, Wd_att)
            pctx__ = pctx_ + pstate_[None,:,:] 
            pctx__ += xc_
            pctx__ = T.tanh(pctx__)
            alpha = T.dot(pctx__, U_att)+c_tt
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
            alpha = T.exp(alpha)
            if context_mask:
                alpha = alpha * context_mask
            alpha = alpha / alpha.sum(0, keepdims=True)
            ctx_ = (cc_ * alpha[:,:,None]).sum(0) # current context
    
            preact = T.dot(s_, U)
            preact += x_
            preact += T.dot(ctx_, Wc)
            preact = T.nnet.sigmoid(preact)
    
            r = _slice(preact, 0, dim)
            u = _slice(preact, 1, dim)
    
            preactx = T.dot(s_, Ux)
            preactx *= r
            preactx += xx_
            preactx += T.dot(ctx_, Wcx)
    
            s = T.tanh(preactx)
    
            s = u * s_ + (1. - u) * s
            s = m_[:,None] * s + (1. - m_)[:,None] * s_
    
            return s, ctx_, alpha.T #, pstate_, preact, preactx, r, u
    
        seqs = [mask, state_below_, state_belowx, state_belowc]
        _step = _step_slice



    
        shared_vars = [self.U_dec_gru,
                       self.W_dec_gru_ctx,
                       self.W_att_s2c,
                       self.U_att_energy, 
                       self.b_att_energy, 
                       self.U_dec_gru_cdd, 
                       self.W_dec_gru_ctx_cdd]
    
        if one_step:
            rval = _step(*(seqs+[init_state, None, None, pctx_, context]+shared_vars))
        else:
            rval, updates = theano.scan(_step, 
                                        sequences=seqs,
                                        outputs_info = [init_state, 
                                                        T.alloc(0., n_samples, context.shape[2]), # sampple * 1024
                                                        T.alloc(0., n_samples, context.shape[0])], # sample * enc_steps
                                        non_sequences=[pctx_,
                                                       context]+shared_vars,
                                        n_steps=nsteps
                                        #profile=profile
                                        )
        return rval


                            
    def build_model(self, lr=0.001):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.matrix('x', dtype = 'int32')
        x_mask = T.matrix('x_mask', dtype='float32')
        y = T.matrix('y', dtype = 'int32')
        y_mask = T.matrix('y_mask', dtype='float32')
        xr = T.matrix('x', dtype = 'int32')
        xr_mask = T.matrix('x_mask', dtype='float32')
        
        xr = x[::-1]
        xr_mask = x_mask[::-1]
    
        n_timesteps = x.shape[0]
        n_timesteps_target = y.shape[0]
        n_samples = x.shape[1]

        ## BIRNN  encoder    
        emb = self.W_enc_emb[x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        # proj : forward hidden 들의 리스트   
        proj = self.gru_enc_f_layer(emb, mask=x_mask)
                                                
                                                
        embr = self.W_enc_emb[xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.dim_word])
        # projr : backward hidden 들의 리스트
        projr = self.gru_enc_b_layer(embr, mask=xr_mask)
                                               
        ctx = self.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        # ctx : step * samples * (dim*2)
        ctx_mean = (ctx * x_mask[:,:,None]).sum(0) / x_mask.sum(0)[:,None]
        # ctx_mean : samples * (dim*2)
        #ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)
    
        # initial decoder state -> samples * 512 : enc h(forward),h(backward) to s0
        init_state = T.nnet.softmax(T.dot(ctx_mean, self.W_dec_init)+self.b_dec_init)

    
        
        emb = self.W_dec_emb[y.flatten()]
        emb = emb.reshape([n_timesteps_target, n_samples, self.dim_word])
        emb_shifted = T.zeros_like(emb)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
        # decoder
        proj = self.gru_dec_layer(emb, mask=y_mask, context=ctx, one_step=False,
                                  init_state=init_state, context_mask=x_mask)
        proj_h = proj[0]
        ctxs = proj[1]                                    
    
        # compute word probabilities
        logit_lstm = T.dot(proj_h, self.W_dec_pred_s2y) + self.b_dec_pred_s2y
        logit_prev = T.dot(emb, self.W_dec_pred_y2y) + self.b_dec_pred_y2y
        logit_ctx = T.dot(ctxs, self.W_dec_pred_c2y) + self.b_dec_pred_c2y
        logit = T.tanh(logit_lstm+logit_prev+logit_ctx)
        logit = T.dot(logit, self.W_dec_pred) + self.b_dec_pred
        logit_shp = logit.shape
        probs = T.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], 
                                                   logit_shp[2]]))
                                                   
    
    
        # cost
        y_flat = y.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.n_vocab + y_flat
        cost = -T.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0],y.shape[1]])
        cost = (cost * y_mask).sum(0).mean()
        
        updates = self.adam(cost=cost, params=self.params, lr=lr)

        return x, x_mask, y, y_mask, cost, updates
        
        
    
    def sgd(self, cost, params, lr=0.001):
        updates = []
        grads = T.grad(cost, params)
        for param, grad in zip(params, grads):
            updates.append((param, param - lr * grad))
        return updates
 
    def adam(self, cost, params, lr=0.001, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(np.float32(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for param, grad in zip(params, grads):
            m = theano.shared(param.get_value() * 0.)
            v = theano.shared(param.get_value() * 0.)
            m_t = (b1 * grad) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(grad)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = param - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))
        updates.append((i, i_t))
        return updates
        
    def train(self, train_x, train_mask_x, 
              train_y, train_mask_y, 
             # valid_x=None, valid_mask_x=None, 
              #valid_y=None, valid_mask_y=None,
             # optimizer=None,
              lr=0.001,
              batch_size=16,
              epoch=100):
        
        
        train_shared_x = theano.shared(np.asarray(train_x, dtype='int32'), borrow=True)
        train_shared_y = theano.shared(np.asarray(train_y, dtype='int32'), borrow=True)
        train_shared_mask_x = theano.shared(np.asarray(train_mask_x, dtype='float32'), borrow=True)
        train_shared_mask_y = theano.shared(np.asarray(train_mask_y, dtype='float32'), borrow=True)

        n_train = train_shared_x.get_value(borrow=True).shape[1]
        n_train_batches = int(np.ceil(1.0 * n_train / batch_size))
        
        index = T.lscalar('index')    # index to a case
        final_index = T.lscalar('final_index')
        
        
        x, x_mask, y, y_mask, cost, updates = self.build_model(lr)
       
       
        batch_start = index * batch_size
        batch_stop = T.minimum(final_index, (index + 1) * batch_size)
     
     
                    
        train_model = theano.function(inputs=[index, final_index],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                         x: train_shared_x[:,batch_start:batch_stop],
                                         x_mask: train_shared_mask_x[:,batch_start:batch_stop],
                                         y: train_shared_y[:,batch_start:batch_stop],
                                         y_mask: train_shared_mask_y[:,batch_start:batch_stop]})
       
        
        i = 0
        while (i < epoch):
            i = i + 1
            print 'epoch : ', i
            for minibatch_idx in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_idx, n_train)
                print 'cost : ' , minibatch_avg_cost, ' [ mini batch \'', minibatch_idx+1, '\' in epoch \'', i ,'\' ]'
       
       
    def concatenate(self, tensor_list, axis=0):
        """
        Alternative implementation of `theano.T.concatenate`.
        This function does exactly the same thing, but contrary to Theano's own
        implementation, the gradient is implemented on the GPU.
        Backpropagating through `theano.T.concatenate` yields slowdowns
        because the inverse operation (splitting) needs to be done on the CPU.
        This implementation does not have that problem.
        :usage:
            >>> x, y = theano.T.matrices('x', 'y')
            >>> c = concatenate([x, y], axis=1)
        :parameters:
            - tensor_list : list
                list of Theano tensor expressions that should be concatenated.
            - axis : int
                the tensors will be joined along this axis.
        :returns:
            - out : tensor
                the concatenated tensor expression.
        """
        concat_size = sum(tt.shape[axis] for tt in tensor_list)
    
        output_shape = ()
        for k in range(axis):
            output_shape += (tensor_list[0].shape[k],)
        output_shape += (concat_size,)
        for k in range(axis + 1, tensor_list[0].ndim):
            output_shape += (tensor_list[0].shape[k],)
    
        out = T.zeros(output_shape)
        offset = 0
        for tt in tensor_list:
            indices = ()
            for k in range(axis):
                indices += (slice(None),)
            indices += (slice(offset, offset + tt.shape[axis]),)
            for k in range(axis + 1, tensor_list[0].ndim):
                indices += (slice(None),)
    
            out = T.set_subtensor(out[indices], tt)
            offset += tt.shape[axis]
    
        return out
