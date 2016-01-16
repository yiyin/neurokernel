from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

cuda_src = """
// %(type)s and %(nneu)d must be replaced using Python string foramtting
#define NNEU %(nneu)d

__global__ void leaky_iaf(
    int neu_num,
    %(type)s dt,
    int      *spk,
    %(type)s *V,
    %(type)s *I,
    %(type)s *V_thres,
    %(type)s *V_rest,
    %(type)s *V_reset,
    %(type)s *refrac,
    %(type)s *Tm,
    %(type)s *C,
    %(type)s *bias,
    %(type)s *in_refrac)
{
    int bid = blockIdx.x;
    int nid = bid * NNEU + threadIdx.x;

    %(type)s v,i,tm,c,vr,vs,ref_left;
    int spike;

    if( nid < neu_num ){
        v = V[nid];
        i = I[nid];
        tm = Tm[nid];
        c = C[nid];
        vr = V_rest[nid];
        vs = V_reset[nid];
        
        ref_left = fmax(in_refrac[nid]-dt, 0);

        // update v
        %(type)s bh = exp( -dt/tm );
        v = v*bh + ((ref_left == 0 ? tm/c*(i+bias[nid]) : 0) + vr)*(1.0-bh);

        // spike detection
        spike = 0;
        if( v >= V_thres[nid] )
        {
            v = vs;
            spike = 1;
            ref_left += refrac[nid];
        }
        

        V[nid] = v;
        spk[nid] = spike;
        in_refrac[nid] = ref_left;
    }
    return;
}
"""

class LIF(BaseNeuron):
    def __init__(self, n_dict, spk, V, dt, debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug
        self.LPU_id = LPU_id

        self.V_rest  = garray.to_gpu( np.asarray( n_dict['V_rest'], dtype=np.float64 ))
        self.V_thres  = garray.to_gpu( np.asarray( n_dict['Vt'], dtype=np.float64 ))
        self.V_reset = garray.to_gpu( np.asarray( n_dict['V_reset'], dtype=np.float64 ))
        self.C   = garray.to_gpu( np.asarray( n_dict['C'], dtype=np.float64 ))
        self.tm   = garray.to_gpu( np.asarray( n_dict['tm'], dtype=np.float64 ))
        self.refrac = garray.to_gpu( np.asarray( n_dict['ref'], dtype=np.float64 ))
        self.V   = garray.GPUArray((self.num_neurons,), dtype=np.float64, gpudata=V)
        self.p   = garray.to_gpu( np.asarray( n_dict['p'], dtype=np.float64)) * self.C
        self.in_refrac = garray.zeros((self.num_neurons), dtype=np.float64)
        #self.V   = garray.to_gpu( np.asarray( n_dict['V'], dtype=np.float64 ))
        self.spk = spk

        _num_dendrite_cond = np.asarray([n_dict['num_dendrites_cond'][i]
                                         for i in range(self.num_neurons)],
                                        dtype=np.int32).flatten()
        _num_dendrite = np.asarray([n_dict['num_dendrites_I'][i]
                                    for i in range(self.num_neurons)],
                                   dtype=np.int32).flatten()

        self._cum_num_dendrite = garray.to_gpu(np.concatenate((
                                    np.asarray([0,], dtype=np.int32),
                                    np.cumsum(_num_dendrite, dtype=np.int32))))
        self._cum_num_dendrite_cond = garray.to_gpu(np.concatenate((
                                    np.asarray([0,], dtype=np.int32),
                                    np.cumsum(_num_dendrite_cond, 
                                              dtype=np.int32))))
        self._num_dendrite = garray.to_gpu(_num_dendrite)
        self._num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
        self._pre = garray.to_gpu(np.asarray(n_dict['I_pre'], dtype=np.int32))
        self._cond_pre = garray.to_gpu(np.asarray(n_dict['cond_pre'],
                                                  dtype=np.int32))
        self._V_rev = garray.to_gpu(np.asarray(n_dict['reverse'],
                                               dtype=np.double))
        self.I = garray.zeros(self.num_neurons, np.double)
        self._update_I_cond = self._get_update_I_cond_func()
        self._update_I_non_cond = self._get_update_I_non_cond_func()
        self.update = self.get_gpu_kernel()
        if self.debug:
            if self.LPU_id is None:
                self.LPU_id = "anon"
            self.I_file = tables.openFile(self.LPU_id + "_I.h5", mode="w")
            self.I_file.createEArray("/","array",
                                     tables.Float64Atom(), (0,self.num_neurons))
            self.V_file = tables.openFile(self.LPU_id + "_V.h5", mode="w")
            self.V_file.createEArray("/","array",
                                     tables.Float64Atom(), (0,self.num_neurons))
    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        self.update.prepared_async_call(
            self.gpu_grid,
            self.gpu_block,
            st,
            self.num_neurons,
            self.dt,
            self.spk,
            self.V.gpudata,
            self.I.gpudata,
            self.V_thres.gpudata,
            self.V_rest.gpudata,
            self.V_reset.gpudata,
            self.refrac.gpudata,
            self.tm.gpudata,
            self.C.gpudata,
            self.p.gpudata,
            self.in_refrac.gpudata)
        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1, -1)))
            self.V_file.root.array.append(self.V.get().reshape((1, -1)))
            

    def get_gpu_kernel( self):
        self.gpu_block = (128, 1, 1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        #cuda_src = open( './leaky_iaf.cu','r')
        mod = SourceModule(
                cuda_src % {"type": dtype_to_ctype(np.float64),
                            "nneu": self.gpu_block[0] },
                options=["--ptxas-options=-v"])
        func = mod.get_function("leaky_iaf")
        func.prepare('idPPPPPPPPPPP')
        return func
        
    def post_run(self):
        if self.debug:
            self.I_file.close()
            self.V_file.close()

    @property
    def update_I_override(self): return True

    def update_I(self, synapse_state, st=None):
        self.I.fill(0)
        if self._pre.size>0:
            self._update_I_non_cond.prepared_async_call(self._grid_get_input,
                self._block_get_input, st, int(synapse_state),
                self._cum_num_dendrite.gpudata, self._num_dendrite.gpudata,
                self._pre.gpudata, self.I.gpudata)
        if self._cond_pre.size>0:
            self._update_I_cond.prepared_async_call(self._grid_get_input,
                self._block_get_input, st, int(synapse_state),
                self._cum_num_dendrite_cond.gpudata, 
                self._num_dendrite_cond.gpudata,
                self._cond_pre.gpudata, self.I.gpudata, self.V.gpudata,
                self._V_rev.gpudata)
        

if __name__ == '__main__':
    import atexit
    cuda.init()
    ctx=cuda.Device(0).make_context()
    atexit.register(ctx.pop)
    SourceModule(
                cuda_src % {"type": dtype_to_ctype(np.float64),
                            "nneu": 128 },
                options=["--ptxas-options=-v"])

