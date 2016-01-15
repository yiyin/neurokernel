"""
Exponential Synapse Model with support for pre-synaptic connection
"""
from basesynapse import BaseSynapse

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_src_synapse_kernel = """
__global__ void exponential_synapse(
    int num,
    %(type)s dt,
    int *spike,
    int *Pre,
    %(type)s *Tau,
    %(type)s *A,
    %(type)s *Gmax,
    %(type)s *Eff,
    %(type)s *cond )
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    int pre;
    %(type)s a,tau,gmax,eff,d_eff;

    for( int i=tid; i<num; i+=tot_threads ){
        // copy data from global memory to register
        pre = Pre[i];
        a = A[i];
        tau = Tau[i];
        eff = Eff[i];
        gmax = Gmax[i];

        // update the exponetial function
        d_eff = -eff/tau;
        if( spike[pre] )
           d_eff += (1-eff)*a;
        eff += dt*d_eff;

        // copy data from register to the global memory
        Eff[i] = eff;
        cond[i] = eff*gmax;
    }
    return;
}
"""
cuda_src_synapse_update_I = """
#define N 32
#define NUM %(num)d

__global__ void get_input(
    double* synapse,
    int* cum_num_dendrite,
    int* num_dendrite,
    int* pre,
    double* I_pre)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;

    int sid;

    __shared__ int num_den[32];
    __shared__ int den_start[32];
    __shared__ double input[32][33];

    if(tidy == 0)
    {
        sid = bid * N + tidx;
        if(sid < NUM)
        {
            num_den[tidx] = num_dendrite[sid];
        }
    } else if(tidy == 1)
    {
        sid = bid * N + tidx;
        if(sid < NUM)
        {
            den_start[tidx] = cum_num_dendrite[sid];
        }
    }

    input[tidy][tidx] = 0.0;

    __syncthreads();

    sid = bid * N + tidy;
    if(sid < NUM){
       int n_den = num_den[tidy];
       int start = den_start[tidy];

       for(int i = tidx; i < n_den; i += N)
       {
           input[tidy][tidx] += synapse[pre[start + i]];
       }
    }


    __syncthreads();

    if(tidy < 8)
    {
        input[tidx][tidy] += input[tidx][tidy + 8];
        input[tidx][tidy] += input[tidx][tidy + 16];
        input[tidx][tidy] += input[tidx][tidy + 24];
    }

    __syncthreads();

    if(tidy < 4)
    {
        input[tidx][tidy] += input[tidx][tidy + 4];
    }

    __syncthreads();

    if(tidy < 2)
    {
        input[tidx][tidy] += input[tidx][tidy + 2];
    }

    __syncthreads();

    if(tidy == 0)
    {
        input[tidx][0] += input[tidx][1];
        sid = bid*N+tidx;
        if(sid < NUM)
        {
            I_pre[sid] += input[tidx][0];
        }
    }
}
//can be improved
"""
class ExpSynapse(BaseSynapse):
    """
    Exponential Decay Synapse
    """
    def __init__( self, s_dict, synapse_state, dt, debug=False):
        self.debug = debug
        self.dt = dt
        self.num = len( s_dict['id'] )

        self.pre  = garray.to_gpu( np.asarray( s_dict['pre'], dtype=np.int32 ))
        self.a    = garray.to_gpu( np.asarray( s_dict['a'], dtype=np.float64 ))
        self.tau  = garray.to_gpu( np.asarray( s_dict['tau'], dtype=np.float64 ))
        self.gmax = garray.to_gpu( np.asarray( s_dict['gmax'], dtype=np.float64 ))
        self.eff  = garray.zeros( (self.num,), dtype=np.float64 )
        self.cond = synapse_state

        _num_dendrite_cond = np.asarray(
            [s_dict['num_dendrites_cond'][i] for i in s_dict['id']],\
            dtype=np.int32).flatten()
        _num_dendrite = np.asarray(
            [s_dict['num_dendrites_I'][i] for i in s_dict['id']],\
            dtype=np.int32).flatten()

        self._cum_num_dendrite = garray.to_gpu(_0_cumsum(_num_dendrite))
        self._cum_num_dendrite_cond = garray.to_gpu(_0_cumsum(_num_dendrite_cond))
        self._num_dendrite = garray.to_gpu(_num_dendrite)
        self._num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
        self._pre = garray.to_gpu(np.asarray(s_dict['I_pre'], dtype=np.int32))
        self._cond_pre = garray.to_gpu(np.asarray(s_dict['cond_pre'], dtype=np.int32))
        self._V_rev = garray.to_gpu(np.asarray(s_dict['reverse'],dtype=np.double))
        self.I = garray.zeros(self.num, np.double)
        #self._update_I_cond = self._get_update_I_cond_func()
        self._update_I_non_cond = self._get_update_I_non_cond_func()

        self.update = self._get_gpu_kernel()

    @property
    def synapse_class(self): return int(0)

    def update_state(self, buffer, V, st = None):
        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            buffer.spike_buffer.gpudata,\
            self.pre.gpudata,\
            self.tau.gpudata,\
            self.a.gpudata,\
            self.gmax.gpudata,\
            self.eff.gpudata,\
            self.cond)

    def update_I(self, synapse_state, st=None):
        self.I.fill(0.)
        if self._pre.size > 0:
            self._update_I_non_cond.prepared_async_call(
                self._grid_get_input,
                self._block_get_input,
                st,
                int(synapse_state),
                self._cum_num_dendrite.gpudata,
                self._num_dendrite.gpudata,
                self._pre.gpudata,
                self.I.gpudata)

    def _get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        # cuda_src = open('./alpha_synapse.cu','r')
        mod = SourceModule( \
                cuda_src_synapse_kernel % {"type": dtype_to_ctype(np.float64)},\
                options=["--ptxas-options=-v"])
        func = mod.get_function("exponential_synapse")
        func.prepare('idPPPPPPP')
#                     [  np.int32,   # syn_num
#                        np.float64, # dt
#                        np.intp,    # spike list
#                        np.intp,    # pre-synaptic neuron list
#                        np.intp,    # tau; time constant
#                        np.intp,    # a; bump size
#                        np.intp,    # gmax array
#                        np.intp,    # eff; efficacy
#                        np.intp ] ) # cond array
        return func

    def _get_update_I_non_cond_func(self):
        mod = SourceModule(\
                cuda_src_synapse_update_I % {"num": self.num},
                options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare('PPPPP')
#                     [np.intp,  # synapse state
#                      np.intp,  # cumulative dendrites number
#                      np.intp,  # dendrites number
#                      np.intp,  # pre-synaptic number ID
#                      np.intp]) # output

        self._block_get_input = (32,32,1)
        self._grid_get_input = ((self.num - 1) / 32 + 1, 1)
        return func

def _0_cumsum(it, dtype=np.int32):
    """
    Like numpy.cumsum but with 0 at the head of output, i.e.
    [0, np.cumsum(it)]
    """
    return np.concatenate((np.asarray([0,], dtype=dtype),
                           np.cumsum(it, dtype=dtype)))
