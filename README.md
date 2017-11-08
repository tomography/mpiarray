MpiArray simplifies distributing numpy arrays efficiently across a cluster.  The arrays can be split across an arbitrary axis and scattered to the different nodes in a customizable way with padding.

The MpiArray object stores the metadata about the whole array and the distribution of the local arrays across the cluster.  Each MPI process creates an instance of the MpiArray object which is then used to distribute data across the cluster.  The following guidlines are followed:
* Data is always sent without pickling (this is more efficient)
* There are no restrictions on number of dimensions or data sizes
    ** Data scatter axis size does NOT need to be a multiple of MPI_Size
    ** Data is evenly distributed across processes by default
    ** Custom contiguous distributions are supported
* Copies of data are avoided (except with padding)
* Data is only gathered through gather and allgather
    ** MpiArray supports arrays larger than available memory on a single node
    ** MpiArray can re-scatter with different axis/padding in a distributed manner
* Only the un-padded data is used when data is re-distributed through 
    scatter or gather (padding is discarded).
* Data is always contiguous in memory
* An mpi4py comm object can be used to define which processes to use


You can create an MPiArray from a global array on a single MPI process or from local arrays on each MPI process.  Remember, all calls to MpiArray are collective and need to called from every process.


Initialization example:

from mpi4py import MPI
from mpiarray import MpiArray
import numpy as np

# load from global array
if MPI.COMM_WORLD.Get_rank() == 0:
    arr = np.zeros((5, 5))
else:
    arr = None
mpiarray = MpiArray(arr)

# load from local arrays
arr = np.zeros((5, 5))
mpiarray = MpiArray(arr, axis=0)
# NOTE: overall shape of mpiarray is (5*mpi_size, 5) in second example.

The following data scatter/gather functions are supported:
* scatter(axis, padding)
    ** splits the data across an axis
    ** padding allows for overlap
    ** returns the local numpy array to every process
* scatterv(distribution)
    ** splits data according to the distribution 
    ** returns the local numpy array to every process
* scattermovezero(axis, padding)
    ** splits the data across an axis, and that axis is moved to axis 0. 
    ** This is very useful for tomographic reconstructions.
    ** returns the local numpy array to every process
* scattervmovezero(distribution)
    ** splits data according to the distribution
    ** then moves distribution axis to axis 0.
    ** returns the local numpy array to every process
* gather(root)
    ** returns the full array on MPI process with rank root. 
* gatherall()
    ** returns the full array to every process.
    
Custom distributions can be used to specify how the data is distributed across an axis to the MPI processes.  They need to provide the following:
* axis - (int) axis data is split on to give to different MPI processes
* sizes - (ndarray of int) size of slice on axis for each MPI process, 
    ordered by rank.  This includes any padding.
* offsets - (ndarray of int) offsets of slice on axis for each MPI 
    process, ordered by rank.  This includes any padding.
* unpadded_sizes - (ndarray of int) size of slice on axis for each MPI 
    process, ordered by rank.  This excludes any padding.
* unpadded_offsets - (ndarray of int) offsets of slice on axis for each
    MPI process, ordered by rank.  This excludes any padding.
NOTE: The unpadded data should have a one-to-one correspondence to a single MPI process (data should only be present in a single unpadded region and all data should be represented by the unpadded regions). 

Similar Python packages:
MpiArray is similar to distarray.  The main difference is that mpiarray never uses pickling or Python slices to send data.  Everything is sent directly using MPI calls.  This is more efficient and supports large array sizes (distarray seemed to have a 1GB limit), however it requires that the arrays always be c contiguous in memory.  distarray has a more general set of array distributions supported.
