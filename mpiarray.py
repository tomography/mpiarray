# As a work of the United States government, this project is in the
# public domain within the United States.
# 
# Additionally, we waive copyright and related rights in the work
# worldwide through the CC0 1.0 Universal public domain dedication.
# 
# ## CC0 1.0 Universal summary
# 
# This is a human-readable summary of the Legal Code located at
# https://creativecommons.org/publicdomain/zero/1.0/legalcode.
# 
# ### No copyright
# 
# The person who associated a work with this deed has dedicated the work to
# the public domain by waiving all rights to the work worldwide
# under copyright law, including all related and neighboring rights, to the
# extent allowed by law.
# 
# You can copy, modify, distribute and perform the work, even for commercial
# purposes, all without asking permission.
# 
# ### Other information
# 
# In no way are the patent or trademark rights of any person affected by CC0,
# nor are the rights that other persons may have in the work or in how the
# work is used, such as publicity or privacy rights.
# 
# Unless expressly stated otherwise, the person who associated a work with
# this deed makes no warranties about the work, and disclaims liability for
# all uses of the work, to the fullest extent permitted by applicable law.
# When using or citing the work, you should not imply endorsement by the
# author or the affirmer.

"""
MpiArray simplifies distributing numpy arrays efficiently across a 
cluster.  The arrays can be split across an arbitrary axis and 
scattered to the different nodes in a customizable way with padding.

The MpiArray object stores the metadata about the whole array and the 
distribution of the local arrays across the cluster.  Each MPI process 
creates an instance of the MpiArray object which is then used to 
distribute data across the cluster.  The following guidlines are 
followed:
* Data is always sent without pickling (this is more efficient)
* There are no restrictions on number of dimensions or data sizes
    ** Data scatter axis size does NOT need to be a multiple of 
        MPI_Size
    ** Data is evenly distributed across processes by default
    ** Custom contiguous distributions are supported
* Copies of data are avoided (except with padding)
* Data is only gathered through gather and allgather
    ** MpiArray supports arrays larger than available memory on a 
        single node
    ** MpiArray can re-scatter with different axis/padding in a 
        distributed manner
* Only the un-padded data is used when data is re-distributed through 
    scatter or gather (padding is discarded).
* Data is always contiguous in memory
* An mpi4py comm object can be used to define which processes to use


You can create an MPiArray from a global array on a single MPI process 
or from local arrays on each MPI process.  Remember, all calls to 
MpiArray are collective and need to called from every process.


Initialization example:

from mpi4py import MPI
from mpiarray import MpiArray
import numpy as np

# load from global array
if MPI.COMM_WORLD.Get_rank() == 0:
    global_arr = np.zeros((5, 5))
else:
    global_arr = None
mpiarray = MpiArray.fromglobalarray(global_arr)

# load from local arrays
local_arr = np.zeros((5, 5))
mpiarray = MpiArray.fromlocalarrays(local_arr, axis=0)


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
* gather(root, delete_local)
    ** gathers the full array on MPI process with rank root.  
    ** delete_local=True removes the locally stored data if present.
    ** returns the full array to root process
* gatherall()
    ** returns the full array to every process.
    
Custom distributions can be used to specify how the data is distributed
across an axis to the MPI processes.  They need to provide the 
following:
* axis - (int) axis data is split on to give to different MPI processes
* sizes - (ndarray of int) size of slice on axis for each MPI process, 
    ordered by rank.  This includes any padding.
* offsets - (ndarray of int) offsets of slice on axis for each MPI 
    process, ordered by rank.  This includes any padding.
* unpadded_sizes - (ndarray of int) size of slice on axis for each MPI 
    process, ordered by rank.  This excludes any padding.
* unpadded_offsets - (ndarray of int) offsets of slice on axis for each
    MPI process, ordered by rank.  This excludes any padding.
NOTE: The unpadded data should have a one-to-one correspondence to a 
single MPI process (data should only be present in a single unpadded 
region and all data should be represented by the unpadded regions).  
"""

from mpi4py import MPI
import numpy as np
import logging

logger = logging.getLogger(__name__)

__version__ = "1.0"
__author__ = "Michael Sutherland"
__maintainer__ = "Michael Sutherland"
__email__ = "michael.sutherland@dmea.osd.mil"

__all__ = ["MpiArray", 
           "Distribution",
           "fromglobalarray",
           "fromlocalarrays"]


class Distribution(object):
    """Custom distribution for MpiArray."""

    def __init__(self, axis, sizes, offsets, unpadded_sizes, unpadded_offsets):
        self.axis = axis
        self.sizes = sizes
        self.offsets = offsets
        self.unpadded_sizes = unpadded_sizes
        self.unpadded_offsets = unpadded_offsets


    @staticmethod
    def fromdistribution(distribution):
        return Distribution(distribution.axis,
                            distribution.sizes,
                            distribution.offsets,
                            distribution.unpadded_sizes,
                            distribution.unpadded_offsets)


    @staticmethod
    def default(shape, mpi_size, axis=0, padding=0):
        """Creates a distribution of the data across an axis."""
        sizes, offsets = Distribution.split_array_indices(shape, mpi_size, axis, padding)
        unpadded_sizes, unpadded_offsets = Distribution.split_array_indices(shape, mpi_size, axis, 
                                                                             padding=0)
        return Distribution(axis, sizes, offsets, unpadded_sizes, unpadded_offsets)


    @staticmethod
    def split_array_indices(shape, mpi_size, axis=0, padding=0):
        """Calculate sizes and offsets to evenly split an array across 
        an axis with padding."""
        # nodes calculate offsets and sizes for sharing
        chunk_size = shape[axis] // mpi_size
        leftover = shape[axis] % mpi_size
        sizes = np.ones(mpi_size, dtype=np.int) * chunk_size
        # evenly distribute leftover across workers
        # NOTE: currently doesn't add leftover to rank 0, 
        # since rank 0 usually has extra work to perform already
        sizes[1:leftover+1] += 1
        offsets = np.zeros(mpi_size, dtype=np.int)
        offsets[1:] = np.cumsum(sizes)[:-1]
        # now compensate for padding
        upper_limit = offsets + sizes + padding
        np.clip(upper_limit, 0, shape[axis], upper_limit)
        offsets -= padding
        np.clip(offsets, 0, shape[axis], offsets)
        sizes = upper_limit - offsets
        return sizes, offsets


class MpiArray(object):
    
    def __init__(self, global_arr=None, local_arr=None, axis=0, padding=0, 
                 root=0, distribution=None, comm=None):
        """Suggested to use fromglobalarray or fromlocalarrays to 
        initialize MpiArray.  Initializing from a global array uses 
        global_arr, root, and comm.  Initializing from local arrays uses 
        local_arr, axis, padding, and comm.
        
        Parameters
        ----------
        global_arr : ndarray, optional
            Global array of full dataset loaded on a single MPI process. 
            Only used with global array loading.
        local_arr : ndarray, optional
            Local array of partial datasets loaded on each MPI process.
            These arrays should already be split on a an axis with
            padding.  Currently, no custom distributions are supported.
        axis : int, optional
            Specifies the axis used to load distributed data. Only used 
            with local array loading.  Defaults to zero.
        padding : int, optional
            Specifies the padding used to load distributed data.  This 
            is the overlap between the distributed data on each MPI 
            process.  Only used with local array loading. Defaults to 
            zero.
        distribution : Distribution, optional
            Class instance that sets how data is distributed.  
            Defines how the data is split along the axis among the 
            different MPI processes.  If not given, default Distribution 
            is constructed from axis and padding.
        root : int, optinal
            Rank of root node that has the global array of data.  Only 
            used with global array loading.  Defaults to zero.
        comm : MPI Comm object, optional
            Comm object that defines MPI nodes and is used for 
            communication.  If not provided, COMM_WORLD is used.
        
        Returns
        -------
        MpiArray
            MpiArray object returned to all MPI processes.
        """
        # initialize variables
        self.comm = comm or MPI.COMM_WORLD # MPI comm, used to send/recv
        self.global_arr = global_arr # full array, only stored on root mpi process when not distributed
        self.local_arr = local_arr # local_arr, split along self.axis, stored on every node, with padding
        self.distribution = distribution # class to specify how data distributed
        self.root = root # root mpi_rank that stores full global_arr
        self.shape = None # shape of the whole ndarray
        self.dtype = None # dtype of ndarray
        self._mpi_dtypes_subarray = {} # key is (shape, axis, size)
        # calculate parameters for distributed array or global array
        if self.local_arr is not None:
            if self.distribution is None:
                # calculate static distribution from current distribution
                # get sizes from overall distribution
                size = self.local_arr.shape[axis]
                sizes = np.array(self.comm.allgather(size))
                unpadded_sizes = sizes - padding
                unpadded_sizes[1:-1] -= padding #no padding on edges
                np.clip(unpadded_sizes, 0, size, unpadded_sizes)
                unpadded_offsets = np.zeros((self.mpi_size,), dtype=np.int)
                unpadded_offsets[1:] = np.cumsum(unpadded_sizes[:-1])
                offsets = unpadded_offsets
                offsets[1:] -= padding
                np.clip(offsets, 0, size, offsets)
                self.distribution = Distribution(axis, sizes, offsets, unpadded_sizes, 
                                                 unpadded_offsets)
            # calculate from local_arr sizes along axis
            total_axis_size = np.sum(self.unpadded_sizes)
            # use to calculate shape
            self.shape = self._tuple_replace(local_arr.shape, total_axis_size, self.axis)
            self.dtype = self.local_arr.dtype
        else:
            # take size from root rank that has array (usually zero) 
            if self.global_arr is not None:
                shape = global_arr.shape
                dtype = global_arr.dtype
            else:
                shape = None
                dtype = None
            self.shape = self.comm.bcast(shape, root=root)
            self.dtype = self.comm.bcast(dtype, root=root)


    def __del__(self):
        # free cached MPI dtypes on delete
        for mpi_dtype in self._mpi_dtypes_subarray.itervalues():
            mpi_dtype.Free()

    @property
    def mpi_rank(self):
        """Rank of this MPI process, using self.comm object"""
        return self.comm.Get_rank()
    
    @property
    def mpi_size(self):
        """Total number of MPI processes, using self.comm object"""
        return self.comm.Get_size()


    def is_root(self):
        """Returns true if we are the root node, only valid when global 
        array stored on single node."""
        return self.mpi_rank == self.root

    def is_distributed(self):
        """Returns true if data is distributed"""
        return self.distribution is not None 

    @property
    def ndim(self):
        """Total number of dimensions for full array"""
        return len(self.shape)
    
    @property
    def itemsize(self):
        """Length of data element"""
        return self.dtype.itemsize

    @property
    def axis(self):
        """Axis the array is split along."""
        if self.distribution is not None:
            return self.distribution.axis
        else:
            return None

    @property
    def sizes(self):
        """Sizes along axis for data distribution for all processes.
        This includes padding."""
        if self.distribution is not None:
            return self.distribution.sizes
        else:
            return None

    @property
    def size(self):
        """Size along axis for this process.  This includes padding."""
        if self.distribution is not None:
            return self.distribution.sizes[self.mpi_rank]
        else:
            return None

    @property
    def offsets(self):
        """Offsets relative to global array along axis for data 
        distribution for all processes.  This includes padding"""
        if self.distribution is not None:
            return self.distribution.offsets
        else:
            return None

    @property
    def offset(self):
        """Offset relative to global array along axis for data 
        distribution for this process.  This includes padding."""
        if self.distribution is not None:
            return self.distribution.offsets[self.mpi_rank]
        else:
            return None

    @property
    def unpadded_sizes(self):
        """Sizes along axis for data distribution for all processes.
        This excludes padding."""
        if self.distribution is not None:
            return self.distribution.unpadded_sizes
        else:
            return None        

    @property
    def unpadded_size(self):
        """Size along axis for this process.  This excludes padding."""
        if self.distribution is not None:
            return self.distribution.unpadded_sizes[self.mpi_rank]
        else:
            return None

    @property
    def unpadded_offsets(self):
        """Offsets relative to global array along axis for data 
        distribution for all processes.  This excludes padding"""
        if self.distribution is not None:
            return self.distribution.unpadded_offsets
        else:
            return None        
        
    @property
    def unpadded_offset(self):
        """Offset relative to global array along axis for data 
        distribution for this process.  This excludes padding."""
        if self.distribution is not None:
            return self.distribution.unpadded_offsets[self.mpi_rank]
        else:
            return None

    @property
    def unpadded_local_arr(self):
        """Returns local array with padding removed.  If necessary, a 
        copy is made to keep data contiguous in memory.""" 
        local_offset = self.unpadded_offset - self.offset
        axis_slc = np.s_[local_offset:local_offset + self.unpadded_size]
        unpadded_local_arr = self.local_arr[
            self._tuple_replace((np.s_[:],)*self.ndim, axis_slc, self.axis)]
        return np.require(unpadded_local_arr, requirements="C")

    @property
    def mpi_dtype(self):
        """MPI datatype of numpy array"""
        return self.numpy_to_mpi_dtype(self.dtype)

    @staticmethod
    def numpy_to_mpi_dtype(dtype):
        """Converts numpy type to MPI datatype"""
        return MPI._typedict[dtype.char]
    
    @staticmethod
    def fromglobalarray(global_arr, root=0, comm=None):
        """Initializes MpiArray object from a global ndarray on a 
        single MPI process.  Each process calls fromglobalarray and 
        receives an MpiArray object that represents the full array.
        
        Parameters
        ----------
        global_arr : ndarray
            Global array of full dataset loaded on a single MPI process.
            Other MPI processes should pass in None.
        root : int, optinal
            Rank of root node that has the global array of data. 
            Defaults to zero.
        comm : MPI Comm object, optional
            Comm object that defines MPI nodes and is used for 
            communication.  If not provided, COMM_WORLD is used.
        
        Returns
        -------
        MpiArray
            MpiArray object returned to all MPI processes.
        """        
        return MpiArray(global_arr, root=root, comm=comm)


    @staticmethod
    def fromlocalarrays(local_arr, axis=0, padding=0, distribution=None, comm=None):
        """Initializing MpiArray from local arrays.  The local arrays 
        must have been split along the given axis.  If distribution 
        object given, that is used to determine how data is split.
        Otherwise, the data must be split starting with rank zero and 
        continuing in order of rank, following the axis and padding 
        parameters.
        
        Parameters
        ----------
        local_arr : ndarray, optional
            Local array of partial datasets loaded on each MPI process.
            These arrays should already be split on a an axis with
            padding.  Currently, no custom distributions are supported.
        axis : int, optional
            Only used if distribution is None.  Specifies the axis used 
            to load distributed data. Defaults to zero.
        padding : int, optional
            Only used if distribution is None.  Specifies the padding 
            used to load distributed data.  This is the overlap 
            between the distributed data on each MPI process.  Defaults 
            to zero.
        distribution : Distribution, optional
            Class instance that sets how data is distributed.  
            Defines how the data is split along the axis among the 
            different MPI processes.  If not given, default Distribution 
            is constructed from axis and padding.            
        comm : MPI Comm object, optional
            Comm object that defines MPI nodes and is used for 
            communication.  If not provided, COMM_WORLD is used.
        
        Returns
        -------
        MpiArray
            MpiArray object returned to all MPI processes.
        """        
        return MpiArray(local_arr=local_arr, axis=axis, padding=padding, distribution=distribution, 
                        comm=comm)


    def copy(self, deep=True):
        """Create a copy of MpiArray.  If deep=True, the data is copied 
        as well.
        
        Parameters
        ----------
        deep : boolean, optional
            If true, the arrays are also copied.  Otherwise, each 
            MpiArray has references to the same data. 
        
        Returns
        -------
        MpiArray
            MpiArray object returned to all MPI processes.
        """
        if deep and self.global_arr is not None:
            global_arr = self.global_arr.copy()
        else:
            global_arr = self.global_arr
        if deep and self.local_arr is not None:
            local_arr = self.local_arr.copy()
        else:
            local_arr = self.local_arr
        return MpiArray(global_arr, local_arr, distribution=self.distribution, root=self.root, 
                        comm=self.comm)


    def scatter(self, axis=0, padding=0):
        """Scatter array evenly across axis with padding to all of the 
        MPI processes.  If data is already scattered, rescatters it in a 
        distributed manner.  Will replace any existing padding with new 
        padding from the unpadded region. 
        NOTE: global_arr deleted from root node if present.
        
        Parameters
        ----------
        axis : int, optional
            Specifies the axis used to distribute data. Defaults to 
            zero.
        padding : int, optional
            Specifies the padding used to load distributed data.  This 
            is the overlap between the distributed data on each MPI 
            process.  Defaults to zero.
        
        Returns
        -------
        ndarray
            Local array of data for each MPI process.
        """        
        # create Distribution from axis and padding
        distribution = Distribution.default(self.shape, self.mpi_size, axis, padding)
        return self.scatterv(distribution)
        

    def scatterv(self, distribution):
        """Scatter array using distribution to MPI processes.  If 
        data is already scattered, rescatters it in a distributed 
        manner. 
        NOTE: global_arr deleted from root node if present.
        
        Parameters
        ----------
        distribution : Distribution
            Specifies how to distribute data across an axis to MPI
            processes.
        
        Returns
        -------
        ndarray
            Local array of data for each MPI process.
        """        
        if not self.is_distributed():
            return self._scatterfromglobal(distribution)
        else:
            return self._rescatter(distribution)


    def _scatterfromglobal(self, distribution):
        """Scatter array from global array using distribution"""
        # root node has global_arr
        # scatter data to nodes on axis
        self.distribution = distribution
        self.local_arr = np.empty(self._calc_local_arr_shape(), dtype=self.dtype)
        # send to all nodes
        if self.is_root():
            reqs = []
            sizes = self.sizes # only calculate once
            offsets = self.offsets # only calculate once
            for i in range(self.mpi_size):
                mpi_dtype = self._mpi_dtype_subarray_axis(self.shape, self.axis, sizes[i])
                if mpi_dtype is None:
                    # nothing to send
                    continue
                reqs.append(self.comm.Isend([self.global_arr, 1, offsets[i], mpi_dtype], i))
        if self.local_arr.size: # something to receive
            self.comm.Recv(self.local_arr, source=self.root)
        if self.is_root():
            MPI.Request.Waitall(reqs)
        self.delete_global()
        return self.local_arr
    
    
    def _rescatter(self, distribution):
        if self.axis != distribution.axis:
            return self._rescatternewaxis(distribution)
        else:
            return self._rescattersameaxis(distribution)
    
    
    def _rescatternewaxis(self, distribution):
        """Redistribute data on a new axis.  Do NOT resend padded data.
        """
        # update values and reinitialize local_arr
        new_local_arr = np.empty(self._calc_local_arr_shape(distribution), dtype=self.dtype)
        mpi_dtypes = [] #store them to be deleted later
        requests = []
        for r in xrange(self.mpi_size):
            # do non-blocking sends
            subsize = list(self.local_arr.shape)
            subsize[distribution.axis] = distribution.sizes[r]
            subsize[self.axis] = self.unpadded_size
            if np.prod(subsize): # if we have something to send
                suboffset = [0] * self.ndim
                suboffset[distribution.axis] = distribution.offsets[r]
                suboffset[self.axis] = self.unpadded_offset - self.offset # remove padding
                mpi_dtype = self.mpi_dtype.Create_subarray(self.local_arr.shape, subsize, 
                                                           suboffset)
                mpi_dtype.Commit()
                requests.append(self.comm.Isend([self.local_arr, 1, 0, mpi_dtype], r))
                mpi_dtypes.append(mpi_dtype)
        for r in xrange(self.mpi_size):
            # do non-blocking recvs
            mpi_dtype = self._mpi_dtype_subarray_axis(new_local_arr.shape, self.axis, 
                                                      self.unpadded_sizes[r])
            if mpi_dtype is not None:
                requests.append(self.comm.Irecv(
                    [new_local_arr, 1, self.unpadded_offsets[r], mpi_dtype], r))
        MPI.Request.Waitall(requests)
        for mpi_dtype in mpi_dtypes:
            mpi_dtype.Free()
        # update to new distribution with new local_arr
        self.distribution = distribution
        self.local_arr = new_local_arr
        return self.local_arr
    
    
    def _rescattersameaxis(self, distribution):
        """Array already distributed on the correct axis, but have a 
        new distribution to match.  Will reset padding to come from
        unpadded areas."""
        
        # calculate overlap regions for each MPI process
        # only send from unpadded data
        # create a new array that will replace local_arr
        recv_local_arr = np.empty(self._calc_local_arr_shape(distribution), dtype=self.dtype) 
        requests = []
        for recv_rank in range(self.mpi_size):
            for send_rank in range(self.mpi_size):
                if self.mpi_rank in (recv_rank, send_rank):
                    # look for overlap_slices 
                    cur_offset = self.unpadded_offsets[send_rank]
                    cur_limit = cur_offset + self.unpadded_sizes[send_rank]
                    new_offset = distribution.offsets[recv_rank]
                    new_limit = new_offset + distribution.sizes[recv_rank]
                    if cur_offset < new_limit and new_offset < cur_limit:
                        # there is overlap, define where
                        overlap_offset = max(cur_offset, new_offset)
                        overlap_limit = min(cur_limit, new_limit)
                        overlap_size = overlap_limit - overlap_offset
                        if self.mpi_rank == send_rank:
                            # set up send
                            mpi_dtype = self._mpi_dtype_subarray_axis(
                                            self.local_arr.shape, self.axis, overlap_size)
                            if mpi_dtype is not None:
                                send_offset = overlap_offset - self.offsets[send_rank]
                                requests.append(self.comm.Isend(
                                    [self.local_arr, 1, send_offset, mpi_dtype], recv_rank))
                        if self.mpi_rank == recv_rank:
                            # set up recv
                            mpi_dtype = self._mpi_dtype_subarray_axis(
                                recv_local_arr.shape, self.axis, overlap_size)
                            if mpi_dtype is not None:
                                recv_offset = overlap_offset - new_offset
                                requests.append(self.comm.Irecv(
                                    [recv_local_arr, 1, recv_offset, mpi_dtype], send_rank))
        MPI.Request.Waitall(requests)
        self.local_arr = recv_local_arr
        self.distribution = distribution
        return self.local_arr
        

    def scattermovezero(self, axis=0, padding=0):
        """Move scatter axis to axis zero and scatter data evenly 
        across axis zero with padding to all of the MPI processes.  
        If data is already scattered, rescatters it in a distributed 
        manner.  This function changes the shape of MpiArray.  The 
        following give the same local_arr to each MPI process.
        
        mpiarray = fromglobalarray(arr)
        local_arr = mpiarray.scattermovezero(axis, padding)
        
        arr = np.moveaxis(arr, axis, 0)
        mpiarray = fromglobalarray(arr)
        local_arr = mpiarray.scatter(0, padding)

        NOTE: global_arr deleted from root node if present.        
        
        Parameters
        ----------
        axis : int, optional
            Specifies the axis used to distribute data. Defaults to 
            zero.
        padding : int, optional
            Specifies the padding used to load distributed data.  This 
            is the overlap between the distributed data on each MPI 
            process.  Defaults to zero.
        
        Returns
        -------
        ndarray
            Local array of data for each MPI process.
        """
        distribution = Distribution.default(self.shape, self.mpi_size, axis, padding)
        return self.scattervmovezero(distribution)


    def scattervmovezero(self, distribution):
        """Move scatter axis to axis zero and scatter data using
        distribution across axis zero to all of the MPI processes.  
        If data is already scattered, rescatters it in a distributed 
        manner.  This function changes the shape of MpiArray.  The 
        following give the same local_arr to each MPI process.
        
        mpiarray = fromglobalarray(arr)
        local_arr = mpiarray.scattervmovezero(distribution)
        
        arr = np.moveaxis(arr, distribution.axis, 0)
        mpiarray = fromglobalarray(arr)
        distribution.axis = 0
        local_arr = mpiarray.scatterv(distribution)

        NOTE: global_arr deleted from root node if present.        
        NOTE: self.distribution will be replaced with static 
        distribution with axis of zero. 
        
        Parameters
        ----------
        distribution : Distribution
            Specifies how to distribute data across an axis to MPI
            processes.
        
        Returns
        -------
        ndarray
            Local array of data for each MPI process.
        """         
        if distribution.axis == 0:
            # just do normal scatter
            return self.scatterv(distribution)
        
        # calculate new shape after scatter
        new_shape = list(self.shape)
        new_shape.insert(0, new_shape.pop(distribution.axis))
        new_shape = tuple(new_shape)
        # calculate new distribution on axis 0
        new_distribution = Distribution.fromdistribution(distribution)
        new_distribution.axis = 0
        
        if not self.is_distributed():
            # send global_arr from root and moveaxis to zero
            cur_shape = self.shape #store current shape for sending
            self.distribution = new_distribution
            self.shape = new_shape
            self.local_arr = np.empty(self._calc_local_arr_shape(), dtype=self.dtype)
            # create subarray for a single "col" of axis
            mpi_dtype = self._mpi_dtype_subarray_axis(cur_shape, axis=distribution.axis)
            if mpi_dtype is not None:
                self.comm.Scatterv([self.global_arr, self.sizes, self.offsets, mpi_dtype],
                                   [self.local_arr, self.mpi_dtype], root=self.root)
            self.delete_global()
        elif self.axis == distribution.axis:
            # data already distributed along axis
            # first move it to the correct orientation
            self.local_arr = np.moveaxis(self.local_arr, distribution.axis, 0)
            self.local_arr = np.require(self.local_arr, requirements="C")
            # now update local distribution to reflect axis change
            self.shape = new_shape
            self.distribution = Distribution.fromdistribution(self.distribution)
            self.distribution.axis = 0
            # now rescatter on same axis with new distribution
            self._rescattersameaxis(new_distribution)
        else:
            # data is distributed along self.axis
            # need to scatter along axis and move axis to axis 0
            new_local_arr = np.empty(self._calc_local_arr_shape(new_distribution, new_shape), 
                                     dtype=self.dtype)
            mpi_dtypes = [] #store them to be deleted later
            requests = []
            for r in xrange(self.mpi_size):
                # do non-blocking sends
                subsize = list(self.local_arr.shape)
                subsize[distribution.axis] = 1
                subsize[self.axis] = self.unpadded_size
                if np.prod(subsize) and new_distribution.sizes[r]: # if we have something to send
                    suboffset = [0] * self.ndim
                    suboffset[self.axis] = self.unpadded_offset - self.offset # remove padding
                    mpi_dtype = self.mpi_dtype.Create_subarray(self.local_arr.shape, subsize, 
                                                               suboffset)
                    mpi_dtype = mpi_dtype.Create_resized(
                                    0, int(np.prod(self.local_arr.shape[distribution.axis+1:])) 
                                    * self.dtype.itemsize)
                    mpi_dtype.Commit()
                    requests.append(self.comm.Isend(
                        [self.local_arr, distribution.sizes[r], distribution.offsets[r], mpi_dtype],
                        r))
                    mpi_dtypes.append(mpi_dtype)
            # calculate where the current axis moved to
            moved_axis = self.axis
            if self.axis < distribution.axis:
                moved_axis += 1
            for r in xrange(self.mpi_size):
                # do non-blocking recvs
                mpi_dtype = self._mpi_dtype_subarray_axis(new_local_arr.shape, moved_axis, 
                                                          self.unpadded_sizes[r])
                if mpi_dtype is not None:
                    requests.append(self.comm.Irecv(
                        [new_local_arr, 1, self.unpadded_offsets[r], mpi_dtype], r))
            MPI.Request.Waitall(requests)
            for mpi_dtype in mpi_dtypes:
                mpi_dtype.Free()
            self.local_arr = new_local_arr
            self.distribution = new_distribution
            self.shape = new_shape

        return self.local_arr


    def gather(self, root=0, delete_local=False):
        """Gather global array to MPI process with rank of root.
        
        Parameters
        ----------
        root : int, optional
            Rank of node receiving the global array.  Needs to be set 
            by every MPI process when calling gather.
        delete_local : bool, optional
            If False, MpiArray is not modified by calling gather.
            If True, MpiArray object will reset to a global mode, 
            similar to right after fromglobalarray where the whole 
            dataset is only stored on the root node.
        
        Returns
        -------
        ndarray
            Global array returned to MPI process with rank of root.  
            All other MPI processes are returned None.
        """         
        global_arr = None
        if not self.is_distributed():
            # array hasn't been distributed, check root
            if self.root == root:
                # array already in correct MPI process
                global_arr = self.global_arr
            else:
                # array not in correct MPI process
                if self.is_root():
                    # send data to new root node
                    self.comm.Send(self.global_arr, dest=root)
                elif self.mpi_rank == root:
                    # get data from old root node
                    global_arr = np.empty(self.shape, dtype=self.dtype)
                    self.comm.Recv(global_arr, source=self.root)
        else:
            # arr is distributed, gather from local_arrs
            if root == self.mpi_rank:
                global_arr = np.empty(self.shape, dtype=self.dtype)
                requests = []
                sizes = self.unpadded_sizes # only calculate once
                offsets = self.unpadded_offsets # only calculate once
                for i in range(self.mpi_size):
                    mpi_dtype = self._mpi_dtype_subarray_axis(size=sizes[i])
                    if mpi_dtype is not None:
                        requests.append(self.comm.Irecv([global_arr, 1, offsets[i], mpi_dtype], i))
            # each node sets up send, root node receives
            unpadded_local_arr = self.unpadded_local_arr
            if unpadded_local_arr.size: # check for something to send
                self.comm.Send([unpadded_local_arr, self.mpi_dtype], dest=root)
            if root == self.mpi_rank:
                MPI.Request.Waitall(requests)
        
        if delete_local:
            self.global_arr = global_arr
            self.root = root
            self.delete_local()
        return global_arr


    def allgather(self):
        """Gather global array to all MPI processes.
        
        Returns
        -------
        ndarray
            Global array returned to all MPI processes.
        """         
        global_arr = None
        if self.axis is not None and self.axis == 0:
            # arr is distributed on axis zero, use Allgatherv
            global_arr = np.empty(self.shape, dtype=self.dtype)
            # data already contiguous, can use Allgather
            row_size = int(np.prod(self.shape[1:]))
            recv_sizes = self.unpadded_sizes * row_size
            recv_offsets = self.unpadded_offsets * row_size
            self.comm.Allgatherv([self.unpadded_local_arr, self.mpi_dtype],
                                 [global_arr, recv_sizes, recv_offsets, self.mpi_dtype])
        else:
            if self.axis is not None:
                # data split along non-zero axis.  Needs a different datatype per node,
                # so Allgatherv can't be used.  Instead, gather to a single node and 
                # then Bcast to all nodes.  This should scale on large clusters, whereas 
                # a large number of sends would not.
                global_arr = self.gather(0)
                root = 0
            else:
                # data stored globally on root node
                root = self.root
                if self.is_root():
                    global_arr = self.global_arr
            if self.mpi_rank != root:
                # initialize global_arr on all non-root nodes
                global_arr = np.empty(self.shape, dtype=self.dtype)
            # broadcast global_arr from root node to non-root node
            self.comm.Bcast([global_arr, self.mpi_dtype], root=root)
        return global_arr


    def delete_local(self):
        """Delete local arrays"""
        self.local_arr = None
        self.padding = 0
        self.distribution = None


    def delete_global(self):
        """Delete global array"""
        self.global_arr = None

    
    def _calc_local_arr_shape(self, distribution=None, shape=None):
        """Calculate local array shape from global array shape and 
        distribution parameters."""
        if shape is None:
            shape = self.shape
        if distribution is None:
            distribution = self.distribution
        size = distribution.sizes[self.mpi_rank]
        return self._tuple_replace(tuple(shape), size, distribution.axis)


    def _mpi_dtype_subarray_axis(self, shape=None, axis=None, size=1):
        """Create MPI datatype for a subarray along a specific axis 
        with size.  Datatype is resized to a single instance of the 
        axis."""
        if shape is None:
            shape = self.shape
        if axis is None:
            axis = self.axis
        if not np.prod(shape) or not size:
            # this is a zero size subarray, just return None.
            return None
        mpi_dtype = self._mpi_dtypes_subarray.get((shape, axis, size))
        if mpi_dtype is None:
            subsizes = self._tuple_replace(shape, size, axis)
            starts = (0,)*self.ndim
            mpi_dtype = self.mpi_dtype.Create_subarray(shape, subsizes, starts)
            mpi_dtype = mpi_dtype.Create_resized(0, int(np.prod(shape[axis+1:]))*self.dtype.itemsize)
            mpi_dtype.Commit()
            self._mpi_dtypes_subarray[(shape, axis, size)] = mpi_dtype
        return mpi_dtype

    
    @staticmethod
    def _tuple_replace(t, value, index):
        """Replace element in tuple and return new tuple"""
        return t[:index] + (value,) + t[index+1:]


# create module level copies of loading functions
fromlocalarrays = MpiArray.fromlocalarrays
fromglobalarray = MpiArray.fromglobalarray 
