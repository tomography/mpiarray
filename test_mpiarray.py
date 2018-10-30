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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from mpi4py import MPI
from mpiarray import MpiArray, Distribution
import numpy as np
import unittest
from numpy.testing import assert_array_equal
import sys
import traceback

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

import logging
logging.basicConfig(format='%03d:'%(mpi_rank) + '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

# catch Python errors, log them, and force abort
def abort_mpi_on_error(exctype, value, tb):
    # log error and then properly shut down MPI
    logger.critical(''.join(traceback.format_tb(tb)))
    logger.critical('{0}: {1}'.format(exctype, value))
    MPI.COMM_WORLD.Abort(1)

sys.excepthook = abort_mpi_on_error


class TestMpiArray(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def slice_axis(arr, slc, axis):
        return arr[[slice(None) if i != axis else slc for i in range(arr.ndim)]]

    def check_fields(self, arr1, arr2):
        self.assertEqual(arr1.shape, arr2.shape, "shape mismatch")
        self.assertEqual(arr1.dtype, arr2.dtype, "dtype mismatch")
        self.assertEqual(arr1.ndim, arr2.ndim, "ndim mismatch")
        self.assertEqual(arr1.itemsize, arr2.itemsize, "itemsize mismatch")

    def calc_scatter(self, arr, axis=0, padding=0):
        # calculate expected scatter from arr
        # returns test_local_arr
        sizes, offsets = Distribution.split_array_indices(arr.shape, mpi_size, axis, padding)
        slc = [np.s_[:]]*arr.ndim
        slc[axis] = np.s_[offsets[mpi_rank]:offsets[mpi_rank]+sizes[mpi_rank]]
        return arr[tuple(slc)]


    def load_fromglobalarray(self, shape=(16, 16, 8)):
        # only rank zero provides the global_arr for loading
        if mpi_rank == 0:
        # load noncontiguous by default
            if len(shape) <= 1:
                # single dimension, just load it
                arr = np.random.rand(*shape)
            else:
                # swap axis 0 and 1 to make noncontiguous (unless axis has length of 1...)
                noncontiguous_shape = (shape[1], shape[0])
                if len(shape) > 2:
                    noncontiguous_shape += shape[2:]
                arr = np.random.rand(*noncontiguous_shape)
                arr = np.swapaxes(arr, 0, 1)
        else:
            arr = None
        mpiarray = MpiArray(arr)
        # share array to all MPI nodes
        arr = comm.bcast(arr)
        return arr, mpiarray


    def load_to_scattered(self, shape, axis=0, padding=0):
        # load array to scattered form.  This loads noncontiguous for testing purposes.
        arr, mpiarray = self.load_fromglobalarray(shape)
        local_arr = mpiarray.scatter(axis, padding=padding)
        if len(local_arr.shape) > 1:
            # swap axis(0,1), then make contiguous, then swap back
            new_local_arr = np.swapaxes(local_arr, 0, 1)
            new_local_arr = np.require(new_local_arr, requirements="C")
            new_local_arr = np.swapaxes(new_local_arr, 0, 1)
            assert_array_equal(local_arr, new_local_arr) #sanity check
            local_arr = new_local_arr
            mpiarray = MpiArray(local_arr, distribution=mpiarray.distribution)
        return arr, mpiarray


    def scatter_params(self, size=1):
        for padding in (0, 1, 2):
            for axis, shape in ((0, (mpi_size*size, 10)),
                                (0, (mpi_size*size,)),
                                (0, (mpi_size*size, 2, 2)), 
                                (0, (mpi_size*size, 1020, 4)),
                                (1, (5, mpi_size*size, 10)),
                                (1, (1, mpi_size*size, 2)),
                                (1, (24, mpi_size*size, 6)),
                                (2, (24, 10, mpi_size*size, 6)),
                                (2, (24, 12, mpi_size*size)),
                                (3, (24, 12, 6, mpi_size*size)),
                                # uneven splits
                                (0, (mpi_size*size+1, 10)),
                                (0, (mpi_size*size+2, 10)),
                                (0, (mpi_size*size+3, 10)),
                                (1, (3, 100, 7)),
                                (2, (10, 10, 1)),
                                ):
                yield shape, axis, padding


    def test_fromglobalarray(self):
        # load from array and check values
        for shape in [(10,),
                      (10, 4),
                      (16, 16, 8)]:
            arr, mpiarray = self.load_fromglobalarray(shape)
            self.check_fields(arr, mpiarray)
  
  
    def test_fromlocalarray(self):
        # load from local array and check values
        size = 7
        for padding in (0, 1, 2):
            shape = (mpi_size*size, 10, 8)
            arr = np.random.rand(*shape)
            local_arr = arr[max(0, mpi_rank*size-padding):(mpi_rank+1)*size+padding]
            mpiarray = MpiArray(local_arr, padding=padding)
            self.check_fields(arr, mpiarray)
 
 
    def test_rescatter(self):
        # load from array, scatter axis, and check global and local values
        size = 3
        for shape, axis, padding in self.scatter_params(size):
            arr, mpiarray = self.load_to_scattered(shape, axis, padding)
            for axis2 in range(arr.ndim):
                for padding2 in (0, 1, 2):
                    local_arr = mpiarray.scatter(axis2, padding2)
                    self.check_fields(arr, mpiarray)
                    test_local_arr = self.calc_scatter(arr, axis2, padding2)
                    assert_array_equal(test_local_arr, local_arr, "scatter axis: %d"%axis)
     
      
    def test_scatter(self):
        # load from array, scatter axis, and check global and local values
        size = 3
        for shape, axis, padding in self.scatter_params(size):
            arr, mpiarray = self.load_fromglobalarray(shape)
            local_arr = mpiarray.scatter(axis, padding=padding)
            self.check_fields(arr, mpiarray)
            test_local_arr = self.calc_scatter(arr, axis, padding)
            assert_array_equal(test_local_arr, local_arr, "scatter axis: %d"%axis)
   
   
    def test_gather_from_global(self):
        size = 3
        for shape, axis, _ in self.scatter_params(size):
            arr, mpiarray = self.load_fromglobalarray(shape)
            for rank in range(mpi_size):
                test_arr = mpiarray.gather(rank)
                self.check_fields(arr, mpiarray)
                if mpi_rank == rank:
                    self.check_fields(test_arr, mpiarray)
                    assert_array_equal(test_arr, arr, "gather axis: %d, rank: %d"%(axis, rank))
 
   
    def test_gather_from_scattered(self):
        # load from array, scatter axis, gather, check arr is the same
        size = 3
        for shape, axis, padding in self.scatter_params(size):
            arr, mpiarray = self.load_to_scattered(shape, axis, padding)
            for rank in range(mpi_size):
                test_arr = mpiarray.gather(rank)
                self.check_fields(arr, mpiarray)
                if rank == mpi_rank:
                    self.check_fields(test_arr, mpiarray)
                    assert_array_equal(test_arr, arr, "gather axis: %d, rank: %d"%(axis, rank))
                else:
                    assert test_arr is None
  
  
    def test_allgather(self):
        # load from array, allgather, scatter, allgather
        size = 3
        for shape, axis, padding in self.scatter_params(size):
            arr, mpiarray = self.load_fromglobalarray(shape)
            test_arr = mpiarray.allgather()
            self.check_fields(test_arr, mpiarray)
            assert_array_equal(test_arr, arr)
            mpiarray.scatter(axis, padding=padding)
            test_arr = mpiarray.allgather()
            self.check_fields(test_arr, mpiarray)
            assert_array_equal(test_arr, arr)
   
   
    def test_scattermovezero_global(self):
        # load from array, scattermovezero, test
        size = 3
        for shape, axis, padding in self.scatter_params(size):
            arr, mpiarray = self.load_fromglobalarray(shape)
            local_arr = mpiarray.scattermovezero(axis, padding=padding)
            arr = np.moveaxis(arr, axis, 0)
            self.check_fields(arr, mpiarray)
            test_local_arr = self.calc_scatter(arr, 0, padding)
            assert_array_equal(test_local_arr, local_arr, "scattermovezero axis: %d"%axis)

   
    def test_scattermovezero_local(self):
        # load from array, scatter, scattermovezero, test
        size = 3
        for shape, axis, padding in self.scatter_params(size):
            arr, mpiarray = self.load_to_scattered(shape, axis, padding)
            for axis2 in range(len(shape)-1, 0, -1):
                local_arr = mpiarray.scattermovezero(axis2, padding)
                arr = np.moveaxis(arr, axis2, 0)
                self.check_fields(arr, mpiarray)
                test_local_arr = self.calc_scatter(arr, 0, padding)
                assert_array_equal(test_local_arr, local_arr, "scattermove0zeroaxis: %d"%axis2)
    
    
    def test_copy(self):
        # load global_arr, copy, test, change val, test 
        # scatter, copy, test, change val, test
        for deep in (True, False):
            # load global_arr for global global_arr testing
            size = 5
            shape = (mpi_size*size, mpi_size*size, 8)
            arr, mpiarray = self.load_fromglobalarray(shape)
            # copy
            mpiarray2 = mpiarray.copy(deep)
            # test
            self.check_fields(arr, mpiarray)
            self.check_fields(mpiarray, mpiarray2)
            assert_array_equal(mpiarray.arr, mpiarray2.arr)
            gathered_arr = mpiarray2.gather(0)
            if mpi_rank == 0:
                assert_array_equal(arr, gathered_arr)
                # change value in mpiarray2 to check if deep copy
                self.assertEqual(mpiarray.arr[0, 0, 0], mpiarray2.arr[0, 0, 0])
                mpiarray2.arr[0,0,0] += 1
                # test
                if deep:
                    self.assertNotEqual(mpiarray.arr[0, 0, 0], mpiarray2.arr[0, 0, 0])
                else:
                    self.assertEqual(mpiarray.arr[0, 0, 0], mpiarray2.arr[0, 0, 0])
            del mpiarray2
            # scatter tests
            for padding in (0, 1, 2):
                for axis in (0, 1):
                    arr, mpiarray = self.load_fromglobalarray(shape)
                    mpiarray.scatter(axis, padding)
                    mpiarray2 = mpiarray.copy(deep)
                    # test values
                    self.check_fields(arr, mpiarray)
                    self.check_fields(mpiarray, mpiarray2)
                    test_local_arr = self.calc_scatter(arr, axis, padding)
                    assert_array_equal(test_local_arr, mpiarray.arr, "scatter axis: %d"%axis)
                    assert_array_equal(test_local_arr, mpiarray2.arr, "scatter axis: %d"%axis)
                    # change val and test again
                    self.assertEqual(mpiarray.arr[0, 0, 0], mpiarray2.arr[0, 0, 0])
                    mpiarray2.arr[0, 0, 0] += 1
                    if deep:
                        self.assertNotEqual(mpiarray.arr[0, 0, 0], mpiarray2.arr[0, 0, 0])
                    else:
                        self.assertEqual(mpiarray.arr[0, 0, 0], mpiarray2.arr[0, 0, 0])
                    mpiarray2.arr[0, 0, 0] = mpiarray.arr[0, 0, 0] #reset value


if __name__ == "__main__":
    unittest.main()


