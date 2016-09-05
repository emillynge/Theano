from __future__ import absolute_import, division, print_function

import unittest
import numpy
import theano

from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
from theano.misc.pycuda_init import pycuda_available
from theano.sandbox.cuda.cusolver import cusolver_available

from theano.sandbox.cuda import cusolver

if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda not available')
if not pycuda_available:
    raise SkipTest('Optional package pycuda not available')
if not cusolver_available:
    raise SkipTest('Optional package scikits.cuda.cusolver not available')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestCusolver(unittest.TestCase):
    def run_gpu_solve(self, A_val, x_val):
        b_val = numpy.dot(A_val, x_val)
        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")

        solver = cusolver.gpu_solve(A, b)
        fn = theano.function([A, b], [solver])
        res = fn(A_val, b_val)
        x_res = numpy.array(res[0])
        utt.assert_allclose(x_res, x_val)

    def run_gpu_chol_factor(self, A_val):
        A = theano.tensor.matrix("A", dtype="float32")
        from scipy.linalg import cholesky
        solver = cusolver.gpu_chol_factor(A)
        fn = theano.function([A], [solver])
        res = fn(A_val)
        R_res = numpy.array(res[0])
        A_res = R_res @ R_res.T
        utt.assert_allclose(A_val, A_res)

    def test_diag_solve(self):
        numpy.random.seed(1)
        A_val = numpy.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype="float32")
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym = (A_val + A_val.T) / 2.0
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_sym, x_val)

    def test_orth_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_orth = numpy.linalg.svd(A_val)[0]
        x_val = numpy.random.uniform(-0.4, 0.4, (A_orth.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_orth, x_val)

    def test_uni_rand_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     (A_val.shape[1], 4)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_chol_factor(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym =((A_val + A_val.T)  + numpy.eye(5) * 5).astype("float32")


        self.run_gpu_chol_factor(A_sym)