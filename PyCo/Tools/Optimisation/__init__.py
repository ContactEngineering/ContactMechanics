#
# Copyright 2017, 2019 Lars Pastewka
#           2016 Till Junge
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
optimisation helpers in PyCo
"""

from .common import ReachedTolerance, ReachedMaxiter, FailedIterate
from .common import ReachedMaxiterWarning
from .common import intersection_confidence_region
from .common import dogleg
from .common import steihaug_toint
from .common import modified_cholesky
from .common import first_wolfe_condition
from .common import second_wolfe_condition
from .common import line_search
from .common import construct_augmented_lagrangian
from .common import construct_augm_lag_grad
from .common import construct_augm_lag_hess
from .AugmentedLagrangian import augmented_lagrangian
from .ConstrainedConjugateGradients import constrained_conjugate_gradients
from .NewtonLineSearch import newton_linesearch
from .NewtonConfidenceRegion import newton_confidence_region
from .SimpleRelaxation import simple_relaxation
