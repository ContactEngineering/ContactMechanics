#
# Copyright 2019 Antoine Sanner
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
import numpy as np

from PyCo.Topography import make_sphere

##########
# substrate
#
# FrobeniusNorm = np.linalg.norm()
# for i in range(maxiter):
#     beta=max(float(i-1)/(i-2))
#
#     s =




def fro_fourier():

    mat = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]], dtype= float)
    # Parseval's Theorem
    assert np.sum(mat**2) == np.sum(np.absolute(np.fft.fftn(mat))**2) / np.prod(mat.shape)

    assert np.linalg.norm(mat,  ord = "fro") == np.linalg.norm(np.fft.fftn(mat) ,  ord = "fro") / np.sqrt(np.prod(mat.shape))


def iweights_frobenius_norm():
    "compare with the calculation in the real space"

    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace

    nx, ny = 16,16
    sx, sy = 16,16

    hs = PeriodicFFTElasticHalfSpace((nx, ny), young = 1.,size= (sx,sy) )

    realiweights = np.fft.irfftn(hs.weights)

    realnorm = np.linalg.norm(realiweights, ord="fro") * np.sqrt(np.prod(hs.domain_resolution))

    fouriernorm = hs.weights_frobenius_norm()
    assert fouriernorm == realnorm, "fourier: {}, real: {}".format(fouriernorm, realnorm)

def accelerated_gradient_projection(substrate, topography, offset, initialforces = None, maxiter = 100, callback = lambda it, p, d : None):

    heights = topography.heights() + offset

    # for i in len():
    #
    #print(substrate.iweights.shape)
    Frobenius =  substrate.weights_frobenius_norm()
    print("Frobenius {}".format(Frobenius))
    if initialforces is not None:
        p = initialforces
    else:
        p = np.zeros(substrate.domain_resolution)
    pold = p

    for i in range(maxiter):

        beta = max((i-1.)/(i+2.),0)
        s = p + beta * (p - pold)
        gap = - substrate.evaluate_disp(s) - heights

        pold = p#.copy()

        #print(p)
        #project back onto feasible set
        p = np.maximum(s - gap / Frobenius, 0 )

        A_contact= np.sum((p > 0) * 1.)

        d = dict(max_penetration= np.max(-gap),
                 fractional_area = np.float64(A_contact/ np.prod(topography.resolution)).item())
        callback(i, p, d)

    return p, gap



if __name__ == "__main__":


    ### Run some tests
    fro_fourier()
    iweights_frobenius_norm()

    ###
    import sys

    sys.path.append('/Users/antoines')

    import numpy as np
    import matplotlib.pyplot as plt

    from make_rough import Fourier_synthesis
    from PyCo.Topography import Topography
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace

    nx, ny = 64, 64
    sx = float(nx)
    sy = float(ny)
    dx = 1.
    dy = 1.

    surface, surfaceq = Fourier_synthesis(nx, ny, sx, sy, Hurst=0.8,
                                          rms_height=None,
                                          rms_slope=0.1,
                                          short_cutoff=4 * dx,
                                          long_cutoff=sy / 4,
                                          rolloff=1)

    topography = Topography(np.asarray(surface), size=(sx, sy))

    hs = PeriodicFFTElasticHalfSpace((nx, ny), young=1., size=(sx, sy))

    fig, ax = plt.subplots()
    ax.set_yscale("log")


    def callback(it, p, d):
        # print("maxpen = {} ".format(np.max(-gap)))
        # print(p)
        ax.plot(it, d["max_penetration"], "^r")


    p, gap = accelerated_gradient_projection(hs, topography, offset=0,
                                             maxiter=100, callback=callback)

    fig2, (ax2,ax2p) = plt.subplots(1,2)
    ax2.set_aspect(1)
    ax2p.set_aspect(1)
    plt.colorbar(ax2.pcolormesh(gap), ax = ax2)
    plt.colorbar(ax2p.pcolormesh(p), ax = ax2p)