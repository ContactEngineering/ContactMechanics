import PyCo
import importlib
importlib.reload(PyCo)

import numpy as np
import matplotlib.pyplot as plt 
#from PyCo.Topography import read_topography
from PyCo.Topography import open_topography 
#from PyCo.Topography import Topography

#plt.ion()                                                                                                                                                                                                  
#fn='../data/issues/230/di1.txt'                                                                                                                                                                            
#fn='tests/file_format_examples/opdx2.OPDx'
#fn='tests/file_format_examples/example.opd'
#fn='tests/file_format_examples/example2.x3p'
# fn='tests/file_format_examples/mi1.mi'

def plot(fn):
    r=open_topography(fn)

    t=r.topography()
    if 'unit' in t.info:
        unit = t.info['unit']
    else:
        unit = '?'
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(f"{fn}, channel {r.default_channel.name}")

    ax = fig.add_subplot(2,2,1)
    ax.set_title("pcolormesh(t.heights().T)")
    ax.pcolormesh(t.heights().T)

    ax = fig.add_subplot(2,2,2)
    ax.set_title("pcolormesh(*t.positions_and_heights())")
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    ax.pcolormesh(*t.positions_and_heights())

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("above is correct, if this is like Gwyddion")
    ax.pcolormesh(np.flipud(t.heights().T))

    ax = fig.add_subplot(2,2,4)
    ax.set_title("imshow(t.heights().T)")
    extent = (
        0, t.physical_sizes[0], t.physical_sizes[1], 0
    )
    ax.imshow(t.heights().T, extent=extent)
    fig.subplots_adjust(hspace=0.5)
    fig.show()

    h = t.heights()
    for i,j in [(0,0), (0,-1), (-1,0), (-1,-1)]:
        print(f"h[{i},{j}] == {h[i,j]}")

    return t

if __name__== '__main__':

    plt.close('all')
    # plt.ion()

    filenames = [
        #'../data/issues/230/di1.txt',
        #'tests/file_format_examples/di1.di',
        #'tests/file_format_examples/opdx2.OPDx',
        # 'tests/file_format_examples/example.opd',
        # 'tests/file_format_examples/example2.x3p',
        #'tests/file_format_examples/mi1.mi',
        'tests/file_format_examples/example.ibw',
    ]

    for fn in filenames:
        t = plot(fn)

    input("Press enter to proceed - last topography in variable 't'")


