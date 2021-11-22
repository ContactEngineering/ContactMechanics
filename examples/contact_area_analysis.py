import numpy as np
import io
import matplotlib.pyplot as plt
import SurfaceTopography.Uniform.GeometryAnalysis as CAA

with io.StringIO(
        """
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
        0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0
        1 1 0 0 1 1 0 0 0 0 0 0 1 1 0 1 1 0 0 0 0 0
        0 1 1 0 1 1 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
        0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
        0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        """) as file:
    contacting_points = np.loadtxt(file)

nx, ny = contacting_points.shape

x, y = np.mgrid[:nx, :ny]

fig, ax = plt.subplots()
ax.imshow(contacting_points.T, cmap="Greys")
iper = CAA.inner_perimeter_area(contacting_points, True, stencil=CAA.nn_stencil)
ax.plot(x[iper], y[iper], ".r", label="inner_perimeter, nn")
iper = CAA.inner_perimeter_area(contacting_points, True, stencil=CAA.nnn_stencil)
ax.plot(x[iper], y[iper], "xr", label="inner_perimeter, nnn")

oper = CAA.outer_perimeter_area(contacting_points, True, stencil=CAA.nn_stencil)
ax.plot(x[oper], y[oper], "ob", mfc="none", label="outer_perimeter, nn")
oper = CAA.outer_perimeter_area(contacting_points, True, stencil=CAA.nnn_stencil)
ax.plot(x[oper], y[oper], "+b", label="outer_perimeter, nnn")

ax.legend()

fig.savefig("caa.pdf")
