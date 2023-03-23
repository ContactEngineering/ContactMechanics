# %% [markdown]
# # Decoupling periodic images 

# %% [markdown]
# Here I demonstrate that our FreeFFTElasticHalfSpace yields solutions which are independent of the size of the domain

# %% [markdown]
# ## Nondimensionalisation 
#
# in these units, 
#
# Penetration
# $b = a^2$ 
#
# Force
# $F = a^3$
#
# Max. pressure:
# $p_0 = \frac{3}{2 \pi} a = \frac{3}{2 \pi} F^{1/3} = \frac{3}{2 \pi} \sqrt{b}$

# %%
R = 1
Es = 3 / 4

# %%

# %%
# %pylab inline
from SurfaceTopography import make_sphere
from ContactMechanics.Factory import make_system
from ContactMechanics import FreeFFTElasticHalfSpace

dx = 0.08
nx = 32
sx = nx * dx

topography = make_sphere(radius=R, nb_grid_pts=(nx, nx), physical_sizes=(nx * dx, nx * dx), kind="paraboloid")

substrate = FreeFFTElasticHalfSpace(topography.nb_grid_pts, young=Es, physical_sizes=topography.physical_sizes)
system = make_system(substrate=substrate, surface=topography)
penetration = 1
sol = system.minimize_proxy(offset=penetration)

nx_l = 64
sx_l = nx_l * dx
topography_l = make_sphere(radius=R, nb_grid_pts=(nx_l, nx_l), physical_sizes=(sx_l, sx_l), kind="paraboloid")

substrate_l = FreeFFTElasticHalfSpace(topography_l.nb_grid_pts, Es, physical_sizes=topography_l.physical_sizes)
system_l = make_system(substrate=substrate_l, surface=topography_l)

sol_l = system_l.minimize_proxy(offset=penetration)

# %%

fig, ax = plt.subplots()

plt.colorbar(ax.imshow(system.disp, extent=[0, 2, 0, 2]), label="displacement")

ax.set_xlabel("$x / s_x$")
ax.set_ylabel("$y / s_y$")

# %%

fig, ax = plt.subplots()

plt.colorbar(ax.imshow(system.force / system.area_per_pt, extent=[0, 1, 0, 1]), label="pressure")

ax.set_xlabel("$x / s_x$")
ax.set_ylabel("$y / s_y$")

# %% [markdown]
# ## Comparing the solutions on the large and the small grid

# %% [markdown]
# ### Displacements

# %% [markdown]
# The black lines represent the domain without the padding for the small grid

# %%
fig, ax = plt.subplots()

plt.plot(np.arange(2 * nx_l) * dx - sx_l / 2, system_l.disp[:, nx_l // 2])
plt.plot(np.arange(2 * nx) * dx - sx / 2, system.disp[:, nx // 2])

ax.set_xlabel("$x$")
ax.set_ylabel("displacement")

ax.axvline(sx - sx / 2, c="k")
ax.axvline(0 - sx / 2, c="k")

# %%
fig, ax = plt.subplots()

plt.plot(np.arange(nx_l) * dx - sx_l / 2, system_l.force[:, nx_l // 2] / system_l.area_per_pt)
plt.plot(np.arange(nx) * dx - sx / 2, system.force[:, nx // 2] / system.area_per_pt)

ax.set_xlabel("$x$")
ax.set_ylabel("pressures")

ax.axvline(sx - sx / 2, c="k")
ax.axvline(0 - sx / 2, c="k")

# %%
fig, ax = plt.subplots()

plt.plot(np.arange(nx_l) * dx - sx_l / 2, system_l.force[:, nx_l // 2] / system_l.area_per_pt, ".")
plt.plot(np.arange(nx) * dx - sx / 2, system.force[:, nx // 2] / system.area_per_pt, "+")

ax.set_xlabel("$x $")
ax.set_ylabel("pressures")

r = np.linspace(0, 1, 200)
a = np.sqrt(penetration)

ax.plot(r * a, 3 / (2 * np.pi) * a * np.sqrt(1 - r ** 2), "-k", label="Hertz")

ax.set_xlim(0.6, 1.05)
ax.legend()

# %%

# %%
fig, ax = plt.subplots()

plt.plot(np.arange(nx) * dx - sx / 2,
         system_l.force[nx_l // 4:(3 * nx_l // 4), nx_l // 2] / system_l.area_per_pt - system.force[:,
                                                                                       nx // 2] / system.area_per_pt,
         ".")

ax.set_xlabel("$x $")
ax.set_ylabel("pressure difference")

r = np.linspace(0, 1, 200)
a = np.sqrt(penetration)

# ax.plot(r * a, 3 / (2 * np.pi) * a * np.sqrt(1 - r**2), "-k", label="Hertz")

ax.set_xlim(0.6, 1.05)
ax.legend()

# %% [markdown]
# Grid Refinement

# %%
dx_f = 0.01
nx_f = 256
sx_f = nx_f * dx_f
topography_f = make_sphere(radius=R, nb_grid_pts=(nx_f, nx_f), physical_sizes=(sx_f, sx_f), kind="paraboloid")

substrate_f = FreeFFTElasticHalfSpace(topography_f.nb_grid_pts, Es, physical_sizes=topography_f.physical_sizes)
system_f = make_system(substrate=substrate_f, surface=topography_f)

sol_f = system_f.minimize_proxy(offset=penetration)

# %%
fig, ax = plt.subplots()

plt.plot(np.arange(nx_f) * dx_f - sx_f / 2, system_f.force[:, nx_f // 2] / system_f.area_per_pt, ".")
plt.plot(np.arange(nx) * dx - sx / 2, system.force[:, nx // 2] / system.area_per_pt, "+")

ax.set_xlabel("$x $")
ax.set_ylabel("pressures")

r = np.linspace(0, 1, 200)
a = np.sqrt(penetration)

ax.plot(r * a, 3 / (2 * np.pi) * a * np.sqrt(1 - r ** 2), "-k", label="Hertz")

ax.set_xlim(0.6, 1.05)
