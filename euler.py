import ufl
from ufl import dx, dS, FacetNormal, dot
from mpi4py import MPI
from dolfinx import mesh, fem, io
import numpy as np
from dolfinx.fem.petsc import NonlinearProblem
from meshing import create_naca_4digit, create_naca_4digit_3d


def conserved_vars(U):
    rho = U[0]
    rho_u = ufl.as_vector([U[i] for i in range(1, dim + 1)])
    rho_E = U[dim + 1]

    return rho, rho_u, rho_E


def flow_vars(U):
    rho, rho_u, rho_E = conserved_vars(U)
    return rho, rho_u / rho, rho_E / rho


def pressure(U, gamma):
    rho, u, E = flow_vars(U)
    return (gamma - 1.0) * rho * (E - 0.5 * dot(u, u))


def speed_of_sound(p, rho, gamma):
    return (gamma * p / rho)**0.5


def enthalpy(U, gamma):
    rho, u, E = flow_vars(U)
    p = pressure(U, gamma)
    return E + p / rho


def flux(U):
    rho, u, E = flow_vars(U)
    p = pressure(U, gamma=gamma)
    H = enthalpy(U, gamma=gamma)

    # See Ern III (80.9)
    inertia = rho * ufl.outer(u, u) + p * ufl.Identity(dim)
    res = ufl.as_tensor([rho * u,
                         *[inertia[d, :] for d in range(dim)],
                         rho * H * u])
    return res


def phi(n_F, U_m, U_p, alpha):
    # See Di Pietro and Ern, Equation (3.76) and (3.77)
    # NOTE: Using their terminology for U^+ and U^-, which is opposite to FEniCS
    # U("+") and U("-")
    return 1 / 2 * (dot(flux(U_m), n_F) + dot(flux(U_p), n_F)) \
        + alpha / 2 * (U_m - U_p)


gamma = 1.4
comm = MPI.COMM_WORLD

# msh, ft, boundaries = create_naca_4digit(
#     comm, h_near=0.01, h_far=0.2, r_near=0.8, r_far=2.5)

msh, ft, boundaries = create_naca_4digit_3d(comm)

dim = msh.topology.dim
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

V = fem.functionspace(msh, ("Discontinuous Lagrange", 0, (dim + 2,)))

U = fem.Function(V)
v = ufl.TestFunction(V)

n = FacetNormal(msh)

rho, u, E = flow_vars(U)
p = pressure(U, gamma=gamma)
c = speed_of_sound(p, rho, gamma=gamma)

F = - ufl.inner(flux(U), ufl.grad(v)) * dx

# Lax-Friedrichs parameter. We always use n("+"), since this is the normal
# we are associating with the facet (n_F in Di Pietro and Ern).
alpha = ufl.max_value(abs(dot(u("+"), n("+"))) + c("+"),
                      abs(dot(u("-"), n("+"))) + c("-"))
F += ufl.inner(phi(n("+"), U("+"), U("-"), alpha),
               v("+") - v("-")) * dS

# Inlet
R = 287.0
T_0 = 300.0
p_0 = 101325.0
rho_0 = p_0 / (R * T_0)
M = fem.Constant(msh, 0.3)
if M.value < 1.0:
    # Subsonic inlet
    c_inf = (gamma * R * T_0)**0.5
    u_inf = ufl.as_vector([M * c_inf] + [0.0] * (dim - 1))
    E_inf = p/((gamma-1)*rho_0) + 0.5 * dot(u_inf, u_inf)
    U_star_in = ufl.as_vector([rho_0] + [rho_0 * u_inf[i]
                              for i in range(dim)] + [rho_0 * E_inf])

    c_star_in = ufl.sqrt(gamma * p / rho_0)
    alpha_in = ufl.max_value(
        abs(dot(u, n)) + c, abs(dot(u_inf, n)) + c_star_in)
else:
    # Supersonic inlet
    c_inf = ufl.sqrt(gamma*p_0/rho_0)
    u_inf = ufl.as_vector([M * c_inf] + [0.0] * (dim - 1))
    E_in = p_0/((gamma-1.0)*rho_0) + 0.5 * dot(u_inf, u_inf)
    U_star_in = ufl.as_vector([rho_0] + [rho_0 * u_inf[i]
                              for i in range(dim)] + [rho_0 * E_in])
    alpha_in = ufl.max_value(abs(dot(u, n)) + c,
                             abs(dot(u_inf, n)) + c_inf)


F += ufl.inner(phi(n, U, U_star_in, alpha_in), v) * ds(boundaries["inlet"])

# Outlet
if M.value < 1.0:
    p_out = 100000.0
    E_out = p_out/((gamma-1)*rho) + 0.5 * dot(u, u)
    U_star_out = ufl.as_vector([rho] + [rho * u[i]
                               for i in range(dim)] + [rho*E_out])
    c_star_out = ufl.sqrt(gamma * p_out / rho)
    alpha_out = ufl.max_value(abs(dot(u, n)) + c, abs(dot(u, n)) + c_star_out)
else:
    alpha_out = abs(dot(u, n)) + c
    U_star_out = U

F += ufl.inner(phi(n, U, U_star_out, alpha_out), v) * ds(boundaries["outlet"])

u_ref = u - 2.0*dot(u, n)*n
U_star_wall = ufl.as_vector(
    [rho] + [rho * u_ref[i] for i in range(dim)] + [rho*E])
alpha_wall = abs(dot(u, n)) + c
F += ufl.inner(phi(n, U, U_star_wall, alpha_wall), v) * \
    ds((boundaries["walls"], boundaries["airfoil"]))

residual = F  # - ufl.inner(f, v) * dx

c0 = np.sqrt(gamma*R*T_0)
u0 = M.value * c0
rho0 = p_0/(R*T_0)
E0 = p_0/((gamma-1)*rho0) + 0.5*u0**2
U.interpolate(lambda x: np.array([
    rho0 * np.ones_like(x[0]),
    rho0 * u0 * np.ones_like(x[0]),
    *([rho0 * 0.0 * np.ones_like(x[0])] * (dim-1)),
    rho0 * E0 * np.ones_like(x[0])
]))

petsc_options = {
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
    "snes_monitor": None
}
problem = NonlinearProblem(
    residual, U, petsc_options_prefix="euler_", petsc_options=petsc_options
)

U = problem.solve()
converged_reason = problem.solver.getConvergedReason()
assert converged_reason > 0

V_vis = fem.functionspace(msh, ("Discontinuous Lagrange", 1))
V_vis_vec = fem.functionspace(msh, ("Discontinuous Lagrange", 1, (dim,)))

# Write to file
for name, var, space in (("rho", rho, V_vis), ("p", p, V_vis), ("u", u, V_vis_vec)):
    expr = fem.Expression(var, space.element.interpolation_points)
    f = fem.Function(space)
    f.interpolate(expr)
    with io.VTXWriter(msh.comm, f"{name}.bp", f) as file:
        file.write(0.0)

with io.XDMFFile(comm, "aerofoil_mesh.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_meshtags(ft, msh.geometry)

# error = U - g_d_expr(x)
# e_L2 = (msh.comm.allreduce(fem.assemble_scalar(
#     fem.form(ufl.inner(error, error) * ufl.dx)), op=MPI.SUM))**0.5

# print(f"L2 error: {e_L2}")
