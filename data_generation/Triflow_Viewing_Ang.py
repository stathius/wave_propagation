import triflow as trf
import numpy as np
import scipy.signal
import os
from PIL import Image
import random
import time
import copy


"""
Wilhelm Sorteberg, 2018
wilhelm@sorteberg.eu


"""




def hillshade(array, azimuth, angle_altitude):

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.


    shaded = np.sin(altituderad) * np.sin(slope)\
     + np.cos(altituderad) * np.cos(slope)\
     * np.cos(azimuthrad - aspect)
    the_range = np.sort(np.reshape(shaded, -1))
    minimum = the_range[int(0.005 * len(the_range))]
    maximum = the_range[int(0.995 * len(the_range))]
    norm_shaded = (shaded - minimum) / (maximum - minimum)
    norm_shaded = np.clip(norm_shaded, 0, 1)
    return 255 * norm_shaded


class NonConservative_ShallowWater:
    @staticmethod
    def F(fields, pars):
        Ffields = fields.copy()
        x = fields["x"].values
        h = fields["h"].values
        u = fields["u"].values
        v = fields["v"].values
        H = fields["H"].values

        delta_x = x.ptp() / (x.size - 1)
        delta_y = y.ptp() / (y.size - 1)

        def dx(U):
            return (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2 * delta_x)
        def dy(U):
            return (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2 * delta_y)

        def dxx(U):
            return (np.roll(U, 1, axis=1) - 2 * U + np.roll(U, -1, axis=1)) / (delta_x**2)
        def dyy(U):
            return (np.roll(U, 1, axis=0) - 2 * U + np.roll(U, -1, axis=0)) / (delta_y**2)

        eta = h + H
        visc = lambda var: pars["nu"] * (dxx(var) + dyy(var))
        dth = -(dx(u * eta) + dy(v * eta))
        dtu = -(u * dx(u) + v * dy(u)) + pars["f"] * v - 9.81 * dx(h) + visc(u)
        dtv = -(u * dx(v) + v * dy(v)) - pars["f"] * u - 9.81 * dy(h) + visc(v)

        Ffields["h"][:] = dth
        Ffields["u"][:] = dtu
        Ffields["v"][:] = dtv
        return Ffields.uflat
    _indep_vars = ["x", "y"]
    fields_template = trf.core.fields.BaseFields.factory(("x", "y"),
                                                         [("h", ("x", "y")),
                                                          ("u", ("x", "y")),
                                                          ("v", ("x", "y"))],
                                                         [("H", ("x", "y"))])


model = NonConservative_ShallowWater()

def solid_wall(t, fields, pars):
    fields["u"][:, 0] = 0
    fields["u"][:, -1] = 0
    fields["u"][0, :] = 0
    fields["u"][-1, :] = 0
    fields["v"][:, 0] = 0
    fields["v"][:, -1] = 0
    fields["v"][0, :] = 0
    fields["v"][-1, :] = 0
    fields["h"][:, 0] = copy.copy(fields["h"][:, 1])
    fields["h"][:, -1] = copy.copy(fields["h"][:, -2])
    fields["h"][0, :] = copy.copy(fields["h"][1, :])
    fields["h"][-1, :] = copy.copy(fields["h"][-2, :])
    return fields, pars


plot = False

def constrict(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val

location = "./Viewing_Ang_50_Data"
if not os.path.isdir(location):
    os.mkdir(location)

data_points = 500
os.listdir()

while len(os.listdir(location)) < data_points:
    Nx = Ny = 184
    MaxRoll = Nx / 2 - 10
    size = random.uniform(10, 20)
    x_roll = random.randint(-MaxRoll, MaxRoll)
    y_roll = random.randint(-MaxRoll, MaxRoll)
    x = np.linspace(0, size, Nx)           # Physical domain
    y = np.linspace(0, size, Ny)           # Physical domain
    u = np.zeros((Nx, Ny))          # x velocity
    v = np.zeros((Nx, Ny))          # y velocity    #Can add +2 to add a general underlying velocity
    h = np.zeros((3 * Nx, 3 * Ny)) + (scipy.signal.windows.gaussian(3 * Nx, 5) * scipy.signal.windows.gaussian(3 * Ny, 5)[:, None]) * 1
    h = np.roll(h, x_roll, axis=1)        # Drop x position
    h = np.roll(h, y_roll, axis=0)        # Drop Y position
    h = h[Nx: 2 * Nx, Ny: 2 * Ny]
    H = np.ones((Nx, Ny)) * 10
    init_fields = model.fields_template(x=x, y=y, h=h, u=u, v=v, H=H)
    pars = {"f": 0, "nu": 1E-6}  # no coriolis effect, water viscosity #Parameters
    TIME = 1
    dt = 0.01

    scheme = trf.schemes.scipy_ode(model, integrator="dopri5")

    Set_Name = "Size-{:.2f}_Centrex{},y{}".format(size, str(x_roll).zfill(4), str(y_roll).zfill(4))
    if not os.path.isdir(location + "/" + Set_Name):
        os.mkdir(location + "/" + Set_Name)

    time.sleep(0.2)

    for i in range(0, int(TIME/dt)):
        if i == 0:
            new_t, new_fields = scheme(t=0, fields=init_fields.copy(), dt=dt, pars=pars, hook=solid_wall)
        else:
            new_t, new_fields = scheme(t=new_t, fields=new_fields.copy(), dt=dt, pars=pars, hook=solid_wall)

        eta = new_fields["h"] + new_fields["H"]
        norm_eta = (eta - eta.min()) / (eta.max() - eta.min())
        im = Image.fromarray(hillshade(norm_eta, 45, 50))
        im = im.convert('RGB')

        number = str(i).zfill(3)

        #print(Location + "/" + Set_Name + "/" + "img" + number +".j# pg")
        im.save(location + "/" + Set_Name + "/" + "img" + number +".jpg")

        if plot:
            im.show()









