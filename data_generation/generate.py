import triflow as trf
import numpy as np
import scipy.signal
import os
from PIL import Image
import random
import copy
import argparse
import sys
sys.path.append('..')
from utils.io import save_json

"""
Wilhelm Sorteberg, 2018
wilhelm@sorteberg.eu

"""


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--location', type=str, default='./debug_data_gen', help='Folder to save the files')
    parser.add_argument('--azimuth', type=int, default=45, help='Lighting angle')
    parser.add_argument('--viewing_angle', type=int, default=20, help='Viewing angle')
    parser.add_argument('--container_size_min', type=int, default=10, help='How big the water container (box) is')
    parser.add_argument('--container_size_max', type=int, default=20, help='How big the water container (box) is')
    parser.add_argument('--water_depth', type=int, default=10)
    parser.add_argument('--initial_stimulus', type=int, default=1, help='Strength of initial stimuli')
    parser.add_argument('--coriolis_force', type=float, default=0.0, help='Coriolis force coefficient')
    parser.add_argument('--water_viscocity', type=int, default=1e-6, help='Water viscocity')

    parser.add_argument('--total_time', type=float, default=1.0, help='Total sequence time in seconds')
    parser.add_argument('--dt', type=float, default=0.01, help='Time interval between frames in seconds')
    parser.add_argument('--image_size_x', type=int, default=184, help='Pixel size of the output images')
    parser.add_argument('--image_size_y', type=int, default=184, help='Pixel size of the output images')
    parser.add_argument('--data_points', type=int, default=500, help='How many sequences to create')

    args = parser.parse_args()

    return args


class argsclass():
    pass


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


if is_interactive():
    # %matplotlib notebook
    import matplotlib.pyplot as plt
    args = argsclass()
    plot = True
    args.location = "./debug"
    args.azimuth = 45             # 45
    args.viewing_angle = 20       # 20
    args.container_size_min = 10  # 10
    args.container_size_max = 20  # 20
    args.water_depth = 10         # 10
    args.initial_stimulus = 1     # 1
    args.coriolis_force = 0.0    # 0
    args.water_viscocity = 10e-6  # 0
    args.TIME = 1.0               # 1
    args.dt = 0.01                # 0.01
    args.data_points = 5
    args.image_size_x = args.image_size_y = 184
else:
    plot = False
    args = get_args()

if not os.path.isdir(args.location):
    os.mkdir(args.location)

save_json(vars(args), os.path.join(args.location, 'parameters.json'))


def hillshade(array, azimuth, angle_altitude):

    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.
    altituderad = angle_altitude * np.pi / 180.

    shaded = np.sin(altituderad) * np.sin(slope) + \
             np.cos(altituderad) * np.cos(slope) * \
             np.cos(azimuthrad - aspect)
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


def solid_wall(t, fields, pars):
    fields["u"][:, 0] = 0.
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


def constrict(val, min_val, max_val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val


if plot:
    fig = plt.figure(figsize=(6, 6))
    image_real = np.zeros((args.image_size_x, args.image_size_y))
    img = plt.imshow(image_real, cmap='gray')
    plt.draw()
    plt.show(block=False)

model = NonConservative_ShallowWater()

while len(os.listdir(args.location)) - 1 < args.data_points:
    MaxRoll = args.image_size_x / 2 - 10
    size = random.uniform(args.container_size_min, args.container_size_max)  # Container size
    x_roll = random.randint(-MaxRoll, MaxRoll)
    y_roll = random.randint(-MaxRoll, MaxRoll)
    x = np.linspace(0, size, args.image_size_x)           # Physical domain
    y = np.linspace(0, size, args.image_size_y)           # Physical domain
    u = np.zeros((args.image_size_x, args.image_size_y))          # x velocity
    v = np.zeros((args.image_size_x, args.image_size_y))          # y velocity    #Can add +2 to add a general underlying velocity
    h = np.zeros((3 * args.image_size_x, 3 * args.image_size_y)) + (scipy.signal.windows.gaussian(3 * args.image_size_x, 5) * scipy.signal.windows.gaussian(3 * args.image_size_y, 5)[:, None]) * args.initial_stimulus
    h = np.roll(h, x_roll, axis=1)        # Drop x position
    h = np.roll(h, y_roll, axis=0)        # Drop Y position
    h = h[args.image_size_x: 2 * args.image_size_x, args.image_size_y: 2 * args.image_size_y]
    H = np.ones((args.image_size_x, args.image_size_y)) * args.water_depth
    init_fields = model.fields_template(x=x, y=y, h=h, u=u, v=v, H=H)

    scheme = trf.schemes.scipy_ode(model, integrator="dopri5")

    Set_Name = "Size-{:.2f}_Centre_x{},y{}".format(size, str(x_roll).zfill(4), str(y_roll).zfill(4))
    if not os.path.isdir(args.location + "/" + Set_Name):
        os.mkdir(args.location + "/" + Set_Name)

    pars = {"f": args.coriolis_force, "nu": args.water_viscocity}
    for i in range(0, int(args.total_time / args.dt)):
        if i == 0:
            new_t, new_fields = scheme(t=0, fields=init_fields.copy(), dt=args.dt, pars=pars, hook=solid_wall)
        else:
            new_t, new_fields = scheme(t=new_t, fields=new_fields.copy(), dt=args.dt, pars=pars, hook=solid_wall)

        eta = new_fields["h"] + new_fields["H"]
        norm_eta = (eta - eta.min()) / (eta.max() - eta.min())
        im = Image.fromarray(hillshade(norm_eta, args.azimuth, args.viewing_angle))
        im = im.convert('RGB')

        number = str(i).zfill(3)

        print(args.location + "/" + Set_Name + "/" + "img" + number + ".jpg")
        im.save(args.location + "/" + Set_Name + "/" + "img" + number + ".jpg")

        if plot:
            img.set_data(im)
            fig.canvas.draw()
            plt.show(block=False)
