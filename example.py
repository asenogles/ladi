import numpy as np
import ladi
import fasterraster as fr
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import flow_vis

subplt_kwdict = {
    'xticks': [],
    'yticks': []
}

gridspec_kwdict = {
    'left': 0.01,
    'right': 0.94,
    'top': 0.92,
    'bottom': 0.02,
    'wspace': 0.02, 
    'width_ratios':[1, 1, 1,0.05],
    'height_ratios':[1]
}

quiver_kwdict = {
    'scale': 0.005,
    'width': 0.0025,
    'headwidth': 2,
    'headlength': 2.5
}

cm = plt.cm.Reds
cbar_min = 0.
cbar_max = 0.2
norm = Normalize(cbar_min, cbar_max)

fname = Path('./examples/example1.flo')
outname = Path('./examples/output/example1.png')

cp = np.array([400,256], dtype=int)

err_meas_x = 0.5
err_meas_sx = 0.2
err_meas_dx = 0.05
err_meas_sdx = 0.02

if __name__ == '__main__':
    
    # Load example calibration data
    calib = fr.Flo(fname)

    # We will recreate the midpoint of the calibration data as a simple example
    test = calib.deepcopy()
    test.raster = test.raster / 2.

    # We will run ladi on the displacement magnitude data as a simple way to speed up the process
    # We can then return the output to vector form using the unit vector of the calibration data
    # Alternative we could run ladi on each vector component
    mcalib = calib.magnitude()
    mtest = test.magnitude()

    # Now lets simulate variance data to use with ladi
    np.random.seed(1)
    calib_var = np.random.normal(err_meas_x, err_meas_sx, size=(calib.NROWS, calib.NCOLS))
    mxvar_grad = np.random.normal(err_meas_dx, err_meas_sdx, size=(calib.NROWS, calib.NCOLS))
    myvar_grad = np.random.normal(err_meas_dx, err_meas_sdx, size=(calib.NROWS, calib.NCOLS))
    calib_grad_var = np.ma.stack((myvar_grad, mxvar_grad), axis=0)
    covariance = np.ma.zeros((2, calib.NROWS, calib.NCOLS))

    # Get the test displacement at the control point and put it into a sparse array
    ctrl = np.full_like(mtest, np.nan)
    ctrl[cp] = mtest[cp]
    ctrl = np.ma.masked_invalid(ctrl)

    ctrl_var = np.full_like(ctrl, np.nan)
    ctrl_var[cp] = calib_var[cp]
    ctrl_var = np.ma.masked_invalid(ctrl_var)

    # Perform pre-processing and create ladi object
    LD = ladi.pre_process(
        Yd = mcalib,            # Calibration dense displacement data
        Ys = ctrl,              # Control sparse displacement data
        Ydv = calib_var,        # Calibration dense displacement variance data
        Ysv = ctrl_var,         # Control sparse displacement variance data
        Gv = calib_grad_var,    # Calibration dense displacement gradient variance data
        C = covariance,         # Covariance data
        gnorm = True,           # Normalizes the gradient data used for ladi interpolation
        Q = 0.08,               # Process noise coefficent
        numt = 1,               # Number of threads to use in ladi
    )
    
    # Run ladi spatial interpolation
    Xs, Ps = LD.spatial_interpolate()

    # Reconstruct the output of ladi to vector form using unit vector of calibration data
    unit = calib.unit_vector()
    recreate = Xs[:,:,0,None] * unit
    recreate = fr.Flo(fname=None, data=np.array(recreate).copy())

    # Compute end point error grid between test data and output of ladi interpolation
    epe = fr.compute_epe(test, recreate)
    print('average epe: ', np.nanmean(epe))

    # Plot results
    fig, ax = plt.subplots(1,4, figsize=(12.,4.), dpi=300, subplot_kw=subplt_kwdict, gridspec_kw=gridspec_kwdict)
    
    # Plot test data
    ax[0].set_title('Test data')
    ax[0].imshow(flow_vis.flow_to_color(test.raster))
    test.flo_to_quiver(ax[0], step=30, **quiver_kwdict)

    # Plot ladi output
    ax[1].set_title('$\it{LADI}$ Interpolation')
    ax[1].imshow(flow_vis.flow_to_color(recreate.raster))
    recreate.flo_to_quiver(ax[1], step=30, **quiver_kwdict)

    # Plot end-point error
    ax[2].set_title('End-point error')
    ax[2].imshow(epe, cmap=cm, norm=norm)

    # Add error scalebar
    cbar_ticks = np.arange(cbar_min, cbar_max+0.00001, 0.05)
    cbar_ticks_labels = [f'{i:.2f}m' for i in np.arange(cbar_min, cbar_max+0.00001, 0.05)]
    cbar_ticks_labels[-1] = f'>{cbar_ticks_labels[-1]}'
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), cax=ax[3], orientation='vertical')
    cbar.ax.get_yaxis().set_ticks(cbar_ticks)
    cbar.ax.get_yaxis().set_ticklabels(cbar_ticks_labels)

    plt.savefig(outname)
