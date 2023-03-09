import numpy as np

from .ladi import Ladi

def create_Y(m: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Create Y measurement array

    Args:
        m (np.ndarray): Array containing measurements
        g (np.ndarray): Array containing gradient (1st deriv)

    Returns:
        np.ndarray: Y measurement array of shape meas/grad.shape x 2
    """
    assert(m.shape == g.shape)
    Y = np.ma.stack((m, g), axis=-1)
    return Y

def create_R(mv: np.ndarray, gv: np.ndarray, c: np.ndarray = None, D: int = 2) -> np.ndarray:
    """Create R variance/covariance measurement array

    Args:
        mv (np.ndarray): measurement variance array
        gv (np.ndarray): variance of the gradiant of the measurements
        c (np.ndarray, optional): covariance between measurement & gradient. Defaults to None.
        D (int, optional): # of dims. Defaults to 2.

    Returns:
        np.ndarray: R variance/covariance array of shape NROWSxNCOLSxDxD
    """
    R_shape = list(mv.shape)
    if c is None:
        c = np.zeros(R_shape)
    assert(mv.shape == gv.shape == c.shape)
    R_shape.extend([D,D])
    R = np.ma.stack((mv, c, c, gv), axis=-1).reshape(R_shape)
    return R

def create_YR(m: np.ndarray, g: np.ndarray, mv: np.ndarray, gv: np.ndarray, c: np.ndarray = None) -> tuple:
    """create the Y & R arrays for a set of measurements+variance/covariance

    Args:
        m (np.ndarray): Array of measurement values
        g (np.ndarray): Array of measurement gradient (1st deriv) values
        mv (np.ndarray): Array of measurement variance
        gv (np.ndarray): Array of gradient variance
        c (np.ndarray, optional): Array of covariance. Defaults to None.

    Returns:
        tuple: Y & R arrays
    """
    if c is None:
        c = np.zeros(mv.shape)

    assert(m.shape == g.shape == mv.shape == gv.shape == c.shape)

    # create Y & R
    Y = create_Y(m, g)
    R = create_R(mv, gv, c, D=2)

    assert(Y.shape == R.shape[:3])

    return Y, R

def round_to_zero(arr: np.ndarray, T: float = 0.01) -> np.ndarray:
    """rounds elements of array within threshold of zero to zero

    Args:
        arr (np.ndarray): array of values to round
        T (float, optional): threshold within values will be made zero. Defaults to 0.01.

    Returns:
        np.ndarray: rounded array
    """
    arr[np.absolute(arr) < T] = 0.
    return arr

def norm_grad(g: np.ndarray, m: np.ndarray, T: float = 0.01) -> np.ndarray:
    """normalize the gradient by measurement value

    Args:
        g (np.ndarray): gradient array
        m (np.ndarray): measurement array
        T (float, optional): threshold below which values are treated as zero. Defaults to 0.01.

    Returns:
        np.ndarray: the normalized gradient
    """
    g = np.array(g.copy())
    m = np.array(m.copy())
    
    assert(g.shape == m.shape)
    assert(~np.isnan(m).any()) # make sure no nans in measurement

    # use absolute values
    sign = np.sign(g)
    m = np.abs(m)
    g = np.abs(g)

    # treat values below T as zero
    m = round_to_zero(m, T)

    # if meas is zero, set to nan
    m[m == 0.] = np.nan

    # now normalize the gradient by the measurement
    gn = g / m

    # retain grad == 0
    gn[g == 0.] = 0

    # normed grad == grad where meas == 0
    gn[np.isnan(gn)] = g[np.isnan(gn)]

    # reapply sign to the gradients
    gn *= sign

    # mask the nans
    gn = np.ma.masked_invalid(gn)
    
    return gn

def grad_uncertainty(v: np.ndarray, axis: int) -> np.ndarray:
    """estimate uncertainty of the central difference gradient of a 2D array

    Regular central difference gradient equation computed with np.gradient:
        g = ( x[i+1] - x[i-1] ) / 2
    uncertainty calculation is thus:
        dg = sqrt( dx[i+1]^2 + dx[i-1]^2 ) / 2

    Args:
        v (np.ndarray): 2D array of measurement uncertainties
        axis (int): axis to compute gradient uncertainty in

    Raises:
        ValueError: if axis argument is out of range

    Returns:
        np.ndarray: 2D array of gradient uncertainties
    """
    
    assert(v.ndim == 2)
    
    # create empty gradient uncertainty array to fill
    gv = np.empty_like(v)

    # get square of measurement uncertainty
    v_sq = np.power(v, 2)
    
    # compute gradient uncertainty
    if axis == 0:
        gv[1:-1,:] = np.sqrt(v_sq[2:,:] + v_sq[:-2,:]) / 2
        gv[0,:] = np.sqrt(v_sq[1,:] + v_sq[0,:])
        gv[-1,:] = np.sqrt(v_sq[-1,:] + v_sq[-2,:])
    elif axis == 1:
        gv[:,1:-1] = np.sqrt(v_sq[:,2:] + v_sq[:,:-2]) / 2
        gv[:,0] = np.sqrt(v_sq[:,1] + v_sq[:,0])
        gv[:,-1] = np.sqrt(v_sq[:,-1] + v_sq[:,-2])
    else:
        raise ValueError(f'axis == {axis} is invalid')

    return gv

def pre_process(Yd: np.ndarray, Ys: np.ndarray, Ydv: np.ndarray, Ysv: np.ndarray, Gv: np.ndarray = None, C: np.ndarray = None, gnorm: bool = True, Q: float = 0.08, numt: int = 1) -> Ladi:
    """Run all of the pre-processing steps for LADI

    Args:
        Yd (np.ndarray): Calibration measurement array (NROWSxNCOLS)
        Ys (np.ndarray): Control point measurement array (NROWSxNCOLS)
        Ydv (np.ndarray): Calibration measurement variance array (NROWSxNCOLS)
        Ysv (np.ndarray): Control point measurement variance array (NROWSxNCOLS)
        Gv (np.ndarray, optional): Gradient Variance array (2xNROWSxNCOLS). Defaults to None.
        C (np.ndarray, optional): Covariance array (2xNROWSxNCOLS). Defaults to None.
        gnorm (bool, optional): whether to normalize gradients. Defaults to True.
        Q (float, optional): process noise coefficient. Defaults to 0.08.

    Returns:
        Ladi: initialized LADI object
    """

    assert(Yd.shape == Ys.shape == Ydv.shape == Ysv.shape)
    assert(Yd.ndim == 2)

    
    # if no gradient variance is provided, then estimate
    if (Gv is None):
        print('hit compute grad var')
        Gvy = grad_uncertainty(Ydv, axis=0)
        Gvx = grad_uncertainty(Ydv, axis=1)
        Gv = np.ma.stack((Gvy, Gvx), axis=0)

    # if no covariance is provided, then estimate
    if (C is None):
        print('hit compute covar')
        C = Ysv * Gv

    assert(Yd.shape == Gv.shape[1:] == C.shape[1:])

    # Compute gradient using dense (calibration) data
    Gy, Gx = np.gradient(Yd)

    # Normalize the gradient
    if gnorm:
        Gx = norm_grad(Gx, Yd)
        Gy = norm_grad(Gy, Yd)

    # create matrices
    Yx, Rx = create_YR(Ys, Gx, Ysv, Gv[1,:,:], C[1,:,:])
    Yy, Ry = create_YR(Ys, Gy, Ysv, Gv[0,:,:], C[0,:,:])
    assert(np.array_equal(Yx[:,:,0].mask, Rx[:,:,0,0].mask))
    assert(np.array_equal(Yy[:,:,0].mask, Ry[:,:,0,0].mask))

    # increase gradient uncertainty for cells with meas == 0
    if gnorm:
        err_zero_grad = 9999
        Rx[:,:,1,1][Yd == 0.] = err_zero_grad
        Ry[:,:,1,1][Yd == 0.] = err_zero_grad

    return Ladi(Yx.copy(), Yy.copy(), Rx.copy(), Ry.copy(), gnorm=gnorm, Q=Q, numt=numt)
