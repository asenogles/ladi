import numpy as np

from ._ext import *

def prediction_1st_order(Xi: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Predicts the value of the state using the 1st order derivative

    Args:
        Xi (np.ndarray): previous state vector (D)
        A (np.ndarray): transition prediction matrix (DxD)

    Returns:
        (np.ndarray): Containing new x and dx values
    """
    assert(Xi.shape[0] == A.shape[0] == A.shape[1])
    return cy_prediction_1st_order(Xi.astype(np.float32, order='C'), A.astype(np.float32, order='C'))

def prediction_2nd_order(Xi: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Predicts the value of the next state using the 1st and 2nd order derivatives

    Args:
        Xi (np.ndarray): previous state vector (D)
        A (np.ndarray): transition prediction matrix (DxD)
        B (np.ndarray): control variable matrix (D)

    Returns:
        (np.ndarray): Containing new x and dx values
    """
    return np.dot(A, Xi) + B

def predicted_covariance(Pi: np.ndarray, A: np.ndarray, Q: float = 0.08) -> np.ndarray:
    """Predict next process covariance matrix

    Args:
        Pi (np.ndarray): previous process covariance matrix (DxD)
        A (np.ndarray): transition prediction matrix (DxD)
        Q (float, optional): process noise. Defaults to 0.08.

    Returns:
        np.ndarray: DxD new predicted process covariance matrix
    """
    assert(Pi.shape == A.shape)
    return cy_predicted_covariance(Pi.astype(np.float32, order='C'), A.astype(np.float32, order='C'), Q)

def kalman_gain(P: np.ndarray, R: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Compute the Kalman gain

    Args:
        P (np.ndarray): Process covariance matrix (DxD)
        R (np.ndarray): measurement uncertainty matrix (DxD)
        H (np.ndarray): transition matrix (DxD) - transforms P into correct form for Kalman filter

    Returns:
        np.ndarray: DxD kalman gain matrix
    """
    assert(P.shape == R.shape == H.shape)
    return cy_kalman_gain(P.astype(np.float32, order='C'), R.astype(np.float32, order='C'), H.astype(np.float32, order='C'))

def update_state(Xp: np.ndarray, Y: np.ndarray, Kg: np.ndarray) -> np.ndarray:
    """Compute the new state

    Args:
        Xp (np.ndaaray): Predicted state matrix (D)
        Y (np.ndarray): measurement observation matrix (D)
        Kg (np.ndarray): Kalman gain matrix (DxD)

    Returns:
        np.ndarray: D state matrix
    """
    assert(Xp.shape == Y.shape == Kg.shape[:1])
    return cy_update_state(Xp.astype(np.float32, order='C'), Y.astype(np.float32, order='C'), Kg.astype(np.float32, order='C'))

def update_covariance(Pp: np.ndarray, H: np.ndarray, Kg: np.ndarray) -> np.ndarray:
    """Compute the new covariance matrix

    Args:
        Pp (np.ndarray): Predicted covariance matrix (dxd)
        H (np.ndarray): Transition matrix (dxd)
        Kg (np.ndarray): Kalman gain matrix (dxd)

    Returns:
        np.ndarray: dxd prediction covariance matrix
    """
    assert(Pp.shape == H.shape == Kg.shape)
    return cy_update_covariance(Pp.astype(np.float32, order='C'), H.astype(np.float32, order='C'), Kg.astype(np.float32, order='C'))

def smooth_gain(Pf: np.ndarray, Pr: np.ndarray, Hf: np.ndarray, Hr: np.ndarray) -> np.ndarray:
    """Compute the smoothing gain matrix based on the forward and reverse covariance matrices

    Args:
        Pf (np.ndarray): forward covariance matrix (DxD)
        Pr (np.ndarray): reverse covariance matrix (DxD)
        Hf (np.ndarray): forward transition matrix (DxD)
        Hr (np.ndarray): reverse transition matrix (DxD)

    Returns:
        np.ndarray: smoothing gain matrix (DxD)
    """
    assert(Pf.shape == Pr.shape == Hf.shape == Hr.shape)
    return cy_smooth_gain(Pf.astype(np.float32, order='C'), Pr.astype(np.float32, order='C'), Hf.astype(np.float32, order='C'), Hr.astype(np.float32, order='C'))

def multi_smooth_gain(P: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Compute the smoothing gain matrices based on multi covariance matrices

    Args:
        P (np.ndarray): covariance matrices (NxDxD)
        H (np.ndarray): transition matrices (NxDxD)

    Returns:
        np.ndarray: _description_
    """
    assert(P.shape == H.shape)
    # Compute inverse of sum of inverses
    P_sum_inv = np.linalg.pinv(np.linalg.pinv(P).sum(axis=0))
    return cy_multi_smooth_gain(P.astype(np.float32, order='C'), H.astype(np.float32, order='C'), P_sum_inv.astype(np.float32, order='C'))

def smooth_covariance(Pf: np.ndarray, Pr: np.ndarray, Sg: np.ndarray) -> np.ndarray:
    """Computed the smoothed covariance based on the forward and reverse covariances and the smoothing gain

    Args:
        Pf (np.ndarray): forward covariance matrix (DxD)
        Pr (np.ndarray): reverse covariance matrix (DxD)
        Sg (np.ndarray): Smoothing gain matrix (DxD)

    Returns:
        np.ndarray: Smooth covariance matrix (DxD)
    """
    assert(Pf.shape == Pr.shape == Sg.shape)
    return cy_smooth_covariance(Pf.astype(np.float32, order='C'), Pr.astype(np.float32, order='C'), Sg.astype(np.float32, order='C'))

class Ladi():

    def __create_A(self, D: int, dc: int) -> np.ndarray:
        """creates the A transition matrix used in kalman pass

        Args:
            D (int): dimensions of the square matrix
            dc (int): The step size between cells

        Returns:
            np.ndarray: DxD square A transition matrix
        """
        A = (dc * np.eye(D, k=D//2) ) + np.eye(D)
        return A.copy().astype(np.float32)

    def __flip(self, arr: np.ndarray) -> np.ndarray:
        """Flips the array along the columns

        Args:
            arr (np.ndarray): array to flip, must be atleast 2D

        Returns:
            np.ndarray: copy of the flipped array
        """
        return np.flip(arr, axis=1)

    def __rotate(self, arr: np.ndarray) -> np.ndarray:
        """Rotates the 1st two axes of an array

        Args:
            arr (np.ndarray): array to rotate, must be atleast 2D

        Returns:
            np.ndarray: copy of the rotated array
        """
        assert(arr.ndim >= 2)
        axes = [1,0]
        axes.extend(range(arr.ndim)[2:])
        return np.transpose(arr, axes=axes)

    def __create_H(self, ma: np.ndarray) -> np.ndarray:
        """Creates a H transition matrix used for kalman gain matrix formation
            using the mask of the measurement matrices (Y)

        Args:
            ma (np.ndarray): bool mask array, where true=no data, false=data (NROWSxNCOLSxD)

        Returns:
            np.ndarray: H matrix (NROWSxNCOLSxDxD)
        """
        H = np.tile(np.eye(self.D), (self.NROWS, self.NCOLS, 1, 1))
        for i in range(self.D):
            H[ma[:,:,i],i,i] = 0
        return H.copy().astype(np.float32)

    def __create_matrices(self):
        """Creates matrices based off data provided for use in LADI processing
        """
        # transition matrix (DxD) used for prediction of state vector (1st order)
        self.A = self.__create_A(self.D, self.dc)
        
        # transition matrix (DxD) used to transform measurement matrix into correct form
        self.C = np.eye(self.D, dtype=np.float32, order='C')
        
        # identity matrix (DxD)
        self.I = np.eye(self.D, dtype=np.float32, order='C')

    def __kf_pass(self, Y: np.ndarray, R: np.ndarray, H: np.ndarray) -> tuple:
        """Perform single pass across along each row

        Args:
            Y (np.ndarray): measurement vectors (NROWSxNCOLSxD)
            R (np.ndarray): covariance matrices (NROWSxNCOLSxDxD)
            H (np.ndarray): kalman gain transition matrices (NROWSxNCOLSxDxD)
            xn (np.ndarray): col idx of 1st valid measurement along each row

        Returns:
            tuple: the state, covariance, predicted state, and predicted covariance (X, P, Xp, Pp)
        """

        # get position of first non-masked measurements
        xn = np.argmax(~Y.mask[:,:,0], axis=1).astype(np.int32)

        # identify rows with no data
        xn_nodata = ~(np.amax(~Y.mask[:,:,0], axis=1))

        # set the index past the last column index for rows with no data
        xn[xn_nodata] = Y.shape[1]

        # replace masks in measurements with 0
        Y = np.ma.filled(Y.copy(), 0.)
        R = np.ma.filled(R.copy(), 0.)

        # Make copies of each array and ensure 32-bit float
        Y = np.array(Y.copy(), dtype=np.float32, order='C')
        R = np.array(R.copy(), dtype=np.float32, order='C')
        H = np.array(H.copy(), dtype=np.float32, order='C')

        X, Xp, P, Pp = cy_spatial_kf_pass(Y, R, H, XN=xn, Q=self.Q, gnorm=self.gnorm, T=self.T, numt=self.numt)

        # re-apply mask to points before xn
        idx = ~(xn[:,None] <= np.arange(X.shape[1]))
        X[idx, 0] = np.nan
        Xp[idx, 0] = np.nan
        P[idx, 0,0] = np.nan
        Pp[idx, 0,0] = np.nan
        X = np.ma.masked_invalid(X)
        Xp = np.ma.masked_invalid(Xp)
        P = np.ma.masked_invalid(P)
        Pp = np.ma.masked_invalid(Pp)

        return X, Xp, P, Pp

    def __forward_x(self, Yf: np.ndarray, Rf: np.ndarray, Hf: np.ndarray) -> tuple:
        """Run 1D forward pass across each row (x-component)

        Args:
            Yf (np.ndarray): measurement array (NROWxNCOLxD)
            Rf (np.ndarray): measurement covariance array (NROWxNCOLxDxD)
            Hf (np.ndarray): transition matrix array (NROWxNCOLxDxD)

        Returns:
            tuple: state array, predicted state array, state covariance, predicted state covariance
        """
        Yf = Yf.copy()
        Rf = Rf.copy()
        Hf = Hf.copy()

        Xf, Xpf, Pf, Ppf = self.__kf_pass(Yf, Rf, Hf)

        return Xf, Xpf, Pf, Ppf

    def __forward_y(self, Yf: np.ndarray, Rf: np.ndarray, Hf: np.ndarray) -> tuple:
        """Run 1D forward pass across each column (y-component)

        Args:
            Yf (np.ndarray): measurement array (NROWxNCOLxD)
            Rf (np.ndarray): measurement covariance array (NROWxNCOLxDxD)
            Hf (np.ndarray): transition matrix array (NROWxNCOLxDxD)

        Returns:
            tuple: state array, predicted state array, state covariance, predicted state covariance
        """
        # rotate rasters
        Yf = self.__rotate(Yf).copy()
        Rf = self.__rotate(Rf).copy()
        Hf = self.__rotate(Hf).copy()
        
        Xf, Xpf, Pf, Ppf = self.__kf_pass(Yf, Rf, Hf)

        # rotate rasters back
        Xf = self.__rotate(Xf)
        Xpf = self.__rotate(Xpf)
        Pf = self.__rotate(Pf)
        Ppf = self.__rotate(Ppf)

        return Xf, Xpf, Pf, Ppf

    def __reverse_x(self, Yr: np.ndarray, Rr: np.ndarray, Hr: np.ndarray) -> tuple:
        """Run 1D reverse pass across each row (x-component)

        Args:
            Yr (np.ndarray): measurement array (NROWxNCOLxD)
            Rr (np.ndarray): measurement covariance array (NROWxNCOLxDxD)
            Hr (np.ndarray): transition matrix array (NROWxNCOLxDxD)

        Returns:
            tuple: state array, predicted state array, state covariance, predicted state covariance
        """
        # flip raster order in cols
        Yr = self.__flip(Yr).copy()
        Rr = self.__flip(Rr).copy()
        Hr = self.__flip(Hr).copy()

        # flip sign of derivarive
        Yr[:,:,self.D//2:] = -Yr[:,:,self.D//2:]
        
        Xr, Xpr, Pr, Ppr = self.__kf_pass(Yr, Rr, Hr)

        # flip raster order back in cols
        Xr = self.__flip(Xr)
        Xpr = self.__flip(Xpr)
        Pr = self.__flip(Pr)
        Ppr = self.__flip(Ppr)

        # flip sign of derivarive back
        Xr[:,:,self.D//2:] = -Xr[:,:,self.D//2:]
        Xpr[:,:,self.D//2:] = -Xpr[:,:,self.D//2:]

        return Xr, Xpr, Pr, Ppr

    def __reverse_y(self, Yr: np.ndarray, Rr: np.ndarray, Hr: np.ndarray) -> tuple:
        """Run 1D reverse pass across each column (y-component)

        Args:
            Yr (np.ndarray): measurement array (NROWxNCOLxD)
            Rr (np.ndarray): measurement covariance array (NROWxNCOLxDxD)
            Hr (np.ndarray): transition matrix array (NROWxNCOLxDxD)

        Returns:
            tuple: state array, predicted state array, state covariance, predicted state covariance
        """
        # rotate rasters
        Yr = self.__rotate(Yr).copy()
        Rr = self.__rotate(Rr).copy()
        Hr = self.__rotate(Hr).copy()

        # flip raster order in cols
        Yr = self.__flip(Yr).copy()
        Rr = self.__flip(Rr).copy()
        Hr = self.__flip(Hr).copy()

        # flip sign of derivarive
        Yr[:,:,self.D//2:] = -Yr[:,:,self.D//2:]

        Xr, Xpr, Pr, Ppr = self.__kf_pass(Yr, Rr, Hr)

        # flip raster order back in cols
        Xr = self.__flip(Xr)
        Xpr = self.__flip(Xpr)
        Pr = self.__flip(Pr)
        Ppr = self.__flip(Ppr)

        # flip sign of derivative back
        Xr[:,:,self.D//2:] = -Xr[:,:,self.D//2:]
        Xpr[:,:,self.D//2:] = -Xpr[:,:,self.D//2:]

        # rotate rasters back
        Xr = self.__rotate(Xr)
        Xpr = self.__rotate(Xpr)
        Pr = self.__rotate(Pr)
        Ppr = self.__rotate(Ppr)

        return Xr, Xpr, Pr, Ppr

    def __smooth(self, Xf: np.ndarray, Xr: np.ndarray, Pf: np.ndarray, Pr: np.ndarray, Hf: np.ndarray, Hr: np.ndarray) -> tuple:
        """Run the smoothing filter across each cell

        Args:
            Xf (np.ndarray): forward state array (NROWxNCOLxD)
            Xr (np.ndarray): reverse state array (NROWxNCOLxD)
            Pf (np.ndarray): forward covariance array (NROWxNCOLxDxD)
            Pr (np.ndarray): reverse covariance array (NROWxNCOLxDxD)
            Hf (np.ndarray): forward transition array (NROWxNCOLxDxD)
            Hr (np.ndarray): reverse transition array (NROWxNCOLxDxD)

        Returns:
            tuple: smoothed state, and smoothed state covariance arrays
        """
        # Adjust transition matrices to deal with missing data
        Hf = Hf.copy()
        Hr = Hr.copy()
        Hf[Pr.mask] = 0.
        Hr[~Pf.mask] = 0.

        # find cells with overlapping masks (no data in either pass)
        nodata_X = np.logical_and(Xf.mask, Xr.mask)
        nodata_P = np.logical_and(Pf.mask, Pr.mask)

        # identify rows with data
        valid_rows_X = ~(np.amax(nodata_X[:,:,0], axis=1))

        # make sure missing data is filled
        Xf = np.ma.filled(Xf.copy(), 0.)
        Pf = np.ma.filled(Pf.copy(), 0.)
        Xr = np.ma.filled(Xr.copy(), 0.)
        Pr = np.ma.filled(Pr.copy(), 0.)

        # make sure datatype is 32-bit float
        Xf = np.array(Xf, dtype = np.float32, order='C')
        Xr = np.array(Xr, dtype = np.float32, order='C')
        Pf = np.array(Pf, dtype = np.float32, order='C')
        Pr = np.array(Pr, dtype = np.float32, order='C')
        Hf = np.array(Hf, dtype = np.float32, order='C')
        Hr = np.array(Hr, dtype = np.float32, order='C')

        Xs, Ps = cy_spatial_smooth(Xf, Xr, Pf, Pr, Hf, Hr, valid_rows_X)

        # reapply masks to missing data
        Xs[nodata_X] = np.nan
        Ps[nodata_P] = np.nan
        Xs = np.ma.masked_invalid(Xs)
        Ps = np.ma.masked_invalid(Ps)
        
        return Xs, Ps

    def __multi_smooth(self, X: np.ndarray, P: np.ndarray, H: np.ndarray) -> tuple:
        """Run the smoothing filter for multiple passes across each cell

        Args:
            Xf (np.ndarray): state arrays (NROWxNCOLxNxD)
            Pf (np.ndarray): covariance arrays (NROWxNCOLxNxDxD)
            Hf (np.ndarray): transition arrays (NROWxNCOLxNxDxD)

        Returns:
            tuple: smoothed state, and smoothed state covariance arrays
        """
        # ensure correct dtypes & continous data
        X = X.astype(np.float32, order='C')
        P = P.astype(np.float32, order='C')
        H = H.astype(np.float32, order='C')

        # find cells with overlapping masks (no data in any pass)
        nodata_X = np.all(X.mask[:,:,:,0], axis=2)
        nodata_P = np.all(P.mask[:,:,:,0,0], axis=2)

        # identify cells with data
        valid_X = np.argwhere(~(nodata_X).flatten()).flatten()

        # make sure missing data is filled
        X = np.ma.filled(X.copy(), 0.)
        P = np.ma.filled(P.copy(), 0.)

        # make sure datatype is 32-bit float
        X = np.array(X, dtype = np.float32)
        P = np.array(P, dtype = np.float32)
        H = np.array(H, dtype = np.float32)
        valid_X = np.array(valid_X, dtype = np.int32).copy()

        Xs, Ps = cy_spatial_multi_smooth(X, P, H, valid_X, numt=self.numt)

        # reapply masks to missing data
        Xs[nodata_X] = np.nan
        Ps[nodata_P] = np.nan
        Xs = np.ma.masked_invalid(Xs)
        Ps = np.ma.masked_invalid(Ps)
        
        return Xs, Ps

    def __si_fast(self) -> tuple:
        """Run Ladi spatial interpolation using the fast method

        Returns:
            tuple: interpolated state and covariance arrays
        """

        # create Hx for x passes
        Hx = self.__create_H(self.Yx.mask)

        # run forward pass in x direction
        Xfx, Xpfx, Pfx, Ppfx = self.__forward_x(self.Yx, self.Rx, Hx)

        # run backward pass in x direction
        Xrx, Xprx, Prx, Pprx = self.__reverse_x(self.Yx, self.Rx, Hx)

        # run smooth
        Hf = self.__create_H(Xfx.mask)
        Hr = self.__create_H(Xrx.mask)
        Xsx, Psx = self.__smooth(Xfx, Xrx, Pfx, Prx, Hf, Hr)

        # copy measurements/covariance from x passes to Yy (not gradient) (all measurements)
        Yy  = self.Yy.copy()
        Ry  = self.Ry.copy()
        Yy[:,:,0] = Xsx[:,:,0]
        Ry[:,:,0,0] = Psx[:,:,0,0]
        Yy = Yy.copy()
        Ry = Ry.copy()

        # copy measurements/covariance from x passes to Yy (not gradient) (only missing data)
        # Yy  = self.Yy.copy()
        # Ry  = self.Ry.copy()
        # Yy[:,:,0] = np.ma.filled(Yy[:,:,0], fill_value=Xsx[:,:,0])
        # Ry[:,:,0,0] = np.ma.filled(Ry[:,:,0,0], fill_value=Psx[:,:,0,0])
        # Yy = np.ma.masked_invalid(Yy).copy()
        # Ry = np.ma.masked_invalid(Ry).copy()

        # Create Hy for y passes
        Hy = self.__create_H(Yy.mask)

        # run forward pass in y direction
        Xfy, Xpfy, Pfy, Ppfy = self.__forward_y(Yy, Ry, Hy)

        # run backward pass in y direction
        Xry, Xpry, Pry, Ppry = self.__reverse_y(Yy, Ry, Hy)

        # run smooth
        Hf = self.__create_H(Xfy.mask)
        Hr = self.__create_H(Xry.mask)
        Xsy, Psy = self.__smooth(Xfy, Xry, Pfy, Pry, Hf, Hr)
        
        return Xsy, Psy

    def __si_reg_phase(self, Yx: np.ndarray, Yy: np.ndarray, Rx: np.ndarray, Ry: np.ndarray) -> tuple:
        """Compute one phase of the ladi spatial interpolation regular method

        Args:
            Yx (np.ndarray): Measurement Y array with gradient in x axis
            Yy (np.ndarray): Measurment Y array with gradient in y axis
            Rx (np.ndarray): Covariance R array in x axis
            Ry (np.ndarray): Covariance R array in y axis

        Returns:
            tuple: interpolated state and covariance arrays
        """
        # First run passes in each direction/axis using the measurements only
        Hx = self.__create_H(Yx.mask)  # create Hx for x passes
        Hy = self.__create_H(Yy.mask)  # create Hx for y passes

        # run forward pass in x direction
        Xfx, Xpfx, Pfx, Ppfx = self.__forward_x(Yx, Rx, Hx)
        # run reverse pass in x direction
        Xrx, Xprx, Prx, Pprx = self.__reverse_x(Yx, Rx, Hx)
        # run forward pass in y direction
        Xfy, Xpfy, Pfy, Ppfy = self.__forward_y(Yy, Ry, Hy)
        # run reverse pass in y direction
        Xry, Xpry, Pry, Ppry = self.__reverse_y(Yy, Ry, Hy)

        Hfx = self.__create_H(Xfx.mask)
        Hrx = self.__create_H(Xrx.mask)
        Hfy = self.__create_H(Xfy.mask)
        Hry = self.__create_H(Xry.mask)

        # now run smooth
        X = np.ma.stack((Xfx, Xrx, Xfy, Xry), axis=2)
        P = np.ma.stack((Pfx, Prx, Pfy, Pry), axis=2)
        H = np.ma.stack((Hfx, Hrx, Hfy, Hry), axis=2)
        Xs, Ps = self.__multi_smooth(X, P, H)

        return Xs, Ps
    
    def __si_reg(self) -> tuple:
        """Run Ladi spatial interpolation using the regular method

        Returns:
            tuple: interpolated state and covariance arrays
        """
        Yx = self.Yx.copy()
        Yy  = self.Yy.copy()
        Rx = self.Rx.copy()
        Ry  = self.Ry.copy()

        # Run the 1st phase
        Xs1, Ps1 = self.__si_reg_phase(Yx, Yy, Rx, Ry)

        # copy results
        Yx[:,:,0] = Xs1[:,:,0]
        Yy[:,:,0] = Xs1[:,:,0]
        Rx[:,:,0,0] = Ps1[:,:,0,0]
        Ry[:,:,0,0] = Ps1[:,:,0,0]
        Yx = Yx.copy()
        Yy = Yy.copy()
        Rx = Rx.copy()
        Ry = Ry.copy()

        # run the 2nd phase
        Xs2, Ps2 = self.__si_reg_phase(Yx, Yy, Rx, Ry)

        return Xs2, Ps2

    def __init__(self, Yx: np.ndarray, Yy: np.ndarray, Rx: np.ndarray, Ry: np.ndarray, gnorm: bool = True, Q: float = 0.08, T: float = 0.01, dc: int = 1, numt: int = 1):
        """initialize spatial Kalman filter

        Args:
            Yx (np.ndarray): Measurement matrix with spatial derivative of Y in x direction (NROWxNCOLxD)
            Yy (np.ndarray): Measurement matrix with spatial derivative of Y in y direction (NROWxNCOLxD)
            Rx (np.ndarray): Measurement uncertainty with spatial derivative in x direction (NROWxNCOLxDxD)
            Ry (np.ndarray): Measurement uncertainty with spatial derivative in x direction (NROWxNCOLxDxD)
            gnorm (bool), optional): whether the gradients have been normalized. Defaults to True
            T (float, optional): Rounding Threshold, values below treated as zero. Defaults to 0.01.
            Q (float, optional): Process noise. Defaults to 0.08.
            dc (int), optional): The step size between cells. Defaults to 1
            numt (int, optional): number of threads to use for parallel operations. Defaults to 1.
        """

        # ensure arrays are of the correct dims
        Yx = np.ma.atleast_3d(Yx)
        Yy = np.ma.atleast_3d(Yy)
        assert(Rx.ndim == 4)
        assert(Ry.ndim == 4)

        # Make sure array shapes line up
        assert(Yx.shape == Yy.shape)
        assert(Rx.shape == Ry.shape)
        assert(Yx.shape == Rx.shape[:3])
        
        self.gnorm = gnorm          # gradiant norm status
        self.Q = Q                  # process noise
        self.T = T                  # rounding threshold (below treated as zero)
        self.dc = dc                # step size (number of cells)
        self.numt = numt            # number of threads
        self.NROWS = Yx.shape[0]    # number of rows
        self.NCOLS = Yx.shape[1]    # number of cols
        self.D = Yx.shape[2]        # number of data (measurements) to track

        if self.D != 2:
            raise ValueError(f'Number of dimensions must equal 2, not {self.D}')

        # create copies of arrays and ensure 32bit float type
        self.Yx = Yx.copy().astype(np.float32)
        self.Yy = Yy.copy().astype(np.float32)
        self.Rx = Rx.copy().astype(np.float32)
        self.Ry = Ry.copy().astype(np.float32)

        # create transition matrices
        self.__create_matrices()

    def spatial_interpolate(self, method: str = 'REG') -> tuple:
        """Run LADI Spatial interpolation

        Args:
            method (str, optional): Ladi method to use. Defaults to 'REG'.

        Raises:
            ValueError: invalid method provided.

        Returns:
            tuple: Interpolated state and covariance arrays.
        """

        if method == 'REG':
            Xs, Ps = self.__si_reg()
        elif method == 'FAST':
            Xs, Ps = self.__si_fast()
        else:
            raise ValueError(f'method: {method} is invalid')

        return Xs, Ps

