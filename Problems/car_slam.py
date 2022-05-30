import numpy as np
import scipy.linalg

##############################################################################
# System Parameters ##########################################################
##############################################################################

# Variables/constants specified in the problem statement
xdim = 11
posedim = 3
mapdim = 8
ydim = 12
dt = 0.1
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim) 
vt = 1
mu0 = np.concatenate((np.zeros(posedim), np.array([0,0,10,0,10,10,0,10]))) # Merged mu0 for pose and map
sigma0 = 0.01 * np.eye(xdim)
# Other constants / useful values
duration = 20 # Length of the simulation
nTimesteps = np.floor(duration / dt).astype('int32')
numMarkers = 4
eps = 1e-3 # For iEKF
maxIterations = 30 # For iEKF

verySmallValue = 1e-6 # For division stability

# SIMULATION-ONLY STUFF
# This assumes that we know the map exactly
Q_sim = scipy.linalg.block_diag(0.1*dt*np.eye(posedim), np.zeros((mapdim, mapdim)))
sigma0_sim = scipy.linalg.block_diag(0.01*np.eye(posedim), np.zeros((mapdim, mapdim)))
markerLocs = [np.array([0,0]), np.array([10, 0]), np.array([10,10]), np.array([0,10])]

# NOTE:
# x0 and x0_sim have been moved to the main file due to havign the randomness evaluated only once

##############################################################################
# Functions and Jacobians ####################################################
##############################################################################

def f(x, u):
    px, py, theta, m1x, m1y, m2x, m2y, m3x, m3y, m4x, m4y = x # Unpack
    vt, phi = u
    dx_pose = np.array([vt*np.cos(theta)*dt, vt*np.sin(theta)*dt, phi*dt])
    dx_map = np.zeros(mapdim)
    dx = np.concatenate((dx_pose, dx_map))
    return x + dx

def g(x):
    # Updating this to the simplified measurement model 
    # ^ Range + bearing, but bearing does not incorporate the rotation matrix thing
    px, py, theta, m1x, m1y, m2x, m2y, m3x, m3y, m4x, m4y = x # Unpack
    # Range measurement calculations
    markerLocs = np.array([[m1x, m1y], [m2x, m2y], [m3x, m3y], [m4x, m4y]]) 
    position = np.array([px, py])
    g_range = np.array([np.linalg.norm(position - markerLocs[i]) for i in range(numMarkers)])
    # Bearing measurement calculations
    bearings = []
    for i in range(numMarkers): 
        mx, my = markerLocs[i]
        dx = mx - px
        dy = my - py
        diff = np.array([dx, dy])
        bearing = diff / (np.linalg.norm(diff) + verySmallValue)
        bearings.extend((bearing[0], bearing[1]))
    g_bearing = np.array(bearings)
    g = np.concatenate((g_range, g_bearing))
    return g

def A(x, u):
    px, py, theta, m1x, m1y, m2x, m2y, m3x, m3y, m4x, m4y = x # Unpack
    vt, phi = u
    A_pose =  np.array([[1, 0, -vt*np.sin(theta)*dt], 
                        [0, 1,  vt*np.cos(theta)*dt], 
                        [0, 0,                 1]])
    A_map = np.eye(mapdim)
    return scipy.linalg.block_diag(A_pose, A_map)

def C(x): 
    px, py, theta, m1x, m1y, m2x, m2y, m3x, m3y, m4x, m4y = x # New
    markerLocs = np.array([[m1x, m1y], [m2x, m2y], [m3x, m3y], [m4x, m4y]]) # New

    # Top left: (4,3): Range derivs wrt pose
    C_tl = np.zeros((numMarkers,posedim))
    for i in range(numMarkers):
        mx, my = markerLocs[i]
        dx = mx - px
        dy = my - py
        dist = np.sqrt(dx**2 + dy**2) + verySmallValue
        dri_dpx = -dx / dist
        dri_dpy = -dy / dist
        dri_dth = 0
        C_tl[i,:] = np.array([dri_dpx, dri_dpy, dri_dth]) # Third entry is 0 because dyi_dtheta = 0

    # Top right: (4,8): Range derivs wrt marker locations
    diags = []
    for i in range(numMarkers):
        mx, my = markerLocs[i]
        dx = mx - px
        dy = my - py
        dist = np.sqrt(dx**2 + dy**2) + verySmallValue
        dri_dmix = dx / dist
        dri_dmiy = dy / dist 
        diags.append((dri_dmix, dri_dmiy))
    C_tr = scipy.linalg.block_diag(*diags)

    # Bottom left: (8x3) Bearing derivs wrt pose
    C_bottomLeft_list = []
    for i in range(numMarkers):
        mx, my = markerLocs[i]
        dx = mx - px
        dy = my - py
        dist = np.sqrt(dx**2 + dy**2) + verySmallValue
        dbix_dpx = dx**2 / dist**3 - 1 / dist
        dbix_dpy = dx * dy / dist**3
        dbix_dth = 0
        dbiy_dpx = dx * dy / dist**3
        dbiy_dpy = dy**2 / dist**3 -  1 / dist
        dbiy_dth = 0
        C_bottomLeft_list.append(np.array([[dbix_dpx, dbix_dpy, dbix_dth], 
                                            [dbiy_dpx, dbiy_dpy, dbiy_dth]]))
    C_bl = np.vstack(C_bottomLeft_list)

    # Bottom right: (8,8): Bearing derivs wrt marker locations
    diags = []
    for markerID in range(numMarkers):
        mx, my = markerLocs[markerID]
        dx = mx - px
        dy = my - py
        dist = np.sqrt(dx**2 + dy**2) + verySmallValue
        dbix_dmix = 1 / dist - dx**2 / dist**3
        dbix_dmiy = -dx * dy / dist**3
        dbiy_dmix = -dx * dy / dist**3
        dbiy_dmiy = 1 / dist - dy**2 / dist**3
        diags.append(np.array([[dbix_dmix, dbix_dmiy], 
                               [dbiy_dmix, dbiy_dmiy]]))
    C_br = scipy.linalg.block_diag(*diags)


    C_top = np.hstack((C_tl, C_tr))
    C_bot = np.hstack((C_bl, C_br))
    C = np.vstack((C_top, C_bot))
    return C


def f_PF(X, u):
    raise Exception("Not implemented yet")

def g_PF(X):
    raise Exception("Not implemented yet")
