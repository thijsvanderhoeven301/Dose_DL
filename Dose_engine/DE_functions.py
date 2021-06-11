import numpy as np
import math
from matplotlib.path import Path
import pandas as pd


def coordgrid3D(isocenter, spacing, shape, start_coord, Smod):
    """Makes a grid with the x, y and z locations of three different array
    dimensions, scaled to the DICOM dimensions.

    :param isocenter: isocenter location of DICOM object
    :param spacing: three dimensional voxel dimension
    :param shape: 3 dimensional shape of DICOM object
    :param start_coord: location of voxel [0,0,0]
    :return grid: numpy array with location data
    """
    vec1 = np.linspace(0, shape[0] - 1, shape[0])
    vec2 = np.linspace(0, shape[2] - 1, shape[2])
    vec3 = np.linspace(0, shape[1] - 1, shape[1])
    grid = np.zeros([shape[0], shape[2], shape[1], 3])
    for i in range(len(vec1)):
        for j in range(len(vec2)):
            for k in range(len(vec3)):
                grid[i, j, k, 0] = vec1[i]
                grid[i, j, k, 1] = vec2[j]
                grid[i, j, k, 2] = vec3[k]
    grid = grid * spacing[0]
    grid[:, :, :, 0] = grid[:, :, :, 0] - isocenter[0] - Smod[0] * spacing[0] + start_coord[0]
    grid[:, :, :, 1] = (grid[:, :, :, 1] - isocenter[1] - Smod[2] * spacing[0] + start_coord[1])*-1
    grid[:, :, :, 2] = grid[:, :, :, 2] - isocenter[2] - Smod[1] * spacing[0] + start_coord[2]
    return grid


def sourcelocation(SAD, ang):
    """Calculates the location of the source depending on the angle of
    the beam

    :param SAD: distance from source to isocenter
    :param ang: beamangle
    :return Sloc: 3D source location
    """
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    Sloc = np.array([0, -SAD])
    Sloc = R.dot(Sloc)
    Sloc = np.array([Sloc[0], Sloc[1], 0])
    return Sloc


def hitchecker(array, SAD, ang, MLCpath, ext, pos, isocenter):
    """Defines which voxels are hit by a parallel source field after
    collimation with the multileaf collimator.

    :param array: array with cartesian voxel locations
    :param SAD: distance from source to isocenter
    :param ang: beamangle
    :param MLCpath: 2D path of the MLC leaf positions
    :param ext: numpy boolean mask of body voxels
    :param pos: position of upper left voxel
    :param isocenter: isocenter position
    :return out_arr: Array with hit TERMA voxels
    """
    out_arr = np.zeros([array.shape[0], array.shape[1], array.shape[2]])
    Sourceloc = sourcelocation(SAD, ang)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    entry, vox, unit = surface_entry(array.copy(), ext, pos, isocenter, Sourceloc)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                rloc = array[i, j, k, :]
                rloc_rot = R.dot(rloc[0:2])
                locz = rloc[2]
                locx = rloc_rot[0]
                if ext[i, j, k]:
                    if MLCpath.contains_point([locx, locz]):
                        out_arr[i, j, k] = TERMA([i, j, k], entry, array, unit)
    return out_arr


def ccconv(hit_arr, ext, sourceloc):
    """Collapsed cone convolution algorithm to calculate dose as
    result of scatter.

    :param hit_arr: Array with voxel TERMA values
    :param ext: numpy boolean mask of body voxels
    :param sourceloc: location of the source in cartesian coord
    :return out: Voxel values after collapsed cone convolution.
    """
    trail, angle, dis = stand_conesum()
    #Initializing
    trailsize = trail.shape[0]
    off = np.floor(trailsize/2).astype('int')
    pad = np.zeros([hit_arr.shape[0]+trailsize, hit_arr.shape[1]+trailsize, hit_arr.shape[2]+trailsize])
    out = np.zeros([hit_arr.shape[0], hit_arr.shape[1], hit_arr.shape[2]])
    Nang = len(angle)
    params = np.zeros([Nang, 4])
    dist_a = np.zeros([11, Nang])
    mask = np.zeros([trailsize,trailsize,trailsize,Nang])
    r = np.zeros([12,Nang])

    #create kernel masks
    r[1:,:] = np.cumsum(dis, axis=0)
    pad[off+1:pad.shape[0]-off,off+1:pad.shape[1]-off, off+1:pad.shape[2]-off] = hit_arr
    dist = dist_calc(trailsize)
    Voxvec = sourceloc
    Voxvec[1] = -Voxvec[1]
    phi_u = np.arctan(Voxvec[1] / Voxvec[0])
    theta_u = 0
    for ang in range(Nang):
        params[ang, :] = kernel(angle[ang,1] + theta_u, angle[ang,0] + phi_u)
        if params[ang, 1] == 0:
            params[ang, 1] = 1
        mask[:,:,:,ang] = trail[:,:,:,ang]
        mask[12,12,12,ang] = 1
        dist_a[:,ang] = dist[mask[:,:,:,ang] > 0]
    mask = mask > 0

    #Kernel calculation
    cum_kern_contr = cumkern(params, r)
    cum_kern_contr = cum_kern_contr/np.sum(cum_kern_contr)
    tot = np.zeros([trailsize,trailsize,trailsize])
    for ang in range(Nang):
        tot[mask[:, :, :, ang]] += cum_kern_contr[:, ang]
    for i in range(hit_arr.shape[0]):
        for j in range(hit_arr.shape[1]):
            for k in range(hit_arr.shape[2]):
                if ext[i, j, k]:
                    filtout = pad[i:i + trailsize, j:j + trailsize, k:k + trailsize]
                    if filtout.any():
                        angvalue = np.sum(tot * filtout)
                        out[i,j,k] = angvalue
    return out


def surface_entry(loc_array_c, ext, start_coord, isocenter, sourceloc):
    """Calculates the location of the beam entering the body of
    the patient on the vector to the isocenter.

    :param loc_array_c: Array with 3D position values
    :param ext: numpy boolean mask of body voxels
    :param start_coord: location of voxel [0,0,0]
    :param isocenter: location of the isocenter
    :param sourceloc: location of the source in cartesian coord
    :return loc: coordinates of entry voxel
    :return vox: voxel number of entry voxel
    """
    shape = loc_array_c.shape
    vec1 = np.linspace(0, shape[0] , shape[0] + 1)*4 - isocenter[0] + start_coord[0] - 2
    vec2 = (np.linspace(0, shape[1] , shape[1] + 1)*4 - isocenter[1] + start_coord[1] - 2)*-1
    vec3 = np.linspace(0, shape[2] , shape[2] + 1)*4 - isocenter[2] + start_coord[2] - 2
    vec = {0: vec1, 1: vec2, 2: vec3}
    vox = [0,0,0]
    for i in range(3):
        vox[i] = np.argmin(abs(vec[i]))
    VoxSourceVec = sourceloc - loc_array_c[vox[0],vox[1],vox[2]]
    VSVunit = VoxSourceVec/(np.sqrt(sourceloc[0]**2 + sourceloc[1]**2))
    VSVunit[1] = -VSVunit[1]
    dir = (VSVunit > 0)*2 - 1
    voxdir = (VSVunit > 0)*2 - 1
    voxdir[1] = voxdir[1]*-1
    loc = loc_array_c[vox[0], vox[1], vox[2]]
    while ext[vox[0], vox[1], vox[2]]:
        mindiff = np.zeros([3])
        for i in range(len(dir)):
            dif = (vec[i] - loc[i])*dir[i]
            mindiff[i] = min(dif[dif>0])
        abs_frac = abs(mindiff / VSVunit)
        step = min(abs_frac)
        ax = np.argmin(abs_frac)
        for i in range(len(dir)):
            loc[i] = VSVunit[i]*step + loc[i]
        vox[ax] = vox[ax] + voxdir[ax]
    return loc, vox, VSVunit


def TERMA(VoxLoc, entry, loc_array, unit):
    mu = 0.002
    vec = loc_array[VoxLoc[0],VoxLoc[1],VoxLoc[2]] - entry
    dirdot = np.dot(unit, vec)
    dis = np.sqrt(dirdot**2)
    relTERMA = math.exp(-mu*dis)
    return relTERMA


def stand_conesum():
    """Calculate which voxels are hit in per collapsed cone trail.
    Also includes the distances and angles of the trails

    :return axtrail: Numpy array with trails of all cones
    :return angle: Array of angles of the beams
    :return dis: Numpy array with distances of the voxels to the center voxel
    """
    az = 24
    alt = 12
    degaz = np.linspace(0, az - 1, az)* 360 / az
    degalt = np.linspace(0, alt - 1, alt) * 360 / alt
    angaz = 2 * math.pi * degaz / 360
    angalt = 2 * math.pi * degalt / 360
    phi = math.pi/8
    theta = math.pi/8
    tunit = np.zeros([3])
    axtrail = np.zeros([23,23,23,az*alt])
    dis = np.zeros([11,az*alt])
    angle = np.zeros([az*alt,2])
    n = 0
    for i in range(az):
        tphi = phi + angaz[i]
        for j in range(alt):
            ttheta = theta + angalt[j]
            tunit[0] = np.sin(ttheta) * np.cos(tphi)
            tunit[1] = np.sin(ttheta) * np.sin(tphi)
            tunit[2] = np.cos(ttheta)
            tunit = tunit/np.sqrt(tunit[0]**2 + tunit[1]**2 + tunit[2]**2)
            axtrail[:,:,:,n], dis[:,n] = stand_conetrace(tunit)
            angle[n,0] = tphi
            angle[n,1] = ttheta
            n += 1
    return axtrail, angle, dis


def stand_conetrace(unitvec):
    """Calculates the trail of a single collapsed cone using the unitvector
    of said cone.

    :param unitvec: Unit vector of the individual collapsed cone axial
    :return hit: Array with hit voxels for the axial direction
    :return dis: Array with distance values of the hit voxels
    """
    size = 23
    hit = np.zeros([size, size, size])
    vec1 = np.linspace(0, size, size + 1) * 4 - np.floor(size/2)*4 - 4
    vec2 = (np.linspace(0, size, size + 1) * 4 - np.floor(size/2)*4 - 4) * -1
    vec3 = np.linspace(0, size, size + 1) * 4 - np.floor(size/2)*4 - 4
    vec = {0: vec1, 1: vec2, 2: vec3}
    dir = (unitvec > 0)*2 - 1
    voxdir = (unitvec > 0)*2 - 1
    voxdir[1] = voxdir[1]*-1
    loc = [-2,-2,-2]
    tloc = np.zeros([12,3])
    dis = np.zeros(11)
    vox = np.array([np.ceil(size/2), np.ceil(size/2), np.ceil(size/2)],dtype='int')
    N = 0
    tloc[0,:] = loc
    while N < 11:
        mindiff = np.zeros([3])
        for i in range(len(dir)):
            dif = (vec[i] - loc[i])*dir[i]
            if voxdir[i] > 0:
                mindiff[i] = dif[dif>0][0]
            elif voxdir[i] < 0:
                mindiff[i] = dif[dif > 0][-1]
            if mindiff[i] < 10**(-10):
                mindiff[i] = 4
        abs_frac = abs(mindiff / unitvec)
        step = min(abs_frac)
        ax = np.argmin(abs_frac)
        for i in range(len(dir)):
            loc[i] = unitvec[i]*step + loc[i]
        vox[ax] = vox[ax] + voxdir[ax]
        if N<10:
            hit[vox[0],vox[1],vox[2]] = 1
        N += 1
        tloc[N, :] = loc
        dis[N-1] = np.sqrt(np.sum((tloc[N,:] - tloc[N-1,:])**2))
    return hit, dis


def dist_calc(size):
    """Calulation of distance of center of each voxel to starting voxel.

    :param size: Size of calculation array
    :return r: Array with distances
    """
    r = np.zeros([size,size,size])
    vec1 = np.linspace(0, size, size + 1) * 4 - np.ceil(size/2)*4
    vec2 = np.linspace(0, size, size + 1) * 4 - np.ceil(size/2)*4
    vec3 = np.linspace(0, size, size + 1) * 4 - np.ceil(size/2)*4
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r[i,j,k] = np.sqrt(vec1[i]**2 + vec2[j]**2 + vec3[k]**2)
    return r


def kernel(theta, phi):
    """Calculation of the kernel parameters from  table values

    :param theta: Altidudinal angle
    :param phi: Azimuthal angle
    :return params: Calculated parameters.
    """
    angle = np.arccos(np.sin(theta)*np.cos(phi)) * 360 /(2 * math.pi)
    angle = angle%360
    params = np.zeros([4])
    ang = np.argmin(abs(df_np[:,0] - angle))
    params[0] = df_np[ang,1]
    params[1] = df_np[ang,2]
    params[2] = df_np[ang,3]
    params[3] = df_np[ang,4]
    return params


def cumkern(params, r):
    """Calculates kernel values for traced voxels

    :param params: Kernel calculation parameter
    :param r: distance to voxel
    :return arr: Kernel value output
    """
    arr0 = math.pi / 12 * math.cos(math.pi / 12) * ((np.exp(r[0:11,:] * -params[:, 1]) * params[:, 0] / params[:, 1] +
           np.exp(r[0:11,:] * -params[:, 3]) * params[:, 2] / params[:, 3]) - (params[:, 0] / params[:, 1] +
                                                                       params[:, 2] / params[:, 3]))
    arr1 = math.pi / 12 * math.cos(math.pi / 12) * ((np.exp(r[1:,:] * -params[:, 1]) * params[:, 0] / params[:, 1] +
           np.exp(r[1:,:] * -params[:, 3]) * params[:, 2] / params[:, 3]) - (params[:, 0] / params[:, 1] +
                                                                       params[:, 2] / params[:, 3]))
    arr = arr0 - arr1
    return arr

# Loading kernel data to be used
df = pd.read_csv(r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Lists\newkern.csv', sep=';')
df = df.apply(lambda x: x.str.replace(',', '.'))
df = df.astype('float')
df_np = df.to_numpy()
