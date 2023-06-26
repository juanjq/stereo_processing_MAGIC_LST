import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import os
import shutil
import matplotlib.colors as colors
from datetime import datetime
from scipy.optimize import fsolve

# --- logging --- #
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


##############################################
# --- create a directory if don't exists --- #
##############################################
def createdir(path):
    '''
    create a directory if doesnt exists
    
    Input
    ------------
    --path:
            path of the directory
    
    Output
    ------------

    '''
    
    # checking if exists first
    if not os.path.exists(path):
        os.makedirs(os.path.join(path), exist_ok=True)    
        
        
#########################################
# --- create transparented colormap --- #
#########################################
def transparent_cmap(cmap, ranges=[0,1]):
    '''
    retuns a colormap object turned to transparent
    
    Input
    ------------
    --cmap: 
            initial colormap
    
    --ranges:
            the range you want to return the colormap [0,1] the normal.
            [0,0.5] for example, the transparent starts at 0.5
    
    Output
    ------------
    --transparent_cmap:

    '''
    
    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))
    color_array[:,-1] = np.linspace(*ranges, ncolors)
    
    # building the colormap
    return colors.LinearSegmentedColormap.from_list(name='cmap', colors=color_array)


#############################################
# --- getting the limits of a dataframe --- #
#############################################
def get_limits(ax, plottype='2d'):
    '''
    given an axes object returns the limits of the plot
    
    Input
    ------------
    --ax:
    
    --plottype:
            2d, 2 dimensions or 3d, 3 dimensions
    
    Output
    ------------
    --limits:
            the limits in the coordinates

    '''
    
    # if is 2 dim or 3 dim
    if plottype == '2d':
        xm, xM = ax.get_xlim()
        ym, yM = ax.get_ylim()
        
        return (xm, xM), (ym, yM)
    if plottype == '3d':
        xm, xM = ax.get_xlim()
        ym, yM = ax.get_ylim()
        zm, zM = ax.get_zlim()
        
        return (xm, xM), (ym, yM), (zm, zM)

    
####################################
# --- dataset masking function --- #
####################################
def mask_3d(x, y, z, lims):
    '''
    function to compte a mask to apply a dataset and eliminate the points outside the given limits
    
    Input
    ------------
    --x, y, z: 
            data of each coordinate
            
    --lims: (xmin, xmax, ymin, ymax, zmin, zmax)
            limits to filter the dataset
            
    Output
    ------------
    --mask:

    '''
    
    xmin, xmax, ymin, ymax, zmin, zmax = lims
    
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
    

#########################################################
# --- function to generate a limits for a histogram --- #
#########################################################
def rangehist(x, y, sigma=1):
    '''
    function to generate limits of data of a histogram in terms of the gaussian distribution
    
    Input
    ------------
    --x: array
            array of x data
            
    --y: array
            array of y data
    
    --sigma: float
            times the standard deviation where the range will extend
            the limits. By default = 1
            
    Output
    ------------
    --[[minx, maxx], [miny, maxy]]:
            limits in x and y

    '''
    
    mhx, mhy = np.mean(x), np.mean(y)
    sx,  sy  = np.std(x), np.std(y)

    return [[mhx - sx * sigma, mhx + sx * sigma], [mhy - sy * sigma, mhy + sy * sigma]]


#######################################
# --- finding limits for the data --- #
#######################################
def lim_plot(ref_lower, ref_upper, factor):
    '''
    finding x, y, z limits for a data sample
    
    Input
    ------------
    --ref_lower: (float, float, float)
            lower coordinate in the dataset in x, y, z
            
    --ref_upper: (float, float, float)
            higher coordinate in the dataset in x, y, z
            
    --factor: (float, float, float)
            the fraction of the extension of the data in each coordinate 
            that the limits will extend. 0 will be exactly fitting the data 
    
    Output
    ------------
    --xmin, xmax: float, float
    --ymin, ymax: float, float
    --zmin, zmax: float, float
            limits in each coordinate
    '''
    
    fx, fy, fz = factor
    
    # the absolute extension od the data in each coordinate
    delta_x, delta_y, delta_z = [abs(ref_lower[i] - ref_upper[i]) for i in range(len(ref_lower))]
    
    # reference points
    xrefs = [ref_lower[0], ref_upper[0]]
    yrefs = [ref_lower[1], ref_upper[1]]
    zrefs = [ref_lower[1], ref_upper[1]]
    
    # calculating the maximums and minimums
    xmin, xmax = min(xrefs) - delta_x * fx, max(xrefs) + delta_x * fx
    ymin, ymax = min(yrefs) - delta_y * fy, max(yrefs) + delta_y * fy
    zmin, zmax = min(zrefs) - delta_z * fz, max(zrefs) + delta_z * fz

    return xmin, xmax, ymin, ymax, zmin, zmax


#######################################################
# --- function to move all files from a directory --- #
#######################################################
def move_files(source_folder, destination_folder):
    '''
    function to move files
    
    Input
    ------------
    --source_folder: str
            folder where we have the files we want to move
    
    --destination_folder: str
            folder where we want to put the file in
    
    Output
    ------------

    '''
    
    # iterating over all files inside the folder
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        
        # checking if is a file or a folder
        if os.path.isfile(source_path):
            destination_path = os.path.join(destination_folder, filename)
            
            # moving file by file
            shutil.move(source_path, destination_path)
            
            
####################################
# --- function to delete files --- #
####################################
def delete_directory(directory_path):
    '''
    a function that try to delete a file
    
    Input
    ------------
    --directory_path: str
            the path of the file you want to delete
    
    Output
    ------------

    '''
    
    try:
        os.rmdir(directory_path)
        logger.debug(f"Directory '{directory_path}' deleted successfully.")
        
    except OSError as error:
        logger.debug(f"Error deleting directory '{directory_path}': {error}")
        
        
#####################################
# --- lstchain version managing --- #
#####################################
def versiontuple(v):
    '''
    create a object to compare versions of github pipelines
    
    Input
    ------------
    --v: str
            version as a string
    
    Output
    ------------

    '''
    return tuple(map(int, (v.split("."))))

#################################################################
# find the maximum tuple version of a list of tuples (versions) #
#################################################################
def find_higher_version(versions):  
    '''
    function to compare versions
    
    Input
    ------------
    --versions:
            versions arrays
    
    Output
    ------------
    --higher_version:
            higher version of the given ones

    '''
    v_h = versions[0]
    
    if len(versions)!=1:
    
        for v in versions:
            if v > v_h:
                v_h = v      
    return v_h

#######################################
# --- add the date to a dataframe --- #
#######################################
def add_date(df):
    '''
    function that add the date to a dataframe that contain the segmented information
    in years, monts, days, hors, and minutes. 
    It can be extended to also seconds and so on
    
    Input
    ------------
    --df:
            the original dataframe
    
    Output
    ------------
    --df:
            the modified dataframe

    '''
    years, months, days, hours, minutes = [], [], [], [], []
    for timestamp in df['timestamp'].to_numpy():
        date = datetime.fromtimestamp(timestamp)
        years.append(date.year)
        months.append(date.month)
        days.append(date.day)
        hours.append(date.hour)
        minutes.append(date.minute)
    df['year']   = years
    df['month']  = months
    df['day']    = days
    df['hour']   = hours
    df['minute'] = minutes
    
    return df


##############################
# --- graphic parameters --- #
##############################
def params(n=15):
    '''
    function to set standard parameters for matplotlib
    
    Input
    ------------
    --n: int
            fontsize
            
    Output
    ------------
    
    '''
    plt.rcParams['font.size'] = n
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams['axes.linewidth'] = 1.9
    plt.rcParams['figure.figsize'] = (13, 7)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1.8
    plt.rcParams['ytick.major.width'] = 1.8   
    plt.rcParams['lines.markeredgewidth'] = 2
    pd.set_option('display.max_columns', None)
    

##########################################
# --- create a colormap given colors --- #
##########################################
def create_cmap(cols):
    '''

    Input
    ------------
            
    Output
    ------------
    
    '''    
    return colors.LinearSegmentedColormap.from_list('',  cols)
    
    
##########################################
# --- a class to find function roots --- #
##########################################
class RootFinder:
    '''

    Input
    ------------
            
    Output
    ------------
    
    '''   
    def __init__(self, start, stop, step=0.01, root_dtype="float64", xtol=1e-9):

        self.start = start
        self.stop = stop
        self.step = step
        self.xtol = xtol
        self.roots = np.array([], dtype=root_dtype)

    def add_to_roots(self, x):

        if (x < self.start) or (x > self.stop):
            return  # outside range
        if any(abs(self.roots - x) < self.xtol):
            return  # root already found.

        self.roots = np.append(self.roots, x)

    def find(self, f, *args):
        current = self.start

        for x0 in np.arange(self.start, self.stop + self.step, self.step):
            if x0 < current:
                continue
            x = self.find_root(f, x0, *args)
            if x is None:  # no root found.
                continue
            current = x
            self.add_to_roots(x)

        return self.roots

    def find_root(self, f, x0, *args):

        x, _, ier, _ = fsolve(f, x0=x0, args=args, full_output=True, xtol=self.xtol)
        if ier == 1:
            return x[0]
        return None
    
    
    
    
###########################################
# --- create a color gradient pallete --- #
###########################################
c1 = (5/255,5/255,153/255)
c2 = (102/255,0/255,204/255)
c3 = (255/255,51/255,204/255)
c4 = (204/255,0/255,0/255)
c5 = (255/255,225/255,0/255)

predC = [c1, c2, c3, c4, c5]

def color_cr(x, COLORS=predC):
    
    '''
    function to create a color gradient of 5 colors in this case
    
    Input
    ------------
    --n: float
            the value from 0 to 1 to assign a colour
            
    Output
    ------------
    --r, g, b: float
            the rgb values for the color to assign
    
    '''
    C1, C2, C3, C4, C5 = COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4]
    
    if x >= 0 and x <= 1/4:
        xeff = x
        r = C1[0] * (1 - 4 * xeff) + C2[0] * 4 * xeff
        g = C1[1] * (1 - 4 * xeff) + C2[1] * 4 * xeff
        b = C1[2] * (1 - 4 * xeff) + C2[2] * 4 * xeff
    elif x > 1/4 and x <= 2/4:
        xeff = x - 1/4
        r = C2[0] * (1 - 4 * xeff) + C3[0] * 4 * xeff
        g = C2[1] * (1 - 4 * xeff) + C3[1] * 4 * xeff
        b = C2[2] * (1 - 4 * xeff) + C3[2] * 4 * xeff
    elif x > 2/4 and x < 3/4:
        xeff = x - 2/4
        r = C3[0] * (1 - 4 * xeff) + C4[0] * 4 * xeff
        g = C3[1] * (1 - 4 * xeff) + C4[1] * 4 * xeff
        b = C3[2] * (1 - 4 * xeff) + C4[2] * 4 * xeff
    elif x >= 3/4 and x <= 1:
        xeff = x - 3/4
        r = C4[0] * (1 - 4 * xeff) + C5[0] * 4 * xeff
        g = C4[1] * (1 - 4 * xeff) + C5[1] * 4 * xeff
        b = C4[2] * (1 - 4 * xeff) + C5[2] * 4 * xeff
    else:
        print('Input should be in range [0 , 1]')
        
    return (r, g, b)
