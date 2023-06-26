import numpy as np

########################
# --- circumcenter --- #
########################
# calculus of the circumcenter of a triangle given the x and y 
# positions for each vertex
def circumcenter(xpos, ypos):
    '''
    circumcenter of a triangle
    
    Input
    ------------
    --xpos:
    --ypos:
    
    Output
    ------------
    --(ux, uy):

    ''' 
    
    ax, bx, cx, ay, by, cy = *xpos, *ypos
    
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    
    return (ux, uy)

####################
# --- incenter --- #
####################
# calculus of the incenter of a triangle given the x and y 
# positions for each vertex
def incenter(xpos, ypos):
    '''
    incenter of a triangle
    
    Input
    ------------
    --xpos:
    --ypos:
    
    Output
    ------------
    --(ux, uy):
    
    ''' 
    
    ax, bx, cx, ay, by, cy = *xpos, *ypos
    
    a = np.sqrt((bx - cx)**2 + (by - cy)**2)
    b = np.sqrt((ax - cx)**2 + (ay - cy)**2)
    c = np.sqrt((bx - ax)**2 + (by - ay)**2)
    
    ux = (a * ax + b * bx + c * cx) / (a + b + c)
    uy = (a * ay + b * by + c * cy) / (a + b + c)
    
    return (ux, uy)

####################
# --- centroid --- #
####################
# calculus of the centroid of a triangle given the x and y 
# positions for each vertex
def centroid(xpos, ypos):
    '''
    centroid of a triangle
    
    Input
    ------------
    --xpos:
    --ypos:
    
    Output
    ------------
    --(ux, uy):

    ''' 
    
    ax, bx, cx, ay, by, cy = *xpos, *ypos
    
    ux = (ax + bx + cx) / 3
    uy = (ay + by + cy) / 3
    return (ux, uy)

#######################
# --- orthocenter --- #
#######################
# calculus of the orthocenter of a triangle given the x and y 
# positions for each vertex
def orthocenter(xpos, ypos):
    '''
    othocenter of a triangle
    
    Input
    ------------
    --xpos:
    --ypos:
    
    Output
    ------------
    --(ux, uy):

    ''' 
    
    ax, bx, cx, ay, by, cy = *xpos, *ypos
    
    cbx, cby = cx - bx, cy - by
    acx, acy = ax - cx, ay - cy
    bax, bay = bx - ax, by - ay
    
    dx = -ax * cby - bx * acy - cx * bay
    dy =  ay * cbx + by * acx + cy * bax
    
    ux = ( ay**2 * cby + bx * cx * cby + by**2 * acy + ax * cx * acy + cy**2 * bay + ax * bx * bay) / dx
    uy = (-ax**2 * cbx - by * cy * cbx - bx**2 * acx + ay * cy * acx - cx**2 * bax - ay * by * bax) / dy
    
    return (ux, uy)

##################################


######################################
# --- finding focus of a ellipse --- #
######################################
def focus_points(center, a, b, angle):
    '''
    focus points of a ellipse
       
    Input
    ------------
    --center:
            center of the ellipse
    --a:
            semimajor axis lenght
    --b:
            semiminor axis lenght
    --angle:
            angle of the ellipse
            
    Output
    ------------
    --((x,y), (x,y)):
            the coordinates of the focuses
    ''' 
    
    alpha = angle
    c = np.sqrt(a**2 - b**2)
    
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    
    F1 = center + np.dot(R, np.array([-c, 0]))
    F2 = center + np.dot(R, np.array([c, 0]))
    
    return [F1[0], F2[0]], [F1[1], F2[1]]

###############################
# --- plotting an ellipse --- #
###############################
def plot_ellipse(u, v, a, b, t_rot, ax, lw=2, alpha=1):
    '''
    plotting a ellipse given the parameters
    
    Input
    ------------
    --(u,v):
            center of the ellipse
    --a:
            semimajor axis lenght
    --b:
            semiminor axis lenght
    --t_rot:
    --ax:
    
    Output
    ------------

    ''' 
    
    t        = np.linspace(0, 2 * np.pi, 100)
    Ell1     = np.array([a * np.cos(t) , b * np.sin(t)])  
    R_rot1   = np.array([[np.cos(t_rot) , -np.sin(t_rot)],[np.sin(t_rot) , np.cos(t_rot)]])  
    Ell_rot1 = np.zeros((2, Ell1.shape[1]))
    
    for i in range(Ell1.shape[1]):
        Ell_rot1[:,i] = np.dot(R_rot1, Ell1[:,i])
        
    ax.plot(u + Ell_rot1[0,:], v + Ell_rot1[1,:], 'w', lw=lw, alpha=alpha)
    

##########################
# z projection of a line #
##########################
def z_projection(x0, y0, z0, zd, az, altitude):
    '''
    projection of a line in a determined z plane
    
    Input
    ------------
    --(x0, y0, z0):
            point of the line
    --(zd, az):
            direction of the line
    --altitude:
            plane_z = altitude
    Output
    ------------
    --(x,y,z):
            intersection point

    '''     
    
    zd, az = np.deg2rad(zd), np.deg2rad(az)

    t = (altitude - z0) / np.cos(zd)
    x = x0 + t * np.cos(az) * np.sin(zd)
    y = y0 + t * np.sin(az) * np.sin(zd)
    z = altitude
    
    return x, y, z

############################
# --- plane definition --- #
############################
def z_plane(x, y, point, direction):
    '''
    z point of a given plane given x and y coordinates
    
    Input
    ------------
    --(x,y):
            the x, y coordinates
    --point:
            point of the plane
    --direction:
            direction of the plane
    
    Output
    ------------
    --z:
            the z value of the coordinates 

    ''' 
    
    x0, y0, z0 = point
    zd, az = direction
    zd, az = np.deg2rad(zd), np.deg2rad(az)
    
    ux, uy, uz = np.cos(az) * np.sin(zd), np.sin(az) * np.sin(zd), np.cos(zd)
    
    xy = (x - x0) * ux + (y - y0) * uy
    
    return z0 - xy / uz

############################################
# --- projection of a plane and a line --- #
############################################
def plane_projection(plane_point, plane_direction, xl, yl, zl, zdl, azl):
    '''
    projection of a plane and a line
    
    Input
    ------------
    --plane_point:
            point of the plane
    --plane_direction:
            director vector of the plane
    --(xl,yl,zl):
            point of the line
    --(zdl,azl):
            direction of the line
    
    Output
    ------------
    --(x,y,z):
            point of projection
    ''' 
    
    xp, yp, zp = plane_point
    zdp, azp   = plane_direction
    
    # converting to rad
    zdp, azp = np.deg2rad(zdp), np.deg2rad(azp)
    zdl, azl = np.deg2rad(zdl), np.deg2rad(azl)
    
    # director vectors
    upx, upy, upz = np.cos(azp) * np.sin(zdp), np.sin(azp) * np.sin(zdp), np.cos(zdp)
    ulx, uly, ulz = np.cos(azl) * np.sin(zdl), np.sin(azl) * np.sin(zdl), np.cos(zdl)
    ulup = upx * ulx + upy * uly + upz * ulz
    
    t = ((xp - xl) * upx + (yp - yl) * upy + (zp - zl) * upz) / ulup
    
    x = xl + t * ulx
    y = yl + t * uly
    z = zl + t * ulz
    
    return x, y, z

###############################
# --- line point distance --- #
###############################
def line_point_distance(xl, yl, zl, zd, az, point):
    '''
    the calculus of the distance of a line and a point
    
    Input
    ------------
    --(xl,yl,zl):
            point of the line
    --(zd,az):
            direcion of the line
    --point:
            point to calculate the distance
    
    Output
    ------------
    --distance:

    ''' 
    xp, yp, zp = point
    
    # converting to rad
    zd, az = np.deg2rad(zd), np.deg2rad(az)
    
    # director vectors
    ux, uy, uz = np.cos(az) * np.sin(zd), np.sin(az) * np.sin(zd), np.cos(zd)
    
    t = (xp - xl) * ux + (yp - yl) * uy + (zp - zl) * uz 
    
    x = xl + t * ux
    y = yl + t * uy
    z = zl + t * uz
    
    return dist3d(x, y, z, *point)


############################################
# --- distance calculation in 3d space --- #
############################################
def dist3d(p1x, p1y, p1z, p2x, p2y, p2z):
    '''
    3d distance
    
    Input
    ------------
    --(p1x, p1y, p1z):
            first point
    --(p2x, p2y, p2z):
            second point
    Output
    ------------
    --distance:

    ''' 
    distance = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2 + (p1z - p2z) ** 2)
    
    return distance


############################################
# --- plot a line in a given direction --- #
############################################
def line_in_direction(angle, point, ax):
    '''
    plot a line in the axes of a figure
    
    Input
    ------------
    --angle:
            anlgle of the line in degrees
    --point:
            point of the line
    --ax:
            matplotlib axes
    Output
    ------------
    --([x1,y1],[x2,y2]):
            give the points to plot the line

    ''' 
    
    # Convert angle from degrees to radians
    angle = np.deg2rad(angle)
    angle = angle % (2 * np.pi)

    # Compute direction vector
    direction_vector = np.array([np.cos(angle), np.sin(angle)])
    
    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xm, xM = xlim
    ym, yM = ylim

    
    # Compute intersections
    if angle == 0:
        x = xM
        y = point[1]
        
    elif angle <= np.pi/2 and angle >= 0:
        yintx = point[1] + np.tan(angle) * (xM - point[0])
        xinty = (yM - point[1]) / np.tan(angle) + point[0]
        
        if yM < yintx:
            y = yM
            x = xinty
        else:
            x = xM
            y = yintx
            
    elif angle <= np.pi and angle >= np.pi/2:
        yintx = point[1] + np.tan(angle) * (xm - point[0])
        xinty = (yM - point[1]) / np.tan(angle) + point[0]
        
        if xm < xinty:
            x = xm
            y = yintx
        else:
            y = yM
            x = xinty
        
    elif angle <= 3*np.pi/2 and angle >= np.pi:
        yintx = point[1] + np.tan(angle) * (xm - point[0])
        xinty = (ym - point[1]) / np.tan(angle) + point[0]
        
        if ym > yintx:
            y = ym
            x = xinty
        else:
            x = xm
            y = yintx
        
    else:
        yintx = point[1] + np.tan(angle) * (xM - point[0])
        xinty = (ym - point[1]) / np.tan(angle) + point[0]
        
        if ym > yintx:
            y = ym
            x = xinty
        else:
            x = xM
            y = yintx
        
    return [point[0], x], [point[1], y]


################################
# --- logparabola function --- #
################################
def logpar(E, *params):
    '''

    Input
    ------------
    
    Output
    ------------

    ''' 
    from gammapy.modeling.models import LogParabolaSpectralModel
    
    model = LogParabolaSpectralModel()
    return E * E * model.evaluate(E, *params)


#####################################################
# --- function to compute bin edges in logscale --- #
#####################################################
def compute_bin_edges(bin_centers):
    '''

    Input
    ------------
    
    Output
    ------------

    '''    
    bin_edges = [None]
    for i in range(len(bin_centers)-1):
        Emid = 10 ** ((np.log10(bin_centers[i]) + np.log10(bin_centers[i+1])) / 2)
        
        bin_edges.append(Emid)
        
    # we also add the first and last points
    bin_edges[0] = 10 ** (2 * np.log10(bin_centers[0]) - np.log10(bin_centers[0] + (bin_edges[1] - bin_centers[0])))
    bin_edges.append(10 ** (2 * np.log10(bin_centers[-1]) - np.log10(bin_edges[-1])))
    
    return bin_edges


##########################################
# --- compute bin errors in logscale --- #
##########################################
def compute_left_right_errors(bin_centers):
    '''

    Input
    ------------
    
    Output
    ------------

    '''    
    bin_edges = compute_bin_edges(bin_centers)
    
    error_left, error_right = [], []
    for i in range(len(bin_centers)):
        error_left.append(bin_centers[i]  - bin_edges[i])
        error_right.append(bin_edges[i + 1] - bin_centers[i])
    
    return error_left, error_right
    

############################################################
# --- function to compute bin left errors in log scale --- #
############################################################
def compute_left_errors(points, x_errors):
    '''

    Input
    ------------
    
    Output
    ------------

    '''    
    left_errors = np.zeros_like(points)
    for i in range(len(points)):
        x = points[i]
        x_err = x_errors[i]
        left_errors[i] = x - 10 ** (2 * np.log10(x) - np.log10(x + x_err))
    return left_errors


################################################################
# --- calculate the angular distance between two pointings --- #
################################################################
def angular_distance(zenith1, azimuth1, zenith2, azimuth2):
    
    zenith1, azimuth1, zenith2, azimuth2 = np.deg2rad(zenith1), np.deg2rad(azimuth1), np.deg2rad(zenith2), np.deg2rad(azimuth2)
    
    '''
    Calculates the angular total distance in 3D given two points' zenith distance and azimuth.
    The inputs should be in radians.

    Parameters:
    - zenith1 (float): zenith distance of the first point
    - azimuth1 (float): azimuth of the first point
    - zenith2 (float): zenith distance of the second point
    - azimuth2 (float): azimuth of the second point

    Returns:
    - angular_dist (float): angular total distance in 3D
    ''' 
    
    cos_angular_dist = np.sin(zenith1) * np.sin(zenith2) + np.cos(zenith1) * np.cos(zenith2) * np.cos(azimuth2 - azimuth1)
    angular_dist     = np.arccos(cos_angular_dist)
    
    return np.rad2deg(angular_dist)