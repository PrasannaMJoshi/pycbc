import math
import cmath
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 
# This code has been written to generate points in parameter in the space of sine-gaussians such that
# the projection of two adjacent points on one another is greater than or equal to a given minimum 
# projection. This is done by defining a metric on the parameter space of appropriately time-shifted 
# sine-gaussians. 
#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 

#---------------------------------------------------------------------------------------------------- 
# Main parameter space generating function. This function takes into account the time-lag that the 
# sinegaussians have with respect to the template. After ignoring a few terms in the metric due to 
# excellent approximations, we get a scaled Euclidean on the space. The following section contains
# the functions to generate the parameter space itself and a wrapper function.
#---------------------------------------------------------------------------------------------------- 
def crd_trnsfrm_5(f0, q):
    ''' Function to perform coordinate transformation:
    f0, q  ----->  w, nu '''

    w = 2.0 * np.pi * f0
    nu = 2.0 * np.pi * f0 / q
    return w, nu

def inv_trnsfrm_5(w, nu):
    ''' Inverse coordinate transformation:
    w, nu  ----->  f0, q'''

    f0 = w/(2.0*np.pi)
    q = w/ nu
    return q, f0

def crd_trnsfrm_6(w, nu, chirp_m):
    ''' Function to perform coordinate transformation:
    w, nu  ----->  z, y
    chirp_m -> Chirp mass of the binary''' 

    y = np.log(nu)
    z = (w*chirp_m)**(-5.0/3.0)
    return z, y

def inv_trnsfrm_6(z, y, chirp_m):
    ''' Inverse coordinate transformation:
    z, y  -----> w, nu
    chirp_m -> Chirp mass of the binary''' 

    nu = np.exp(y)
    w = z**(-3.0/5.0)/chirp_m
    return w, nu

def parameter_space_sg(Ny, min_proj, qmin, qmax, f0min, f0max, chirp_m, c2, out):
    '''This function generates a set of points on the parameter space assuming a scaled rectangular metric on the space.
    We consider "Ny" number of lines of constant Q. Along each line of constant Q, we consider points such that the projection between
    adjacent points is greater than or equal to the given minimum projection. This function returns an array containing values of 
    constant Q, total number of points, list of Q/nu/y coordinate of points, list of corresponding f0/w/z coordinate of points.

    Ny          ->      Number of constant Q lines.
    min_proj    ->      Minimum allowed projection.
    qmin, qmax  ->      Bounds on parameter Q
    f0min,f0max ->      Bounds on parameter f0
    chirp_m     ->      Chirp mass of binary.
    c2          ->      A constant that defines the metric on the space.
    out         ->      Arguement to specify which coordinate system to return output in 
                        "qf" - Q, f0 coordinates
                        "nw" - w, nu coordinates
                        "zy" - z, y coordinates
    '''

    # Calculate the distance corresponding to the minimum projection.
    dl = 2.0*math.sqrt(1.0 - min_proj)
    
    #print("dl is =", dl)

    # Transform the bounds on the parameter space to z-y coordinate system.
    w1, nu1 = crd_trnsfrm_5(f0min, qmin)
    w2, nu2 = crd_trnsfrm_5(f0min, qmax)
    w3, nu3 = crd_trnsfrm_5(f0max, qmin)
    w4, nu4 = crd_trnsfrm_5(f0max, qmax)
    
    z1, y1 = crd_trnsfrm_6(w1, nu1, chirp_m)
    z2, y2 = crd_trnsfrm_6(w2, nu2, chirp_m)
    z3, y3 = crd_trnsfrm_6(w3, nu3, chirp_m)
    z4, y4 = crd_trnsfrm_6(w4, nu4, chirp_m)
    
    #print('z1,z2,z3,z4=',z1,z2,z3,z4)
    #print('y1,y2,y3,y4=',y1,y2,y3,y4)
    
    # Calculate number of points along the z-axis (same as the number of points along a constant Q line.) 
    
    Q_mean=qmax
    f0_mean=f0max
    GMbyc3= 4.91657e-6
    
    g_zz=((2**(-14.0/3)/Q_mean**2.0) + (9*Q_mean**2.0/(100*(2.0*np.pi*f0_mean*chirp_m)**(-10.0/3))))
    delta_z = np.sqrt((1.0-min_proj)/g_zz)
    #print "g_zz", g_zz,delta_z, chirp_m/GMbyc3
    
    z_min=np.min([z1,z2,z3,z4])
    z_max=np.max([z1,z2,z3,z4])
    y_min=np.min([y1,y2,y3,y4])
    y_max=np.max([y1,y2,y3,y4])
    
    Nz = int(math.ceil((z_max-z_min)/delta_z))
    #print("number of z points =", Nz)
    z_points = [z3+x*delta_z for x in range(Nz)]
    #print(z_points)
    
    g_yy=1.0/2
    delta_y=np.sqrt((1.0-min_proj)/g_yy)
    #print (g_yy, delta_y)
    Ny= int(math.ceil((y_max-y_min)/delta_y))
    #print("number of y points =", Ny)
    y_points = [y2+x*delta_y for x in range(Ny)]
    #print(y_points)
    
    #selecting points in the region of interest
    y_list=[]
    z_list=[]
    q_list=[]
    f0_list=[]
    slop1= -np.abs(y3-y1)/np.abs(z1-z3) #overall sign of slope is negative
    slop2= -np.abs(y4-y2)/np.abs(z2-z4) #overall sign of slope is negative
    c1=y1-slop1*z1
    c2=y2-slop2*z2
    
    for y in y_points:
         for z in z_points:
              
              if ((z>z_min) and (z<z_max) and (y<(slop1*z+c1)) and (y>(slop2*z+c2))):
                   y_list.append(y)
                   z_list.append(z)
                   w_crdnt, nu_crdnt = inv_trnsfrm_6(z, y, chirp_m)
                   q_crdnt, f0_crdnt = inv_trnsfrm_5(w_crdnt, nu_crdnt)
                   
                   q_list.append(q_crdnt)
                   f0_list.append(f0_crdnt)
                   
                   
    #print(y_list)
    #print(z_list)
    
    '''
    z_axis=np.linspace(0.0,100.0,512)
    y1=slop1*z_axis+c1
    y2=slop2*z_axis+c2
        
    plt.plot(z_axis,y1, color='k')
    plt.plot(z_axis,y2, color='k')
    plt.plot(np.ones(512)*z1,np.linspace(0.0,8.0,512), color='k') 
    plt.plot(np.ones(512)*z3,np.linspace(0.0,8.0,512), color='k') 
    plt.scatter(z_list, y_list, marker='o', color='red')
    plt.xlabel('z')
    plt.ylabel('y')
    #plt.xlim(20.0,75.0)
    #plt.ylim(4.0,8.0)
    #plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.grid(which='major', color='grey', linestyle=':')
    plt.savefig("/home/sunil.choudhary/SG_ChiSq_Proj/z_y_points.png")  
    '''
    return z_list, y_list, q_list, f0_list

def mass_split(mass1, mass2):
    ''' This function is decides the value of minimum projection to generate parameter space based on the total mass of the binary
    mass1 , mass2       ->      The component masses of the binary'''
    tot = mass1 + mass2 
    bounds = [10.0, 70.0, 90.0, 100.0, 120.0, 130.0, 170.0]
    projs = [0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    for i in range(len(bounds) - 1):
        if tot >= bounds[i] and tot < bounds[i+1]:
            mini_proj = projs[i]

    return mini_proj

def parspace_sg_wrap(par_ver, Ny, min_proj, qmin, qmax, f0min, f0max, m1, m2, c2, GMbyc3, out):
    ''' This is a wrapper function to choose the parameter space.
    par_ver         ->      Choice of the kind of parameter space - "uniform" or "totalM_based"
    Ny              ->      Number of constant Q lines. Can be a list or integer based on choice of par_ver
    min_proj    ->      Minimum allowed projection.
    qmin, qmax  ->      Bounds on parameter Q
    f0min,f0max ->      Bounds on parameter f0
    m1, m2      ->      Component masses of the binary
    c2          ->      A constant that defines the metric on the space.
    GMbyc3      ->      G*M_sun/c^3 in SI units. required for conversion of units of mass to units of time.
    out         ->      Arguement to specify which coordinate system to return output in 
                        "qf" - Q, f0 coordinates
                        "nw" - w, nu coordinates
                        "zy" - z, y coordinates
    '''

    if par_ver == 'uniform':
        # Parameter space will be sampled uniformly in the given bounds using the generator function.
        # Calculate chirp mass
        chirp_m = GMbyc3*(m1*m2)**(3.0/5.0)/(m1 + m2)**(1.0/5.0)

        return parameter_space_sg(Ny, min_proj, qmin, qmax, f0min, f0max, chirp_m, c2, out)

    elif par_ver == 'totalM_based':
        # Choose a higher value of minimum projection for binaries of higher total mass.
        # Ny here is a list of number of constant Q lines to be considered.
        Sy = math.log(qmax/qmin)/math.sqrt(2.0)
        #min_proj = mass_split(m1, m2)
        dell = 2.0*math.sqrt(1.0 - min_proj)
        Ny = int(math.ceil(Sy/dell))
        return parspace_sg_wrap('uniform', Ny, min_proj, qmin, qmax, f0min, f0max, m1, m2, c2, GMbyc3, out)

    elif par_ver == 'lowQ_tM':
        # Same version as above but defined for lowQ. (A redundant functionality. Pls ignore.)
        if m1 + m2 <= 30.0:
            return parspace_sg_wrap('uniform', Ny[0], 0.9, qmin, qmax, f0min, f0max, m1, m2, c2, GMbyc3, out)
        elif m1 + m2 > 30.0:                        
            return parspace_sg_wrap('uniform', Ny[1], 0.95, qmin, qmax, f0min, f0max, m1, m2, c2, GMbyc3, out)

    elif par_ver == 'lowQ_uni':
        # Version of uniform space for lowQ. (Again, a redundant functionality. Pls ignore.)
        return parspace_sg_wrap('uniform', Ny, min_proj, qmin, qmax, f0min, f0max, m1, m2, c2, GMbyc3, out)


#---------------------------------------------------------------------------------------------------- 
# The following are the possible parameter spaces that we could have used. The code is kept just in 
# I need to go back to it.
#---------------------------------------------------------------------------------------------------- 
def coord_trnsfrm_1(q, f0):
    w = f0
    nu = f0/q
    return w, nu

def coord_trnsfrm_2(w, nu):
    x = math.log(w)
    y = math.log(nu)
    return x, y
def inv_trnsfrm_2(x,y):
    w = math.e**x
    nu = math.e**y
    return w, nu

def inv_trnsfrm_1(w, nu):
    f0 = w
    q = w/nu
    return q, f0

def N_q_cal(q, nx, dx, dy):
    res = (dx/dy)*math.sqrt((2.0+q**2)/2.0)*nx
    res = int(math.floor(res))
    return res
    
def coord_trnsfrm_3(q, f0, chirp_m):
    w = 2*math.pi*f0*chirp_m
    nu = 2*math.pi*f0*chirp_m/q
    return w, nu

def inv_trnsfrm_3(w, nu, chirp_m):
    f0 = w/(2.0*math.pi*chirp_m)
    q = w/nu
    return q, f0

def coord_trnsfrm_4(w, nu):
    z = w**(-5.0/3.0)
    return z, nu

def inv_trnsfrm_4(z, nu):
    w = z**(-3.0/5.0)
    return w, nu

def num_int(q, xmin, xmax, c2, metric):
    c1 = (q**2 + 2.0)/4.0
    if metric == 'sc':
        Cq = c2/(c1*q**2)
    elif metric == 'pj':
        Cq = (c2/c1)*(1.0/q**2 + 1.0)
    #print(Cq*math.e**(-10.0*xmin/3.0))
    N = 100
    h = (xmax - xmin)/float(N)
    integral = 0.0
    for i in range(N):
        x_val = xmin + h*(float(i)+0.5)
        integral = integral + h*math.sqrt(1.0 + Cq*math.e**(-10.0*x_val/3.0))
        #print(integral)

    integral = integral*math.sqrt(c1)
    #print(integral)
    return integral
        
def integrand(x, cq):
    res = math.sqrt(cq*math.e**(-10.0*x/3.0) + 1.0)
    return res

def ana_int(xval, q, c2, metric):
    c1 = (q**2 + 2.0)/4.0
    if metric == 'sc':
        Cq = c2/(c1*q**2)
    elif metric == 'pj':
        Cq = (c2/c1)*(1.0/q**2 + 1.0)
    res = math.sqrt(c1)*(0.3*(2.0*(math.log(integrand(xval, Cq) + 1.0) - integrand(xval, Cq))) + xval)
    return res


def N_q_cal_2(q, nx, x_min, x_max, dy, dl, c2, metric):

    #Sy = (math.log(50.0) - math.log(5.0))/math.sqrt(2.0)
    #Sq = num_int(q, x_min, x_max, c2)
    Sq = ana_int(x_max, q, c2, metric) - ana_int(x_min, q, c2, metric)
    res = int(math.ceil(Sq / dl))
    #res = int(math.ceil(nx*Sq/Sy))
    return res

def gen_points(dl, q, nq, xmin, xmax, c2, metric):
    N = 500*nq
    hx = (xmax - xmin)/float(N)
    x_list = []
    dist_ref = -1.0*ana_int(xmin, q, c2, metric)
    x_i_prev = xmin
    i = 0
    count = 0
    while count < nq:
        if count == 0:
            target = dl/2.0 
            x_i = xmin + hx*float(i)
            dist = dist_ref + ana_int(x_i, q, c2, metric)
            if dist >= target:
                x_list.append(x_i_prev)
                dist_ref = -1.0*ana_int(x_i_prev, q, c2, metric)
                count = count + 1
            else:
                x_i_prev = x_i
                i = i + 1
        else:
            target = dl
            x_i = xmin + hx*float(i)
            dist = dist_ref + ana_int(x_i, q, c2, metric)
            if dist >= target:
                x_list.append(x_i_prev)
                dist_ref = -1.0*ana_int(x_i_prev, q, c2, metric)
                count = count + 1
            else:
                x_i_prev = x_i
                i = i + 1

    return x_list

def parameter_space_sg_2(n_x, dl, q_min, q_max, f0_min, f0_max, chirp_m, c2, metric,  out):
    '''This function generates points in parameter space of sine-gaussians. These points 
    will be used to model glitches in GW data. The procedure to produce these points involves 
    changing to a coordinate system in which metric on the space is diagonal. In this system, 
    constant Q lines are identified and number of points on each line is calculated subject to the
    condition that distance between points along constant Q direction and constant x direction is 

    q_min, q_max = range of Q-values
    f0_min, f0_max = range of f0 values
    out - string input that decides the form of output.
        'qf' - output of Q and f0 coordinates
        'nw' - output of nu and w coordinates
        'xy' - output of x and y coordinates'''
    
    # Transform limits into new coordinates 
    w_crd_min , sssss = coord_trnsfrm_3(q_min, f0_min, chirp_m)
    w_crd_max , sssss = coord_trnsfrm_3(q_max, f0_max, chirp_m)
    sssss, nu_crd_min = coord_trnsfrm_3(q_max, f0_min, chirp_m)
    sssss, nu_crd_max = coord_trnsfrm_3(q_min, f0_min, chirp_m)
    
    x_crd_min, y_crd_min = coord_trnsfrm_2(w_crd_min, nu_crd_min)
    x_crd_max, y_crd_max = coord_trnsfrm_2(w_crd_max, nu_crd_max)
    #print(z_crd_min, nu_crd_min)
    #print(z_crd_max, nu_crd_max)
    
    # Define dx and dy as length of range of x and y coordinates
    dy = y_crd_max - y_crd_min

    # To generate points in the space such that they are equidistant along constant x_crd and constant q 
    # decide number of constant q lines we need. find the value of q for which to draw a constant q line.
    # find number of points in on a constant q line for fixed nx.
    q_list = []
    #list_xcrd = []
    #list_ycrd = []
    q_crd_list = []
    f0_crd_list = []
    n_points = 0
    for k in range(n_x):
        # calculate value of q and add it to a list
        y_k = y_crd_max - dy/float(n_x)*(0.5 + float(k))
        q_k = math.e**(x_crd_min - y_k)
        q_list.append(q_k)
    
        # calculate number of points on the constant q line.
        n_q = N_q_cal_2(q_k, n_x, x_crd_min, x_crd_max, dy, dl, c2, metric)
        n_points = n_points + n_q

        # calculate the coordinates of the points and store in list.
        list_x_q = gen_points(dl, q_k, n_q, x_crd_min, x_crd_max, c2, metric)
        list_y_q = [i - math.log(q_k) for i in list_x_q]
        list_q = []
        list_f0 = []
        for i in range(len(list_x_q)):
            x_crd = list_x_q[i]
            y_crd = list_y_q[i]
            if out == 'xy':
                list_q.append(y_crd)
                list_f0.append(x_crd)
            
            if out == 'nw':
                w1, nu1 = inv_trnsfrm_2(x_crd, y_crd)
                list_q.append(nu1)
                list_f0.append(w1)

            if out == 'qf':
                w1, nu1 = inv_trnsfrm_2(x_crd, y_crd)
                q_crd, f0_crd = inv_trnsfrm_3(w1, nu1, chirp_m)
                list_q.append(q_crd)
                list_f0.append(f0_crd)
        #list_xcrd.append(list_x_q)
        #list_ycrd.append(list_y_q)
        q_crd_list = q_crd_list + list_q
        f0_crd_list = f0_crd_list + list_f0
        #list_xcrd = list_xcrd + list_x_q
        #list_ycrd = list_ycrd + list_y_q
    
    return q_list, n_points, q_crd_list, f0_crd_list

def parameter_space_sg_4(dl, qmin, qmax, f0min, f0max, chirp_m, c2, out):

    w1, nu1 = crd_trnsfrm_5(f0min, qmin)
    w2, nu2 = crd_trnsfrm_5(f0min, qmax)
    w3, nu3 = crd_trnsfrm_5(f0max, qmin)
    w4, nu4 = crd_trnsfrm_5(f0max, qmax)
    
    z1, y1 = crd_trnsfrm_6(w1, nu1, chirp_m)
    z2, y2 = crd_trnsfrm_6(w2, nu2, chirp_m)
    z3, y3 = crd_trnsfrm_6(w3, nu3, chirp_m)
    z4, y4 = crd_trnsfrm_6(w4, nu4, chirp_m)

    # Find the new threshold Q values.
    y3_new = y3 + dl*math.sqrt(2.0)/2.0
    y2_new = y2 - dl*math.sqrt(2.0)/2.0
    qmin_new = z3**(-3.0/5.0)*math.e**(-1.0*y3_new)/chirp_m
    qmax_new = z2**(-3.0/5.0)*math.e**(-1.0*y2_new)/chirp_m

    #y2_new = y2
    #y3_new = y3
    #qmin_new = qmin
    #qmax_new = qmax

    # Parameter space in y-z coordinates.
    Dz = math.sqrt(9.0*c2/25.0)*(z1 - z3)
    Nz = int(math.ceil(Dz/dl))
    z_points = [z3 + (0.5 + float(x))*(z1 - z3)/float(Nz) for x in range(Nz)]
    
    Dy = (y3_new - y2_new)/math.sqrt(2.0)
    Ny = int(math.ceil(Dy/dl))
    y_points = [y2_new + (0.5 + float(x))*(y3_new - y2_new)/float(Ny) for x in range(Ny)]

    q_crd_list = []
    f0_crd_list = []
    for i in range(Ny):
        for j in range(Nz):
            z_crd = z_points[j]
            y_crd = y_points[i]
            q_check = z_crd**(-3.0/5.0)*math.e**(-1.0*y_crd)/chirp_m
            if q_check > qmin_new and q_check < qmax_new:
                if out == 'zy':
                    q_crd_list.append(y_crd)
                    f0_crd_list.append(z_crd)
                
                if out == 'nw':
                    w_crd, nu_crd = inv_trnsfrm_6(z_crd, y_crd, chirp_m)
                    q_crd_list.append(nu_crd)
                    f0_crd_list.append(w_crd)

                if out == 'qf':
                    w_crd, nu_crd = inv_trnsfrm_6(z_crd, y_crd, chirp_m)
                    q_crd, f0_crd = inv_trnsfrm_5(w_crd, nu_crd)
                    q_crd_list.append(q_crd)
                    f0_crd_list.append(f0_crd)
    
    n_points = len(q_crd_list)
    return n_points, q_crd_list, f0_crd_list
    


# To save data generated or to plot graphs.
##for x in range(20):
##    print(format(n_x_list[x]) + '\t' + format(n_points_list[x]))
##---------------------------------------------------------------------------------------------------- 
## show or save data generated.
##print(q_list)
##print(n_points)
#for k in range(n_x):
#    #plt.plot(list_xcrd[k], list_ycrd[k], label = 'q = '+format(q_list[k], '.4f'))
#    #plt.plot(f0_crd_list[k], q_crd_list[k], label = 'q = '+format(q_list[k], '.4f'))
#    #print(format(q_list[k], '.4f'))
#    for i in range(len(list_xcrd[k])):
#        #print(format(list_xcrd[k][i], '.4f') + '\t' + format(list_ycrd[k][i], '.4f'))
#        print(format(f0_crd_list[k][i], '.3f') + '\t' + format(q_crd_list[k][i], '.3f'))
#
### create a box to represent region of parameter space under consideration (for x-y plot)
##x_val = [x_crd_min + float(i)*dx for i in range(2)]
##bot_line = [x - math.log(50.0) for x in x_val]
##top_line = [x - math.log(5.0) for x in x_val]
##end_q = [5.0, 50.0]
##x_left = [x_crd_min for i in end_q]
##x_right = [x_crd_max for i in end_q]
##left_line = [x_crd_min - math.log(x) for x in end_q]
##right_line = [x_crd_max - math.log(x) for x in end_q]
##plt.plot(x_val, bot_line)
##plt.plot(x_val, top_line)
##plt.plot(x_left, left_line)
##plt.plot(x_right, right_line)
#
### create a box to represent region of parameter space under consideration (for f0-q plot)
##f0_val = [f0_min, f0_max]
##bot_list = [q_min, q_min]
##top_list = [q_max, q_max]
##q_val = [q_min, q_max]
##left_list = [f0_min, f0_min]
##right_list = [f0_max, f0_max]
##plt.plot(f0_val, bot_list)
##plt.plot(f0_val, top_list)
##plt.plot(left_list, q_val)
##plt.plot(right_list, q_val)
#
##plt.grid()
##plt.legend()
#plt.show()
