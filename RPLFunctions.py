#import all necessary packages used in code
import numpy as np
import random
from sensor_msgs.msg import LaserScan
import rospy
import subprocess

#function returns scanvals - the variable holding formatted scan data [intensities angles ranges]
def getScan():
    #waits for a pubished message from the /scan topic - (topic, data type, timeout (sec))
    scan = rospy.wait_for_message('/scan', LaserScan, 15) 

    #yields range data (float32[]) -- ranges is an array which gives the distance measured for each angle bin
    #Ranges are returned in meters -- convert to mm 
    ranges = np.array(scan.ranges)*1000

    #np.array(dataype?, copy(bool): True - object is copied/False - copy only made if array returns copy, subok(bool): True - subclasses will be passed through/False - forced to be base-class array, ndmn(int): specifies minimum number of dimensions resulting array should have)
    #[:] -- colon returns all indicies in the array
    ranges = np.array(ranges[:], copy=False, subok=True, ndmin=2).T

    #yields the intensity data (float32[]) -- intensities of 0 return no value, as the intensity increases the quality of the reading improves
    intensities = np.array(scan.intensities)
    intensities = np.array(intensities[:], copy=False, subok=True, ndmin=2).T
    
    #converts the minimum/maximum angle (angle at which the first/last scan point is taken) from the scan from radians to degrees (float32)
    anglemin = np.rad2deg(scan.angle_min) 
    anglemax = np.rad2deg(scan.angle_max)

    #yields the angular resolution or the angle between one scan point and the next
    angleincr = np.rad2deg(scan.angle_increment)
    angles = []
   
    #the LaserScan message reports a list of ranges and the angle increment -- calculates the individual angles corresponding to their range values
    for i in range(len(scan.ranges)):
        angles.append(anglemin + angleincr*i)
    angles = np.array(angles[:], copy=False, subok=True, ndmin=2).T
    
    #sets up the scanvals variable as a numpy array with [intensities angles ranges]
    scanvals = np.concatenate((intensities,angles,ranges),axis=1)
    #deletes rows where intesnity == 0 
    scanvals = np.delete(scanvals, np.where(scanvals[:,0]==0),axis=0)
    #inserts a fourth column of zeros for later use
    scanvals = np.insert(scanvals, 3, 0, axis=1)
    return scanvals

#executes a child program as a new process which starts a rplidar node and runs rplidar client process to print raw scan data 
def start():
    roslaunch_command = 'roslaunch rplidar_ros rplidar.launch'.split()
    return subprocess.Popen(roslaunch_command)

#kills a subprocess
def kill(process):
    process.kill()

#returns an array of x and y values    
def scan2coord():
    #calls the function "getScan()" created and explained above
    scanvals = getScan()

    #takes data from scanvals converts to radians and assigns it as either the x or y value respectively
    #[:,1] -- returns an array of each value in all rows of column 1
    x = (scanvals[:,2])*(np.sin(np.deg2rad(scanvals[:,1])))
    y = (scanvals[:,2])*(np.cos(np.deg2rad(scanvals[:,1])))

    #Return all X and Y values in the form of an array (X column and Y column)
    return np.c_[x,y]

#initiates the listener node
def listener():
    return rospy.init_node('RPL_Listen')

#breaks the scan data into segments to be processed
def segment(scanvals, segthreshold):
    i=0 #incremental number
    iterseg = 0 #segment number
   #Compare the range (column 2 in scanvals) from one row to the next. If the difference is greater than segthreshold, increment the segment number. If not, assign it the same segment number.
    for row in scanvals:
        if abs(scanvals[i][2] - scanvals[i-1][2]) > segthreshold:
            iterseg+=1
            scanvals[i][3] = iterseg
            i+=1
        else:
            scanvals[i][3] = iterseg
            i+=1
        if i >= np.size(scanvals, axis=0):
            break
    return scanvals

# Returns rotation matrix for angle of rotation rot
def getRotMat(rot):
    return np.array(((np.cos(rot),-1*np.sin(rot)),(np.sin(rot),np.cos(rot))))
###############################################################################
# Performs gradient descent for circle curve fit
# trans: data points of potential circle
# (x0,y0): starting guess for circle center
# r: starting guess for circle radius
# Returns center and radius of best fit circle as well as regression error E
def descent(trans, x0, y0, r):
    mu = 0.002
    E = 200
    oldE = 0
    count = 0
    print(x0,y0,r)
    print('here')
    while np.abs(E - oldE) > 0.1 and count < 5000:
        count += 1
        oldE  = E
        dEdx0 = 0
        dEdy0 = 0
        dEdr  = 0
        E     = 0
        for i in range(trans.shape[0]):
            xi = trans[i,0]
            yi = trans[i,1]
            sqrt = np.absolute(np.sqrt(r**2 - (x0 - xi)**2 + 0J))
            E     += (y0 - yi + sqrt)**2
            dEdx0 += -1*((2*x0 - 2*xi)*(y0 - yi + sqrt))/sqrt
            dEdy0 += 2*y0 - 2*yi + 2*sqrt
            dEdr  += (2*r*(y0 - yi + sqrt))/sqrt
        x0 -= mu*dEdx0
        y0 -= mu*dEdy0
        r  -= mu*dEdr
        print(x0,y0,r,count)
        r = r
    E = E/trans.shape[0]
    return (x0,y0,r,E)
###############################################################################
def checkCircle(data):
    import matplotlib as plot
    # Transform data by rotation so that all data appears in the firt and
    # second quadrants (so that positive square root solution is valid)
    rot = np.arctan2(data[-1,1]-data[0,1] , data[-1,0]-data[0,0])
    R = getRotMat(rot + np.pi)
    trans = np.matmul(data,R)
    trans[:,1] *= -1

    # Display transformed data used for the actual fit
    plot.figure(2)
    plot.clf()
    plot.plot(trans[:,0],trans[:,1],'*')
    plot.axis("equal")

    thresh = 0.01
    E = 10
    count = 0
    # Perform gradient descent up to four times with slightly different guesses
    # until a low E solution is found.
    while E > thresh and count<4:
        count += 1
        xBar = np.mean(trans[:,0])*(0.95+np.random.rand(1)*0.1)[0]
        yBar = np.min(trans[:,1])*(0.95+np.random.rand(1)*0.1)[0]
        rBar = np.abs((trans[0,0] - trans[-1,0])/2)*(0.95+np.random.rand(1)*0.1)[0]
        x0,y0,r,E = descent(trans, xBar,yBar,rBar)
        print(E)
        
    # More plotting of transformed data
    circle = plot.Circle((x0,y0),r, color='r', fill = False)
    plot.gca().add_artist(circle)
    
    # Rotate back to original coordinates
    y0 *= -1
    center = np.matmul([x0,y0],np.linalg.inv(R))
    if E < thresh:
        return center, r
    else:
        return None, None
############################################################################
def getLine(data):
    thresh = 30 #threshold value for distance
    if (len(data)/3) > 40:
        spot = round(len(data)/3)
    else:
        spot = 30 #minimum number of points to form a line
    go = True
    c = None
    while go:
        if spot >= data.shape[0]:
            if c is None:
                c = np.polyfit(data[:,0],data[:,1],1) #creates coefficient matrix
                spot = data.shape[0] #spot becomes equal to number of points in segment j
            go = False
        else:
            c = np.polyfit(data[:spot,0],data[:spot,1],1) #coeefficient matrix [slope intercept]
            m = c[0] #filling slope of C
            b = c[1] #filling intercept of C
            x0 = data[spot,0]
            y0 = data[spot,1]
            dist = np.abs(m*x0+b - y0)/np.sqrt(m**2+1)
            if dist < thresh:
                spot += 1
            else:
                go = False
    return spot, c #returns No. of points in line segment and coefficient matrix
###############################################################################
# Checks to see whether points in data form a corner or not
# Returns line coefficients (line1, line2) number of points in the line 
# segments (corner, end)
def checkCorner(data):
    corner, line1 = getLine(data)
    if corner == np.size(data,0):
        return line1, None, corner, None
    remain = data[corner:,:]
    end, line2 = getLine(remain)
    if np.abs(np.abs(np.arctan(line1[0]) - np.arctan(line2[0])) - np.pi/2) < np.pi*5/180:
        return line1, line2, corner, end
    else:
        return None, None, None, None


#Given the position of a landmark (that has a known global location) from the robot's perspective, calculate the robot's global coordinates - The inverse of global2local
def local2global(RPLpose, LocalLM1):
    #Point conversion to global x coordinate
    Xg=(LocalLM1[0]*np.cos(np.deg2rad(RPLpose[2]))-LocalLM1[1]*np.sin(np.deg2rad(RPLpose[2])))+RPLpose[0] 
    #Point conversion to global y coordinate
    Yg=(LocalLM1[0]*np.sin(np.deg2rad(RPLpose[2]))+LocalLM1[1]*np.cos(np.deg2rad(RPLpose[2])))+RPLpose[1] 
    return ([Xg, Yg, RPLpose[2]])

#Given a known global robot pose and a known global landmark, calculate the position of the landmark from the robot's perspective - The inverse of local2global
def global2local(RPLpose, GlobalLM):
    import numpy as np
    #Point conversion to local x coordinate
    Xl=((-RPLpose[0]+GlobalLM[0])*np.cos(np.deg2rad(RPLpose[2])))+((-RPLpose[1]+GlobalLM[1])*np.sin(np.deg2rad(RPLpose[2]))) 
    #Point conversion to local y coordinate
    Yl=((-RPLpose[0]+GlobalLM[0])*-np.sin(np.deg2rad(RPLpose[2])))+((-RPLpose[1]+GlobalLM[1])*np.cos(np.deg2rad(RPLpose[2]))) 
    return ([Xl,Yl,0])

#Given an object's position, twist, velocity and a timestep, calculate the new position. Works with an array of n length
def newPose(RPLpose,dtheta,v,dt):
    #Change in theta
    DeltaTheta=dtheta*dt

    #calculate linear distance traveled
    Length=2*(v/DeltaTheta)*np.sin(DeltaTheta/2)
    
    #find new x and y coordinates
    x_P=Length*np.sin(np.deg2rad(RPLpose[:,2])+DeltaTheta/2)+RPLpose[:,0]
    y_P=Length*np.cos(np.deg2rad(RPLpose[:,2])+DeltaTheta/2)+RPLpose[:,1]
    
    #np.column_stack((1D array, 1D array...)) returns an array composed of a column of each array passed.
    modPart = np.column_stack((x_P,y_P,(RPLpose[:,2]+DeltaTheta),RPLpose[:,3],RPLpose[:,4],RPLpose[:,5]))
    return modPart

#Particle Filter: Given robot pose, landmarks, and number of desired particles, generate n random particles
def createRandParticle(RPLpose, landmark, n):
    import random
    #The max +/- variation of x, y and theta. Tune this!
    Xvar=100
    Yvar=100
    Tvar=15

    #Generate random numbers and add them to existing values within +/- var. random.uniform(a,b) returns a random number between a and b
    RandRPLpose=(RPLpose[0]+random.uniform(-Xvar,Xvar),RPLpose[1]+random.uniform(-Yvar,Yvar),RPLpose[2]+random.uniform(-Tvar,Tvar))

    #n times, calculate a particle position and score it based on how much x and y changed.
    RandPoseArray=np.empty((0,6))
    for p in range(n):
        Xrand=RandRPLpose[0]+random.uniform(-Xvar,Xvar)
        Yrand=RandRPLpose[1]+random.uniform(-Yvar,Yvar)
        Trand=RandRPLpose[2]+random.uniform(-Tvar,Tvar)
        #Send each random particle to global2Local, determine the the landmark's location from each particle's perspective. Score based on difference between landmark location from robot's perspective and landmark from particle's perspective.
        dx = abs(Xrand-((global2local([Xrand,Yrand,Trand],landmark))[0]))
        dy = abs((Yrand-(global2local([Xrand,Yrand,Trand],landmark))[1]))
        score = 500/(((dx+dy)/2)+500)
        RandPose=[Xrand,Yrand,Trand,score,dx,dy]
        RandPoseArray=np.vstack([RandPoseArray,RandPose])
    return RandPoseArray

#Separate scoring function, same function as within createRandPart.

def scorePart(particles, landmark):
    out = np.empty((0,6))
    #Send each random particle to global2Local, determine the the landmark's location from each particle's perspective. Score based on difference between landmark location from robot's perspective and landmark from particle's perspective.
    for i in range(len(particles)):
        Xrand = particles[i,0]
        Yrand = particles[i,1]
        Trand = particles[i,2]
        dx = abs(Xrand-((global2local([Xrand,Yrand,Trand],landmark))[0]))
        dy = abs((Yrand-(global2local([Xrand,Yrand,Trand],landmark))[1]))
        score = 500/(((dx+dy)/2)+500)
        out = np.vstack([out, [Xrand,Yrand,Trand, score, dx, dy]])
    return out

#Linear Regression formula implementation. Find line of best fit (m and b), and calculate r ("goodness of fit")
def linearRegression(data):
    #Calculate best fit line via Linear Regression
    #Formulas: m = ((XY)avg - XavgYavg) / (X^2)avg - Xavg^2)
    #Formulas: b = Yavg - mXavg
    #Formulas: r = ((XY)avg - XavgYavg) / (sqrt((X^2)avg - Xavg^2))*sqrt(f)
    Xavg = sum(data[:,0])/len(data)
    Xavg2 = (Xavg*Xavg)
    X2avg = (sum(data[:,0]*data[:,0])/len(data))
    Yavg = sum(data[:,1])/len(data)
    Yavg2 = (Yavg*Yavg)
    Y2avg = (sum(data[:,1]*data[:,1])/len(data))
    XYavg = (sum(data[:,0]*data[:,1])/len(data))
    num = XYavg - (Xavg*Yavg)
    den1 = X2avg - Xavg2
    den2 = Y2avg - Yavg2
    m = num/den1
    b = Yavg - (m*Xavg)
    r = num / (np.sqrt(den1)*np.sqrt(den2))
    return ([m,b,r])

#Get Lines
def checkCorner2(data):
    #Set a resolution value based on the size of data
    #int() truncates a decimal, rounds a decimal value down to an int.
    if(len(data)/35 > 4):
        resolution = int(len(data)/35)
    elif (len(data)/25 > 4):
        resolution = int(len(data)/25)
    elif (len(data)/15 > 4):
        resolution = int(len(data)/15)
    else:
        resolution = int(len(data)/5)
    lines = np.empty([0,3])
    if (math.floor(len(data)/resolution) >= 2):
        i=1

        #Calculate section and lines for every "resolution" within data
        for i in range(1,(resolution+1)):
            #sections the data into the next chunk of "resolution" size. Send that smaller chunk to the linearRegression function and add that to the lines array
            section = data[math.floor((len(data)/resolution))*(i-1):math.floor(((len(data)/resolution)*i)),:]
            lines = np.vstack((lines, linearRegression(section)))
        return lines
    else:
        raise ValueError("not enough data!")


#Pronounced "yiggle"
#Particle Filter: After particles are generated, scored, moved to the next pose and lotteried, add a random perterbation to each particle pose.
def jiggle(modPart, jigSize):
    #Initialize an array to store modModPart
    modModPart = np.empty((0,6))
    #For each row in modPart, add a random movement of within +/- jigsize to the x, y, and theta of modpart. Stack these values into modModPart.
    for i in range(len(modPart)):
        row = [modPart[i,0]+random.uniform(-jigSize[0],jigSize[0]),(modPart[i,1]+random.uniform(-jigSize[0],jigSize[0])),(modPart[i,2]+random.uniform(-jigSize[1],jigSize[1])),(modPart[i,3]),(modPart[i,4]),(modPart[i,5])]
        modModPart = np.vstack((modModPart,row))
    return modModPart

