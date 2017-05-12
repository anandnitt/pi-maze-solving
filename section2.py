import numpy as np
import cv2
import math
import time

## Reads image in HSV format. Accepts filepath as input argument and returns the HSV
## equivalent of the image.

def findpath(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = findpath(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

def readImageHSV(filePath):
    mazeImg = cv2.imread(filePath)
    hsvImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2HSV)
    return hsvImg

## Reads image in binary format. Accepts filepath as input argument and returns the binary
## equivalent of the image.
def readImageBinary(filePath):
    mazeImg = cv2.imread(filePath)
    grayImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2GRAY)
    ret,binaryImage = cv2.threshold(grayImg,200,255,cv2.THRESH_BINARY)
    return binaryImage

##  Returns sine of an angle.
def sine(angle):
    return math.sin(math.radians(angle))

##  Returns cosine of an angle
def cosine(angle):
    return math.cos(math.radians(angle))

##  This function accepts the img, level and cell number of a particular cell and the size of the maze as input
##  arguments and returns the list of cells which are traversable from the specified cell.
def findNeighbours(filePath, level, cellnum, size):
    neighbours = []
    ############################# Add your Code Here ################################

    img=cv2.imread(filePath)
    #level=3
    #cellnum=3

    l=len(img)/2

    beh1=0
    beh2=0
    fro=0
    anclk=0
    clk=0

    radius=level*40+40 ##in terms of level

    if level==1:
        startangle=(cellnum-1)*60
        endangle=(cellnum-1)*60+60  ##in terms of level

    elif level==2:
        startangle=(cellnum-1)*30
        endangle=(cellnum-1)*30+30  ##in terms of level
    elif level==3 or level==4 or level==5:
        startangle=(cellnum-1)*15
        endangle=(cellnum-1)*15+15  ##in terms of level
    elif level==6:
        startangle=(cellnum-1)*7.5
        endangle=(cellnum-1)*7.5+7.5  ##in terms of level

    ###########    End angle for level6 is float

    if level==6:
        i=startangle
        while(i<=endangle):
            theta = math.radians(90-i) 
            x = radius*math.cos(theta) 
            y = radius*math.sin(theta) 
            m,n,o=img[(int)(x+l),(int)(y+l)]
            if((m,n,o)==(255,255,255)):
                if i<(startangle+(endangle-startangle)/2):
                    beh1+=1
                else:
                    beh2+=1
            i+=0.5
        radius=level*40

        i=startangle
        while(i<=endangle):
            theta = math.radians(90-i) 
            x = radius*math.cos(theta) 
            y = radius*math.sin(theta) 
            m,n,o=img[(int)(x+l),(int)(y+l)]
            if((m,n,o)==(255,255,255)):
                fro+=1
            i+=0.5

    else:
        for angle in range(startangle, endangle+1): 
            theta = math.radians(90-angle) 
            x = radius*math.cos(theta) 
            y = radius*math.sin(theta) 
            m,n,o=img[(int)(x+l),(int)(y+l)]
            if((m,n,o)==(255,255,255)):
                if angle<(startangle+(endangle-startangle)/2):
                    beh1+=1
                else:
                    beh2+=1
            #img[x+l,y+l]=(255,0,0)

        radius=level*40

        for angle in range(startangle, endangle+1): 
            theta = math.radians(90-angle) 
            x = radius*math.cos(theta) 
            y = radius*math.sin(theta) 
            m,n,o=img[(int)(x+l),(int)(y+l)]
            if((m,n,o)==(255,255,255)):
                fro+=1

            
            #img[x+l,y+l]=(255,0,0)

    for radius in range(level*40,level*40+40):
        theta = math.radians(90-startangle) 
        x = radius*math.cos(theta) 
        y = radius*math.sin(theta) 
        m,n,o=img[(int)(x+l),(int)(y+l)]
        if((m,n,o)==(255,255,255)):
            anclk+=1
       
       # img[x+l,y+l]=(255,0,0)

    for radius in range(level*40,level*40+40):
        theta = math.radians(90-endangle) 
        x = radius*math.cos(theta) 
        y = radius*math.sin(theta) 
        m,n,o=img[(int)(x+l),(int)(y+l)]
        if((m,n,o)==(255,255,255)):
            clk+=1
       
      #  img[x+l,y+l]=(255,0,0)
        
    if anclk>=4:
        if level==1:
            if cellnum==1:
                neighbours.append((1,6))
            else:
                neighbours.append((1,cellnum-1))
        elif level==2:
            if cellnum==1:
                neighbours.append((2,12))
            else:
                neighbours.append((2,cellnum-1))
        elif level==6:
            if cellnum==1:
                neighbours.append((6,48))
            else:
                neighbours.append((6,cellnum-1))
        else:
            if cellnum==1:
                neighbours.append((level,24))
            else:
                neighbours.append((level,cellnum-1))
                
    if clk>=4:
        if level==1:
            if cellnum==6:
                neighbours.append((1,1))
            else:
                neighbours.append((1,cellnum+1))
        elif level==2:
            if cellnum==12:
                neighbours.append((2,1))
            else:
                neighbours.append((2,cellnum+1))
        elif level==6:
            if cellnum==48:
                neighbours.append((6,1))
            else:
                neighbours.append((6,cellnum+1))
        else:
            if cellnum==24:
                neighbours.append((level,1))
            else:
                neighbours.append((level,cellnum+1))

    if fro>=4:
        if level==1:
            neighbours.append((0,0))
        elif level==4 or level==5:
            neighbours.append((level-1,cellnum))
        elif level==2 or level==3 or level==6:
            if cellnum%2==0:
                neighbours.append((level-1,cellnum/2))
            elif cellnum%2==1:
                neighbours.append((level-1,(cellnum+1)/2))

    if beh1>=3:    ## here 4 value can also become 3 if error comes
        if level==1 or level==2 or level==5:
            neighbours.append((level+1,cellnum*2-1))
        elif level==3 or level==4 and size<>1:
            neighbours.append((level+1,cellnum))

    if beh2>=3 and level<>3 and level<>4:
        if level==1 or level==2 or level==5:
            neighbours.append((level+1,cellnum*2))
        elif level==3 or level==4 and size<>1:
            neighbours.append((level+1,cellnum))
    

    #################################################################################
    return neighbours

##  colourCell function takes 5 arguments:-
##            img - input image
##            level - level of cell to be coloured
##            cellnum - cell number of cell to be coloured
##            size - size of maze
##            colourVal - the intensity of the colour.
##  colourCell basically highlights the given cell by painting it with the given colourVal. Care should be taken that
##  the function doesn't paint over the black walls and only paints the empty spaces. This function returns the image
##  with the painted cell.
def colourCell(img, level, cellnum, size, colourVal):
    ############################# Add your Code Here ################################
    
    t=np.uint8([colourVal,colourVal,colourVal])
    l=len(img)/2
    if level==0:
        startangle=0
        endangle=360
    if level==1:
        startangle=(cellnum-1)*60
        endangle=(cellnum-1)*60+60  ##in terms of level

    elif level==2:
        startangle=(cellnum-1)*30
        endangle=(cellnum-1)*30+30  ##in terms of level
    elif level==3 or level==4 or level==5:
        startangle=(cellnum-1)*15
        endangle=(cellnum-1)*15+15  ##in terms of level
    elif level==6:
        startangle=(int)((cellnum-1)*7.5)
        endangle=(int)((cellnum-1)*7.5+7.5)  ##in terms of level
    srad=level*40
    erad=level*40+40
    
    for i in range(srad,erad):
        for j in range(startangle,endangle):
            theta = math.radians(90-j) 
            x = i*math.cos(theta) 
            y = i*math.sin(theta)
            m,n,o=img[(int)(x+l),(int)(y+l)]
            if m==255:
                img[(int)(x+l),(int)(y+l)]=colourVal


    #################################################################################  
    return img

##  Function that accepts some arguments from user and returns the graph of the maze image.
def buildGraph(filePath,coords,size):      ## You can pass your own arguments in this space.
    
    ############################# Add your Code Here ################################
    graph = []
    
    graph={(j[0],j[1]):findNeighbours(filePath,j[0],j[1],size) for j in coords}



    #################################################################################
    return graph

##  Function accepts some arguments and returns the Start coordinates of the maze.
def findStartPoint(filePath, size):     ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    img=cv2.imread(filePath)

    l=len(img)
    
    l=l/2
    
    if size==1:
        radius=200
    else:
        radius=280
    
    for angle in range(0, 361): 
        theta = math.radians(90-angle) 
        x = radius*math.cos(theta) 
        y = radius*math.sin(theta) 
    
        m,n,o=img[(int)(x+l),(int)(y+l)]
        
        if((m,n,o)==(255,255,255)):
            if size==1:
                start=(4,(int)(angle/15)+1)
                break
            else:
                start=(6,(int)(angle/7.5)+1)
                break


    #################################################################################
    return start

##  Finds shortest path between two coordinates in the maze. Returns a set of coordinates from initial point
##  to final point.


## The findMarkers() function returns a list of coloured markers in form of a python dictionaries
## For example if a blue marker is present at (3,6) and red marker is present at (1,5) then the
## dictionary is returned as :-
##          list_of_markers = { 'Blue':(3,6), 'Red':(1,5)}
def findMarkers(filePath):             ## You can pass your own arguments in this space.
    list_of_markers = {}
    ############################# Add your Code Here ################################
    img=cv2.imread(filePath)

    l=len(img)/2  
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cnt,z=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img1=cv2.inRange(img1,(0,100,100),(10,255,255))      ####for red colour
    cnts = cv2.findContours(img1.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        
    if len(cnts)>0:
        c=max(cnts,key=cv2.contourArea)
        M=cv2.moments(c)
        (p,q) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    p=p-l
    q=q-l



    level1=(int)((math.sqrt(p*p+q*q))/40)

    m=abs(q)/abs((p*1.0))

    x=math.atan(m)





    x=x*180/3.14

    if p<0 and q>0:
        x=180-x
    elif p>0 and q<0:
        x=360-x

    elif p<0 and q<0:
        x=180+x
       
    if x>360:
        
        x=x-(360*(int(x/360)))



    if level1==1:
          
            cellnum1=(int)(x/60+1)
    elif level1==2:

           cellnum1=(int)(x/30+1)

    elif level1==3 or level1==4 or level1==5:

            cellnum1=(int)(x/15+1)
    elif level1==6:

            cellnum1=(int)(x/7.5+1)

    img=cv2.imread(filePath)

    l=len(img)/2  
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cnt,z=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img1=cv2.inRange(img1,(110,50,50),(130,255,255))      ####for blue colour
    cnts = cv2.findContours(img1.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        
    if len(cnts)>0:
        c=max(cnts,key=cv2.contourArea)
        M=cv2.moments(c)
        (p,q) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    p=p-l
    q=q-l



    level2=(int)((math.sqrt(p*p+q*q))/40)

    m=abs(q)/abs((p*1.0))

    x=math.atan(m)





    x=x*180/3.14

    if p<0 and q>0:
        x=180-x
    elif p>0 and q<0:
        x=360-x

    elif p<0 and q<0:
        x=180+x
       
    if x>360:
        
        x=x-(360*(int(x/360)))



    if level2==1:
          
            cellnum2=(int)(x/60+1)
    elif level2==2:

           cellnum2=(int)(x/30+1)

    elif level2==3 or level2==4 or level2==5:

            cellnum2=(int)(x/15+1)
    elif level2==6:

            cellnum2=(int)(x/7.5+1)

    list_of_markers = { 'Blue':(level2,cellnum2), 'Red':(level1,cellnum1)}
    


    #################################################################################
    return list_of_markers

## The findOptimumPath() function returns a python list which consists of all paths that need to be traversed
## in order to start from the START cell and traverse to any one of the markers ( either blue or red ) and then
## traverse to FINISH. The length of path should be shortest ( most optimal solution).
def findOptimumPath(graph,listofMarkers,start):     ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    path1=[]
    path2=[]
    pathArray=[]
    (rlevel,rcellnum)=listofMarkers['Red']
    (blevel,bcellnum)=listofMarkers['Blue']
    pathr1=findpath(graph,start,(rlevel,rcellnum))
    pathr2=findpath(graph,(rlevel,rcellnum),(0,0))
    
    pathb1=findpath(graph,start,(blevel,bcellnum))
    pathb2=findpath(graph,(blevel,bcellnum),(0,0))
    
    #print (len(pathr1)+len(pathr2)),(len(pathb2)+len(pathb1))
    if ((len(pathr1)+len(pathr2))<(len(pathb2)+len(pathb1))):
            pathArray.append(pathr1)
            pathArray.append(pathr2)
##        for i in range(0,len(pathr1)):
##            j,l=pathr1[i]
##            pathArray.append((j,l))
##        for i in range(0,len(pathr2)):
##            j,l=pathr2[i]
##            pathArray.append((j,l))
    elif((len(pathr1)+len(pathr2))>(len(pathb2)+len(pathb1))):
            pathArray.append(pathb1)
            pathArray.append(pathb2)
##        for i in range(0,len(pathb1)):
##            j,l=pathb1[i]
##            pathArray.append((j,l))
##        for i in range(0,len(pathb2)):
##            j,l=pathb2[i]
####            pathArray.append((j,l))

    elif((len(pathr1)+len(pathr2))==(len(pathb2)+len(pathb1))):
            if(len(pathr1)<len(pathb1)):
                    pathArray.append(pathr1)
                    pathArray.append(pathr2)
##                    for i in range(0,len(pathr1)):
##                        j,l=pathr1[i]
##                        pathArray.append((j,l))
##                    for i in range(0,len(pathr2)):
##                        j,l=pathr2[i]
##                        pathArray.append((j,l))
            else:
                    pathArray.append(pathb1)
                    pathArray.append(pathb2)
##                    for i in range(0,len(pathb1)):
##                        j,l=pathb1[i]
##                        pathArray.append((j,l))
##                    for i in range(0,len(pathb2)):
##                        j,l=pathb2[i]
##                        pathArray.append((j,l))

                    

    #print pathArray
    #################################################################################
    return pathArray

## The colourPath() function highlights the whole path that needs to be traversed in the maze image and
## returns the final image.
def colourPath(img,path,size):   ## You can pass your own arguments in this space. 
    ############################# Add your Code Here ################################
    for i in path:
        
        
        img = colourCell(img,i[0],i[1], size, 150)


    #################################################################################
    return img

#####################################    Add Utility Functions Here   ###################################
##                                                                                                     ##
##                   You are free to define any functions you want in this space.                      ##
##                             The functions should be properly explained.                             ##




##                                                                                                     ##
##                                                                                                     ##
#########################################################################################################

## This is the main() function for the code, you are not allowed to change any statements in this part of
## the code. You are only allowed to change the arguments supplied in the findMarkers(), findOptimumPath()
## and colourPath() functions.    
def main(filePath, flag = 0):
    img = readImageHSV(filePath)
    imgBinary = readImageBinary(filePath)
    img1=cv2.imread(filePath)
    if len(img) == 440:
        size = 1
        coords=[(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),
(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(2,11),(2,12),
(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),
(3,13),(3,14),(3,15),(3,16),(3,17),(3,18),(3,19),(3,20),(3,21),(3,22),(3,23),(3,24),
(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),
(4,13),(4,14),(4,15),(4,16),(4,17),(4,18),(4,19),(4,20),(4,21),(4,22),(4,23),(4,24)]
               

    else:
        size = 2
        coords=[(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),
(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(2,11),(2,12),
(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),
(3,13),(3,14),(3,15),(3,16),(3,17),(3,18),(3,19),(3,20),(3,21),(3,22),(3,23),(3,24),
(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),
(4,13),(4,14),(4,15),(4,16),(4,17),(4,18),(4,19),(4,20),(4,21),(4,22),(4,23),(4,24),
(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),
(5,13),(5,14),(5,15),(5,16),(5,17),(5,18),(5,19),(5,20),(5,21),(5,22),(5,23),(5,24),
(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,12),
(6,13),(6,14),(6,15),(6,16),(6,17),(6,18),(6,19),(6,20),(6,21),(6,22),(6,23),(6,24),
(6,25),(6,26),(6,27),(6,28),(6,29),(6,30),(6,31),(6,32),(6,33),(6,34),(6,35),
(6,36),(6,37),(6,38),(6,39),(6,40),(6,41),(6,42),(6,43),(6,44),(6,45),(6,46),(6,47),(6,48)]
                        

    listofMarkers = findMarkers(filePath)
    start=findStartPoint(filePath, size)
    graph=buildGraph(filePath,coords,size)
    path = findOptimumPath(graph,listofMarkers,start)
    
    img = colourPath(img1,path[0],size)
    img = colourPath(img1,path[1],size)

    
    print path
    print listofMarkers
    if __name__ == "__main__":                    
        return img
    else:
        if flag == 0:
            return path
        elif flag == 1:
            return str(listofMarkers) + "\n"
        else:
            return img
    
## The main() function is called here. Specify the filepath of image in the space given.
if __name__ == "__main__":
    filePath = "image_09.jpg"     ## File path for test image
    img = main(filePath)           ## Main function call
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
