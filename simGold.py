#!/usr/bin/env python

#
# Author: Rui Yan <yan49@purdue.edu>, Oct 2014
# Copyright (c) 2012 Purdue University
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
#
#

from EMAN2 import *
import numpy as np
import os, sys, math
from skimage.draw import (polygon, ellipse)
from skimage.morphology import (diamond, octagon, square, rectangle, star)
import random

def main():
	progname = os.path.basename(sys.argv[0])
	usage = """
        Simulate different shape of gold markers and add them to particles.
        You can genereate one shape or multiple shapes at once.
        Example:
        python simGold.py --imagefile ptcl_stack.hdf  --shape triangle,rectangle,square,diamond,octagon,star,ellipse,circle
        --triangle_side 70 --rect_width 50 --rect_height 10 --square_width 30 --diamond_radius 30 --octagon_m 10 --octagon_n 10
        --star_size 20 --ellipse_yradius 20 --ellipse_xradius 30 --circle_radius 40 --ptcl_radius 40 --marker_pixel 20
	--marker_pixel_offset 2 --centerShift 4 --verbose 10
        """
                
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)
        parser.add_argument("--imagefile", type=str, metavar="<filename>", dest="imagefile", \
                            help="The file containing boxed particles, the simulated gold markers will be applied to these particles.")
        #parser.add_argument("--outfile", type=str, metavar="<filename>", dest="outfile", help="the output file name.")
        parser.add_argument("--shape", type=str, metavar="['triangle', 'rectangle', 'square', 'diamond', 'octagon', 'star', 'circle', 'ellipse']", \
                            dest="shape", help="The shapes of simulated gold markers, ")
        #choices=['triangle','rectangle', 'square', 'diamond', 'octagon', 'star', 'circle', 'ellipse']
        parser.add_argument('--ptcl_radius', type=float, metavar="<n>", dest="ptcl_radius", help='Radius of real particles, this is used in "normalize.mask.circlemean".')
        parser.add_argument('--marker_pixel', type=float, metavar="<n>", dest="marker_pixel", default=1, help='How many times are the marker pixel higher than image pixel.')
        parser.add_argument('--marker_pixel_offset', type=float, metavar="<n>", dest="marker_pixel_offset", default=0, \
			    help='We will generate marker pixel in the range of (marker_pixel - marker_pixel_offset, marker_pixel + marker_pixel_offset).')
        #if shape is triangle
        parser.add_argument("--triangle_side", type=int, metavar="<n>", dest="triangle_side", help="The length of side of an equilateral triangle.")
        
        #if shape is rectangle
        parser.add_argument("--rect_width", type=int, metavar="<n>", dest="rect_width", help="The width of a rectangle.")
        parser.add_argument("--rect_height", type=int, metavar="<n>", dest="rect_height", help="The height of a rectangle.")
        
        #if shape is square
        #parser.add_argument("--half_square_width", type=int, metavar="<n>", dest="half_square_width", \
        #                    help="The half width of a square, in order to make the square_width as an even number. Well, you can also generate a square shape by setting width==height in rectangle shape.")
        parser.add_argument("--square_width", type=int, metavar="<n>", dest="square_width", help="The width of a square.")
			    
	    
        #if shape is diamond
        parser.add_argument("--diamond_radius", type=int, metavar="<n>", dest="diamond_radius", help="The radius of the diamond-shaped structuring element.")
        
        #if shape is octagon
        parser.add_argument("--octagon_m", type=int, metavar="<n>", dest="octagon_m", help="The size of the horizontal and vertical sides.")
        parser.add_argument("--octagon_n", type=int, metavar="<n>", dest="octagon_n", help="The height or width of the slanted sides.")
                            
        #if shape is star
        parser.add_argument("--star_size", type=int, metavar="<n>", dest="star_size", help="The radius of the diamond-shaped structuring element.")
        
        #if shape is ellipse
        parser.add_argument("--ellipse_yradius", type=int, metavar="<n>", dest="ellipse_yradius", help="Minor and major semi-axes. (x/xradius)**2 + (y/yradius)**2 = 1.")
        parser.add_argument("--ellipse_xradius", type=int, metavar="<n>", dest="ellipse_xradius", help="Minor and major semi-axes. (x/xradius)**2 + (y/yradius)**2 = 1.")
        
        #if shape is circle
        parser.add_argument('--circle_radius', type=float, metavar="<n>", dest="circle_radius", help='The radius of circle.')
	
	parser.add_argument("--centerShift", type=int, dest="centerShift", default=0, \
			    help="A range to shift the simulated gold from the center, 2 random numbers will be generated randomly in this range for shift in x and y direction. \
			    E.g. the particles will be shifted fro center in the range of [-4, 4] if set --centerShift 4 ")
	
        parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level, higner number means higher level of verboseness")
        parser.add_argument('--ppid', type=int, help="Set the PID of the parent process, used for cross platform PPID",default=-1)
        
        (options, args) = parser.parse_args()
	logger = E2init(sys.argv, options.ppid)
        
        imagefile=options.imagefile

        '''
        if (options.outfile):
            outfile = options.outfile
        else:
            temp = imagefile.rfind('.')
            outfile = imagefile[0:temp]+'_sim.hdf'
            #print "output filename is %s"%outfile
        '''
        temp = imagefile.rfind('.')
                        
        n=EMUtil.get_image_count(imagefile)
        d=EMData()
	d.read_image(imagefile, 0, True)
        #print d["nx"],d["ny"],n
	print "%s\t %d images \t%d x %d"%(imagefile,n,d["nx"],d["ny"])
        
        nx=d["nx"]
        ny=d["ny"]
        if (nx==ny):
            boxsize=d["nx"]
        else:
            print "ERROR: nx!=ny"
            sys.exit()
            
        ptclImg=EMData(boxsize,boxsize)
        
        maskImg = EMData(boxsize,boxsize)
        maskImg.to_zero()
        print options.shape
        shapeList = (options.shape).split(",")
        print shapeList
        
        
        if ('triangle' in shapeList):
            print "The shape of gold is triangle."
            if (not options.triangle_side):
                print "triangle_side is required!"
                
            else:
                goldMask = triangleMask(maskImg, options.triangle_side)
                if (options.verbose >= 10):
                    goldMask.write_image('triMask.hdf')
                outfile = imagefile[0:temp]+'_simTriangle.hdf'
                applyMask(imagefile, goldMask, outfile, options) 
        '''    
        if ('rectangle' in shapeList):
            print "The shape of gold is rectangle."
            if ((not options.rect_width) or (not options.rect_height)):
                print "Both rect_width and rect_height are required!"

            else:
                goldMask = rectMask(maskImg, options.rect_width, options.rect_height)
                if (options.verbose >= 10):
                    goldMask.write_image('rectMask.hdf')
                outfile = imagefile[0:temp]+'_simRectangle.hdf'
                applyMask(imagefile, goldMask, outfile, options)
	'''
	if ('rectangle' in shapeList):
            print "The shape of gold is rectangle."
            if ((not options.rect_width) or (not options.rect_height)):
                print "Both rect_width and rect_height are required!"

            else:
                goldMask = rectMask(maskImg, options.rect_width, options.rect_height)
                if (options.verbose >= 10):
                    goldMask.write_image('rectMask.hdf')
                outfile = imagefile[0:temp]+'_simRectangle.hdf'
                applyMask(imagefile, goldMask, outfile, options) 
	
	
	
        '''        
        if ('square' in shapeList):
            print "The shape of gold is square."
            if (not options.half_square_width):
                print "half_square_width is required!"
       
            else:
                goldMask = squareMask(maskImg, options.half_square_width)
                if (options.verbose >= 10):
                    goldMask.write_image('squareMask.hdf')
                outfile = imagefile[0:temp]+'_simSquare.hdf'
                applyMask(imagefile, goldMask, outfile, options)
        '''
	if ('square' in shapeList):
            print "The shape of gold is square."
            if (not options.square_width):
                print "square_width is required!"
       
            else:
                goldMask = squareMask(maskImg, options.square_width)
                if (options.verbose >= 10):
                    goldMask.write_image('squareMask.hdf')
                outfile = imagefile[0:temp]+'_simSquare.hdf'
                applyMask(imagefile, goldMask, outfile, options)
	
	
	
        if ('diamond' in shapeList):
            print "The shape of gold is diamond."
            if (not options.diamond_radius):
                print "diamond_radius is required!"

            else:
                goldMask = diamondMask(maskImg, options.diamond_radius)
                if (options.verbose >= 10):
                    goldMask.write_image('diamondMask.hdf')
                outfile = imagefile[0:temp]+'_simDiamond.hdf'
                applyMask(imagefile, goldMask, outfile, options)
            
        if ('octagon' in shapeList):
            print "The shape of gold is octagon."
            if ((not options.octagon_m) or (not options.octagon_n)):
                print "Both octagon_m and octagon_n are required!"

            else:
                goldMask = octMask(maskImg, options.octagon_m, options.octagon_n)
                if (options.verbose >= 10):
                    goldMask.write_image('octMask.hdf')
                outfile = imagefile[0:temp]+'_simOctagon.hdf'
                applyMask(imagefile, goldMask, outfile, options)
            
        if ('star' in shapeList):
            print "The shape of gold is star."
            if (not options.star_size):
                print "star_size is required!"

            else:
                goldMask = starMask(maskImg, options.star_size)
                if (options.verbose >= 10):
                    goldMask.write_image('starMask.hdf')
                outfile = imagefile[0:temp]+'_simStar.hdf'
                applyMask(imagefile, goldMask, outfile, options)
            
        if ('ellipse' in shapeList):
            print "The shape of gold is ellipse."
            if ((not options.ellipse_yradius) or (not options.ellipse_xradius)):
                print "Both ellipse_xradius and ellipse_yradius are required!"

            else:
                goldMask = ellipseMask(maskImg, options.ellipse_yradius, options.ellipse_xradius)
                if (options.verbose >= 10):
                    goldMask.write_image('ellipseMask.hdf')
                outfile = imagefile[0:temp]+'_simEllipse.hdf'
                applyMask(imagefile, goldMask, outfile, options)
            
        if ('circle' in shapeList):
            print "The shape of gold is circle."
            if (not options.circle_radius):
                print "circle_radius is required!"

            else:    
                goldMask = circleMask(maskImg, options.circle_radius)
                if (options.verbose >= 10):
                    goldMask.write_image('circleMask.hdf')
                outfile = imagefile[0:temp]+'_simCircle.hdf'
                applyMask(imagefile, goldMask, outfile, options)
            
        if (not shapeList):
            print "You have to provide a shape of simulated markers by --shape <shape>."
            sys.exit()


def applyMask(imagefile, goldMask, outfile, options):
    n=EMUtil.get_image_count(imagefile)
    d=EMData()
    d.read_image(imagefile, 0, True)
    boxsize=d["nx"]
    img=EMData(boxsize,boxsize)
    ptcl_radius=options.ptcl_radius
    marker_pixel=options.marker_pixel
    m=marker_pixel
    ############################ Do we need to put this inside the for loop?
    offset = options.marker_pixel_offset
    marker_pixel=random.uniform(m-offset,m+offset)
    
    #marker_pixel=random.uniform(m,m+2)
    #print marker_pixel
    goldMask.mult(marker_pixel)
    
    #############################
    if (options.centerShift):
	start = options.centerShift * (-1)
	end = options.centerShift
    else:
	start = 0
	end = 0	
    #print start, end
    #a, b = random.randint(start, end), random.randint(start, end)
    #this random number is needed to generate inside the for loop
    #a=random.randint(-4,4)
    #b=random.randint(-4,4)
    #goldMask.process_inplace('xform.translate.int', {'trans':(a, b)})
    #goldMask.write_image("goldMask.hdf")

    if(boxsize != goldMask.get_xsize()):
        print "ERROR: the size of simulated gold mask != the boxsize of real particles!"
        sys.exit()
    #n=3
    for i in range(n):
        img.read_image(imagefile, i)
        img.process_inplace('normalize.mask.circlemean', {'norm':1, 'mask':0, 'radius':ptcl_radius, 'masksoft':3})
	
	#########################
	a, b = random.randint(start, end), random.randint(start, end)
	#print a, b
	goldMaskTrans = goldMask.process('xform.translate.int', {'trans':(a, b)})
        
        #imgArray = EMNumPy.em2numpy(img)
        #meanVal = np.mean(imgArray)
        #m is the pixel value of simulated gold
        #m = math.fabs(meanVal) * marker_pixel
        #print meanVal, m
	
	img.add(goldMaskTrans)
        #img.add(goldMask)
        #goldMask.write_image("gold_mask.hdf")
        img.write_image(outfile, i)
        

def circleMask(maskImg, circle_radius):
    boxsize = maskImg.get_xsize()
    circleImg = EMData(boxsize,boxsize)
    circleImg.to_one()
    circleImg.process_inplace("mask.sharp", {'outer_radius':circle_radius})
    
    return circleImg
    
    
def ellipseMask(maskImg, ellipse_yradius, ellipse_xradius):
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= ellipse_yradius * 2 or boxsize <= ellipse_xradius * 2):
        print "ERROR: ellipse_xradius or ellipse_yradius is larger than the boxsize of particles."
        sys.exit()
    
    #from skimage.draw import ellipse
    #Generate coordinates of pixels within ellipse.
    ####################################
    '''
    ellipse_xradius=random.randint(ellipse_xradius-10,ellipse_xradius+10)
    print ellipse_xradius
    ellipse_yradius=random.randint(ellipse_yradius-10,ellipse_yradius+10)
    print ellipse_yradius
    '''
    size = max([ellipse_xradius, ellipse_yradius]) * 2 + 4
    cx = size/2
    cy = size/2
    ellipseArray = np.zeros((size, size), dtype=np.uint8)
    rr, cc = ellipse(cy, cx, ellipse_yradius, ellipse_xradius)
    ellipseArray[rr, cc] = 1
    
    m, n = ellipseArray.shape
    assert m==n
    
    if (m%2 == 0):
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2
    else:
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2+1
        
    ellipseArrayPad = np.pad(ellipseArray, (pad_before, pad_after), mode='constant')
    ellipseImg = EMNumPy.numpy2em(ellipseArrayPad)
    return ellipseImg


def starMask(maskImg, star_size):
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= star_size):
        print "ERROR: star size is larger than the boxsize of particles."
        sys.exit()
        
    #from skimage.morphology import star
    #Generates a star shaped structuring element that has 8 vertices and is an overlap of square of size 2*a + 1 with its 45 degree rotated version.
    #The slanted sides are 45 or 135 degrees to the horizontal axis.
    starArray = star(star_size, dtype=np.uint8)
    m, n = starArray.shape
    assert m==n
    
    if (m%2 == 0):
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2
    else:
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2+1
    
    starArrayPad = np.pad(starArray, (pad_before, pad_after), mode='constant')
    starImg = EMNumPy.numpy2em(starArrayPad)
    return starImg
    
    
        
def octMask(maskImg, octagon_m, octagon_n):
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= octagon_n * 2 or boxsize <= octagon_m):
        print "ERROR: (2 * the slanted sides) or (the horizontal and vertical sides) is larger than the boxsize of particles."
        sys.exit()
    
    #from skimage.morphology import octagon
    #Generates an octagon shaped structuring element with a given size of horizontal and vertical sides and a given height or width of slanted sides.
    #The slanted sides are 45 or 135 degrees to the horizontal axis and hence the widths and heights are equal.
    octArray = octagon(octagon_m, octagon_n, dtype=np.uint8)
    m, n = octArray.shape
    assert m==n
    
    if (m%2 == 0):
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2
    else:
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2+1
    
    octArrayPad = np.pad(octArray, (pad_before, pad_after), mode='constant')
    octImg = EMNumPy.numpy2em(octArrayPad)
    return octImg


def diamondMask(maskImg, diamond_radius):
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= (diamond_radius * 2 + 1)):
        print "ERROR: the width of the square cannot be larger than the boxsize of particles."
        sys.exit()
        
    #from skimage.morphology import diamond
    #Generates a flat, diamond-shaped structuring element of a given radius.
    #A pixel is part of the neighborhood (i.e. labeled 1) if the city block/manhattan distance between it and the center of the neighborhood is no greater than radius.
    diamArray = diamond(diamond_radius, dtype=np.uint8)
    m, n = diamArray.shape
    assert m==n
    
    if (m%2 == 0):
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2
    else:
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2+1
        
    diamArrayPad = np.pad(diamArray, (pad_before, pad_after), mode='constant')
    diamImg = EMNumPy.numpy2em(diamArrayPad)
    return diamImg

'''
def squareMask(maskImg, half_square_width):
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    square_width = half_square_width * 2
    if (boxsize <= square_width):
        print "ERROR: the width of the square cannot be larger than the boxsize of particles."
        sys.exit()
        
    #from skimage.morphology import square
    #Generates a flat, square-shaped structuring element.
    #Every pixel along the perimeter has a chessboard distance no greater than radius (radius=floor(width/2)) pixels.
    squareArray = square(square_width, dtype=np.uint8)
    
    pad_width = (boxsize - square_width)/2
    
    squareArrayPad = np.pad(squareArray, pad_width, mode='constant')
    squareImg = EMNumPy.numpy2em(squareArrayPad)
    return squareImg
'''
def squareMask(maskImg, square_width): #both odd and even square_with are allowed
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= square_width):
        print "ERROR: the width of the square cannot be larger than the boxsize of particles."
        sys.exit()
        
    #from skimage.morphology import square
    #Generates a flat, square-shaped structuring element.
    #Every pixel along the perimeter has a chessboard distance no greater than radius (radius=floor(width/2)) pixels.
    squareArray = square(square_width, dtype=np.uint8)
    m, n = squareArray.shape
    assert m==n
    
    if (m%2 == 0):
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2
    else:
        pad_before = (boxsize - m)/2
        pad_after = (boxsize - m)/2+1
    #pad_width = (boxsize - square_width)/2
    #print "m, n, pad_before, pad_after", m, n, pad_before, pad_after
    #squareArrayPad = np.pad(squareArray, pad_width, mode='constant')
    squareArrayPad = np.pad(squareArray, (pad_before, pad_after), mode='constant')
    
    squareImg = EMNumPy.numpy2em(squareArrayPad)
    return squareImg

'''       
def rectMask(maskImg, width, height):
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= width or boxsize <= height):
        print "ERROR: the width or height of the rectangle cannot be larger than the boxsize of particles."
        sys.exit()
        
    #from skimage.morphology import rectangle
    #Generates a flat, rectangular-shaped structuring element of a given width and height.
    #Every pixel in the rectangle belongs to the neighboorhood.
    rectArray = rectangle(height, width, dtype=np.uint8)
    #m, n = rectArray.shape
    
    pad_x = (boxsize - height)/2
    pad_y = (boxsize - width)/2
    rectArrayPad = np.pad(rectArray, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
    #m, n = rectArrayPad.shape
    #print m, n
    
    #convert numpy to em image
    rectImg = EMNumPy.numpy2em(rectArrayPad)
    return rectImg
'''   
    
def rectMask(maskImg, width, height):
    boxsize = maskImg.get_xsize()
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= width or boxsize <= height):
        print "ERROR: the width or height of the rectangle cannot be larger than the boxsize of particles."
        sys.exit()
        
    #from skimage.morphology import rectangle
    #Generates a flat, rectangular-shaped structuring element of a given width and height.
    #Every pixel in the rectangle belongs to the neighboorhood.
    rectArray = rectangle(height, width, dtype=np.uint8)
    m, n = rectArray.shape
    
    if (m%2 == 0):
        padRow_before = (boxsize - m)/2
        padRow_after = (boxsize - m)/2
    else:
        padRow_before = (boxsize - m)/2
        padRow_after = (boxsize - m)/2+1
	
	
    if (n%2 == 0):
        padCol_before = (boxsize - n)/2
        padCol_after = (boxsize - n)/2
    else:
        padCol_before = (boxsize - n)/2
        padCol_after = (boxsize - n)/2+1
    
    #pad_x = (boxsize - height)/2
    #pad_y = (boxsize - width)/2
    #rectArrayPad = np.pad(rectArray, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
    rectArrayPad = np.pad(rectArray, ((padRow_before, padRow_after), (padCol_before, padCol_after)), mode='constant')
    #m, n = rectArrayPad.shape
    #print m, n
    
    #convert numpy to em image
    rectImg = EMNumPy.numpy2em(rectArrayPad)
    return rectImg    


    

def triangleMask(maskImg, side):
    boxsize = maskImg.get_xsize()
    #print boxsize
    
    #convert em image to numpy
    maskArray = EMNumPy.em2numpy(maskImg)
    
    if (boxsize <= side + 2):
        print "ERROR: the side of the triangle cannot be larger than the boxsize of particles."
        sys.exit()
    
    #Generate a flat, triangle-shaped structure element 
    arraySize = boxsize
    triArray = np.zeros((arraySize, arraySize), dtype=np.uint8)
    '''
    x0 = side/2
    x1 = side
    x2 = 3 * side/2
    '''
    x0 = boxsize/2 - side/2
    x1 = boxsize/2
    x2 = boxsize/2 + side/2
    
    y0 = boxsize/2 - side * (math.sqrt(3))/6
    y1 = boxsize/2 + side * (math.sqrt(3))/3
    y2 = y0
    
    x = np.array([x0, x1, x2, x0])
    y = np.array([y0, y1, y2, y0])
    #Generate coordinates of pixels within polygon. Here the polygon is triangle
    rr, cc = polygon(y, x)
    triArray[rr, cc] = 1
    
    #pad above triArray to the same size (boxsize) of maskArray(maskImg)
    pad_width = (boxsize - arraySize)/2
    triArrayPad = np.pad(triArray, pad_width, mode='constant') # Pads with a constant value. Default is 0.
    
    #convert numpy to em image
    triImg = EMNumPy.numpy2em(triArrayPad)
    return triImg
        
        
        
        
if __name__ == "__main__":
    main()
