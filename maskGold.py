#!/usr/bin/env python

# by Wen Jiang, 2014-06-30
# $Id$

import os, sys, argparse

import EMAN2
import numpy
from skimage import exposure
from skimage.segmentation import random_walker
from skimage.restoration import denoise_tv_chambolle
from skimage.filter import threshold_otsu

def main():
	args= parse_command_line()

	logid=EMAN2.E2init(sys.argv, -1)
	
	for ifi, imageFile in enumerate(args.imageFiles):
		nImage = EMAN2.EMUtil.get_image_count(imageFile)
		if nImage<1: 
			print "WARNING: 0 particles in image file %s" % (imageFile)
			continue
		imageBaseName = os.path.splitext(imageFile)[0]
		normImageFile = "%s.norm.hdf" % (imageBaseName)
		maskedImageFile = "%s.masked.hdf" % (imageBaseName)
		if args.verbose:
			print "Start processing image file %d/%d: %s (%d particles)" % (ifi+1, len(args.imageFiles), imageFile, nImage)
			if args.verbose<0:
				args.debugFile = "%s.debug.hdf" % (imageBaseName)

		for i in range(nImage):
			if args.verbose<0 or args.verbose>1:
				print "Processing image file %d/%d: %s:%d" % (ifi+1, len(args.imageFiles), imageFile, i)
		
			d = EMAN2.EMData(imageFile, i)
			if args.verbose<0:
				d.write_image(args.debugFile, -1)

			data = EMAN2.EMNumPy.em2numpy(d)
			# scikit-image requires that float image pixel values are in range [-1, 1]
			data = exposure.rescale_intensity(data, out_range=(-1, 1))
			goldmask = findGoldMask(data, args.imageFiles)

			nongoldpixels = data[numpy.where(goldmask==0)]
			mean = numpy.mean(nongoldpixels)
			sigma= numpy.std(nongoldpixels)
			data = (data-mean)/sigma	# now non-gold region has mean=0 sigma=1
			
			dnorm = EMAN2.EMNumPy.numpy2em(data)
			dnorm.write_image(normImageFile, i)
			if args.verbose<0:
				dnorm.write_image(args.debugFile, -1)
			
			dgm = EMAN2.EMNumPy.numpy2em(goldmask)
			if(args.maskpad or args.masksoft):
				dgm.process_inplace("mask.distance", {"pad":args.maskpad, "width":args.masksoft})
			dgm = 1-dgm
			dnorm *= dgm
			
			dnorm.write_image(maskedImageFile, i)
			if args.verbose<0:
				dgm.write_image(args.debugFile, -1)
				dnorm.write_image(args.debugFile, -1)

	EMAN2.E2end(logid)

def findGoldMask(data, options):
	data = denoise_tv_chambolle(data, weight=0.8, multichannel=False)
	data = exposure.rescale_intensity(data, out_range=(-1, 1))
	thresh = threshold_otsu(data)
	sigma1 = numpy.std(data[ numpy.where(data<thresh) ])
	sigma2 = numpy.std(data[ numpy.where(data>thresh) ])
	markers = numpy.zeros(data.shape, dtype=numpy.uint)
	markers[data < thresh-0.5*sigma1] = 1
	markers[data > thresh+0.5*sigma2] = 2
	labels = random_walker(data, markers, beta=10, mode='bf')
	labels[labels != 2]=0
	labels[labels == 2]=1
	
	return labels

def parse_command_line():
	description = "mask gold particle and normalize image using non-gold region"
	epilog  = "Author: Wen Jiang (jiang12@purdue.edu)\n"
	epilog += "Copyright (c) 2014 Purdue University\n"
	
	parser = argparse.ArgumentParser(description=description, epilog=epilog)

	parser.add_argument("imageFiles", nargs="+", help="inumpyut image file(s)", default="")
		
	parser.add_argument("--maskpad", metavar="<n>", type=float, help="pad the mask by this number of pixels. default to 0", default=0)

	parser.add_argument("--masksoft", metavar="<n>", type=float, help="use soft mask with this half width. default to 0", default=0)

	parser.add_argument("--verbose", metavar="<n>", type=int, help="verbose level (0, 1, 2). default to 1", default=1)
	
	args=parser.parse_args()

	if len(args.imageFiles)<1: 
		print "At least one inumpyut image is required"
		parser.print_help()
		sys.exit(-1)
	
	return args


if __name__== "__main__":
	main()


