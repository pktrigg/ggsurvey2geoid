#name:			survey2geoid.py
#created:		June 2032
#by:			paul.kennedy@guardiangeomatics.com
#description:	python module to scan a folder structure, extract the ellipsoidal heigths from the RAW files and create spatial datasets for creating a hydroid / geoid for a survey area
######################
#done
# create the script
# create git repo
# add to github
# create basic reader for raw kmall files to extract ellipsoidal height from the ping MRZ records
#
# extractION
#	time
#	latitude
#	datagram.longitude
#	datagram.ellipsoidHeightReRefPoint_m
#	datagram.heading
#	datagram.txTransducerDepth_m
#	datagram.z_waterLevelReRefPoint_m
#	attitudeHeave (turns out it is not required!)
# make a time series plot to see if we need to subtract heave or z_waterLevelReRefPoint_m or txTransducerDepth_m form teh ellipsoid heights

######################
#2do
# save the time series ellipsoid heights to time/height/quality/txdepth txt format
# save geoide to an XYZ file for gridding
# the ellipsoidal heights are subject to tide heights.  its is burned into the height, so we need to reduce for tide.
# module to read .tid files
# module to read zone definition files
# module to compute the tide given a zdf, tid, time and position using inverse distance weighted tides.
# module to extract tide from GSF files.  use this as a test to validate the tid reduction.
# module to replace tide corrections from gsf and replace with ellipsoid values.  this is a faster mechanism than caris.

######################
# remember: geoid is a raster surface representing the mean sea level.  ellipsoidal heigth measurements from vessel can be used to derive a geoid surface if we apply 'n' seperation
# remember: hydroid is a raster surface representing the lowest astronomical tide. https://www.icsm.gov.au/what-we-do/aushydroid

import sys
import time
import os
import tempfile
import ctypes
import fnmatch
import math
import json
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from datetime import datetime
from datetime import timedelta
from glob import glob
import uuid
import multiprocessing as mp
import pyproj
from py7k import s7kreader
from pygsf import GSFREADER
import readkml
import matplotlib.pyplot as plt

# local from the shared area...
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
import timeseries as ts
import fileutils
import geodetic
import geopackage
import geopackage_ssdm
import ssdmfieldvalue
import kmall
import multiprocesshelper
import numpy as np
##############################################################################
def main():

	parser = ArgumentParser(description='Read any Survey folder and create a OGC compliant GEOPackage in the SSDM Schema summarising the survey. This is a distillation process extracting important spatial attributes from the survey in an automated and rigorous manner.',
			epilog='Example: \n To process all files under a root folder use -i c:/foldername \n', formatter_class=RawTextHelpFormatter)
	parser.add_argument('-i', 		action='store', 		default="",		dest='inputfolder', 	help='input folder to process.')
	parser.add_argument('-o', 		action='store', 		default="",		dest='outputFilename', 	help='output GEOPACKAGE filename.')
	parser.add_argument('-s', 		action='store', 		default="1",	dest='step', 			help='decimate the data to reduce the output size. [Default: 1]')
	parser.add_argument('-odir', 	action='store', 		default="",	dest='odir', 			help='Specify a relative output folder e.g. -odir GIS')
	parser.add_argument('-opath', 	action='store', 		default="",	dest='opath', 			help='Specify an output path e.g. -opath c:/temp')
	parser.add_argument('-odix', 	action='store', 		default="",	dest='odix', 			help='Specify an output filename appendage e.g. -odix _coverage')
	parser.add_argument('-epsg', 	action='store', 		default="4326",	dest='epsg', 			help='Specify an output EPSG code for transforming from WGS84 to East,North,e.g. -epsg 4326')
	parser.add_argument('-all', 	action='store_true', 	default=True, 	dest='all', 			help='extract all supported forms of data (ie do everything).')
	parser.add_argument('-reprocess', 	action='store_true', 	default=False, 	dest='reprocess', 			help='reprocess the survey folders by re-reading input files and creating new GIS features, ignoring the cache files. (ie do everything).')
	parser.add_argument('-cpu', 		dest='cpu', 			action='store', 		default='0', 	help='number of cpu processes to use in parallel. [Default: 0, all cpu]')

	args = parser.parse_args()
	# if len(sys.argv)==1:
	# 	parser.print_help()
	# 	sys.exit(1)

	if len(args.inputfolder) == 0:
		args.inputfolder = os.getcwd() + args.inputfolder

	if args.inputfolder == '.':
		args.inputfolder = os.getcwd() + args.inputfolder

	process(args)

###############################################################################
def process(args):
	if not os.path.isdir(args.inputfolder):
		print ("oops, input is not a folder.  Please specify a survey folder.")
		return

	surveyname = os.path.basename(args.inputfolder) #this folder should be the Survey NAME
	if args.opath == "":
		args.opath = os.path.join(args.inputfolder, "GIS")

	if len(args.outputFilename) == 0:
		args.outputFilename 	= os.path.join(args.opath, args.odir, surveyname + "_SSDM.gpkg")
		args.outputFilename  	= fileutils.addFileNameAppendage(args.outputFilename, args.odix)
		args.outputFilename 	= fileutils.createOutputFileName(args.outputFilename)
	
	# create the gpkg...
	# pkpk disable for now.  we might add the geoid to SSDM as a later feature.
	gpkg = geopackage.geopackage(args.outputFilename, int(args.epsg))

	#load the python proj projection object library if the user has requested it
	geo = geodetic.geodesy(args.epsg)

	# process any and all kmall files
	mp_processKMALL(args, gpkg, geo)

	# process any and all 7k files
	mp_process7k(args, gpkg, geo)

	# process any and all GSF files
	mp_processgsf(args, gpkg, geo)

	print("Completed creation of SSDM tracks to: %s" %(args.outputFilename))

###############################################################################
def findsurveys(args):
	'''high level funciton to recursively find all the surveys in all subfolders'''
	surveys = []
	#add the base folder as a candidate in case this is the folde rth euser specified.
	if folderisasurvey(args.inputfolder):
		surveys.append(args.inputfolder)
		return surveys
	#looks like the user has specificed a higher level folder such as a root drive or project folder so scan deeply	
	print("Scanning for surveys from root folder %s ...." % (args.inputfolder))
	folders = fast_scandir(args.inputfolder)
	for folder in folders:
		if folderisasurvey(folder):
			surveys.append(folder)
	print("surveys to Process: %s " % (len(surveys)))
	return surveys

###############################################################################
def folderisasurvey(dirname):
	'''validate if this is a real survey by testing if there is a survey.mp file in the folder '''
	# filename = os.path.join(dirname, "events.txt")
	# if not os.path.exists(filename):
	# 	return False

	surveys = fileutils.findFiles2(False, dirname, "*survey.mp")
	if len(surveys)==0:
	# filename = os.path.join(dirname, "survey.mp")
	# if not os.path.exists(filename):
		return False

	return True

###############################################################################
def fast_scandir(dirname):
	subfolders = []
	for root, dirs, files in os.walk(dirname, topdown=True):
		# for name in files:
		# 	print(os.path.join(root, name))
		if root in subfolders:
			continue
		for name in dirs:
			dirname = os.path.join(root, name)
			print (dirname)
			if folderisasurvey(dirname):
				subfolders.append(dirname)
				# dirs[:] = []
				sys.stdout.write('.')
				sys.stdout.flush()
	# subfolders = [x[0] for x in os.walk(dirname)]
	# subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
	# if subfolders is not None:
	# 	for dirname in list(subfolders):
	# 		subfolders.extend(fast_scandir(dirname))
	# else:
	# 	print("oops")
	return subfolders
###############################################################################
def update_progress(job_title, progress):
	length = 20 # modify this to change the length
	block = int(round(length*progress))
	msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
	if progress >= 1: msg += " DONE\r\n"
	sys.stdout.write(msg)
	sys.stdout.flush()

# ###############################################################################
def processsurveyPlan(args, surveyfolder, gpkg, geo):
	'''import the survey.mp file into the geopackage'''
	matches = fileutils.findFiles(True, args.inputfolder, "*.kml")

	if len(matches) == 0:
		print("No KML files found for importing to survey line plan, skipping")
		return

	#create the linestring table for the survey plan
	type, fields = geopackage_ssdm.createProposed_Survey_Run_Lines()
	linestringtable = geopackage.vectortable(gpkg.connection, "surveyPlan", gpkg.epsg, type, fields)

	# table = vectortable (connection, tablename, epsg, type, fields)
	# return table

	for filename in matches:
		reader = readkml.reader(filename)
		print ("File: %s Survey lines found: %d" % (filename, len(reader.surveylines)))
		createsurveyplan(reader, linestringtable, geo)

################################################################################
def processKMALL(filename, outfilename, step):
	#now read the kmall file and return the navigation table filename

	# print("Loading KMALL Navigation...")
	r = kmall.kmallreader(filename)
	navigation = r.loadpingdata(step=1)


	rawdata = np.array(navigation)
	rawdates = rawdata[:,0]
	rawdates = np.asarray(rawdates, dtype='datetime64[s]',)

	txdepth = np.array(rawdata[:,5], dtype=float)
	waterline = np.array(rawdata[:,6], dtype=float)

	# print("load the attitude to lists...")
	# attitude = r.loadattitude()
	# timestamps = [i[0] for i in attitude]
	# list_heave = [i[7] for i in attitude]
	# csheave = ts.cTimeSeries(timestamps, list_heave)
	# # now interpolate
	# pingheave = []
	# for p in navigation:
	# 	heaveatpingtime = csheave.getValueAt(p[0])
	# 	pingheave.append(heaveatpingtime)
	# npheave = np.array(pingheave, dtype=float)

	ellipsheights = np.array(rawdata[:,3], dtype=float) 
	# ellipsheights_1 = np.array(rawdata[:,3], dtype=float) - 35
	# ellipsheights_1 = np.array(rawdata[:,3], dtype=float) - npheave - 35

	#make a moving average filter
	# kernel_size = 21
	# kernel = np.ones(kernel_size) / kernel_size
	# ellipsheights_smooth = np.convolve(ellipsheights, kernel, mode='valid')
	#	make a time series plot so we can see if the heave is ok

	from scipy.ndimage import uniform_filter1d
	ellipsheights_smooth = uniform_filter1d(ellipsheights, size=300)
	# ellipsheights_smooth_1 = uniform_filter1d(ellipsheights_1, size=21)

	plt.figure(figsize=(12,4))
	plt.grid(linestyle='-', linewidth='0.2', color='black')
	# plt.ylim(min(sm_tideA)-0.2, max(sm_tideA)+0.2, 1.0)
	plt.plot(rawdates[::2], ellipsheights[::2], color='gray', linewidth=1, label='Ellips Height')
	# plt.plot(rawdates[::2], txdepth[::2], color='red', linewidth=1, label='txdepth')
	# plt.plot(rawdates[::2], ellipsheights_smooth_1[::2], color='green', linewidth=3, label='ellipsheights_smooth_1')
	plt.plot(rawdates[::2], ellipsheights_smooth[::2], color='blue', linewidth=3, label='ellipsheights_smooth')
	# plt.plot(rawdates[::2], sm_tideD[::2], color='yellow', linewidth=1, label='savgol7')
	# plt.set_xlabel('Date')
	# plt.legend(fontsize=18)
	plt.legend(loc="upper left")
	plt.show()

	r.close()

	with open(outfilename,'w') as f:
		json.dump(navigation, f)
	
	return(navigation)

################################################################################
def process7k(filename, outfilename, step):
	#now read the kmall file and return the navigation table filename

	# print("Loading 7K Navigation...")
	r = s7kreader(filename)
	navigation = r.loadNavigation(False)
	r.close()

	with open(outfilename,'w') as f:
		json.dump(navigation, f)
	
	return(navigation)

################################################################################
def processgsf(filename, outfilename, step):
	#now read the kmall file and return the navigation table filename
	navigation = []
	# print("Loading sgy Navigation...")
	r = GSFREADER(filename)
	if (r.fileSize == 0):
		# the file is a corrupt empty file so skip
		return navigation
	navigation = r.loadnavigation()
	r.close()

	with open(outfilename,'w') as f:
		json.dump(navigation, f)
	
	return(navigation)
################################################################################
def processsegy(filename, outfilename, step):
	#now read the kmall file and return the navigation table filename
	navigation = []
	# print("Loading sgy Navigation...")
	r = segyreader(filename)
	if (r.fileSize == 0):
		# the file is a corrupt empty file so skip
		return navigation
	r.readHeader()
	navigation = r.loadNavigation()
	r.close()

	with open(outfilename,'w') as f:
		json.dump(navigation, f)
	
	return(navigation)

################################################################################
def processjsf(filename, outfilename, step):
	#now read the kmall file and return the navigation table filename

	# print("Loading jsf Navigation...")
	r = jsfreader(filename)
	navigation = r.loadNavigation(False)
	r.close()

	with open(outfilename,'w') as f:
		json.dump(navigation, f)
	
	return(navigation)

################################################################################
def mp_process7k(args, gpkg, geo):

	# boundary = []
	boundarytasks = []
	results = []

	rawfolder = os.path.join(args.inputfolder, ssdmfieldvalue.readvalue("MBES_RAW_FOLDER"))
	if not os.path.isdir(rawfolder):
		rawfolder = args.inputfolder

	# rawfilename = ssdmfieldvalue.readvalue("MBES_RAW_FILENAME")

	matches = fileutils.findFiles2(True, rawfolder, "*.s7k")

	# surveyname = os.path.basename(args.inputfolder) #this folder should be the survey NAME

	#create the linestring table for the trackplot
	type, fields = geopackage_ssdm.createSurveyTracklineSSDM()
	linestringtable = geopackage.vectortable(gpkg.connection, "SurveyTrackLine", args.epsg, type, fields)

	for filename in matches:
		root = os.path.splitext(filename)[0]
		root = os.path.basename(filename)
		outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
		os.makedirs(outputfolder, exist_ok=True)
		# makedirs(outputfolder)
		outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
		if args.reprocess:
			if os.path.exists(outfilename):
				os.unlink(outfilename)
		if os.path.exists(outfilename):
			# the cache file exists so load it
			with open(outfilename) as f:
				# print("loading file %s" %(outfilename))
				lst = json.load(f)
				results.append([filename, lst])
		else:
			boundarytasks.append([filename, outfilename])

	multiprocesshelper.log("New 7k Files to Import: %d" %(len(boundarytasks)))		
	cpu = multiprocesshelper.getcpucount(args.cpu)
	multiprocesshelper.log("Extracting S7K Navigation with %d CPU's" %(cpu))
	pool = mp.Pool(cpu)
	multiprocesshelper.g_procprogress.setmaximum(len(boundarytasks))
	# poolresults = [pool.apply_async(processKMALL, (task[0], task[1], args.step)) for task in boundarytasks]
	poolresults = [pool.apply_async(process7k, (task[0], task[1], args.step), callback=multiprocesshelper.mpresult) for task in boundarytasks]
	pool.close()
	pool.join()
	for idx, result in enumerate (poolresults):
		results.append([boundarytasks[idx][0], result._value])
		# print (result._value)

	# now we can read the results files and create the geometry into the SSDM table
	multiprocesshelper.log("Files to Import to geopackage: %d" %(len(results)))		

	multiprocesshelper.g_procprogress.setmaximum(len(results))
	for result in results:
		createTrackLine(result[0], result[1], linestringtable, float(args.step), geo)
		multiprocesshelper.mpresult("")

################################################################################
def mp_processgsf(args, gpkg, geo):

	# boundary = []
	boundarytasks = []
	results = []

	rawfolder = os.path.join(args.inputfolder, ssdmfieldvalue.readvalue("MBES_RAW_FOLDER"))
	if not os.path.isdir(rawfolder):
		rawfolder = args.inputfolder

	# rawfilename = ssdmfieldvalue.readvalue("MBES_RAW_FILENAME")

	matches = fileutils.findFiles2(True, rawfolder, "*.gsf")

	# surveyname = os.path.basename(args.inputfolder) #this folder should be the survey NAME

	#create the linestring table for the trackplot
	type, fields = geopackage_ssdm.createSurveyTracklineSSDM()
	linestringtable = geopackage.vectortable(gpkg.connection, "SurveyTrackLine", args.epsg, type, fields)

	for filename in matches:
		root = os.path.splitext(filename)[0]
		root = os.path.basename(filename)
		outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
		os.makedirs(outputfolder, exist_ok=True)
		# makedirs(outputfolder)
		outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
		if args.reprocess:
			if os.path.exists(outfilename):
				os.unlink(outfilename)
		if os.path.exists(outfilename):
			# the cache file exists so load it
			with open(outfilename) as f:
				# print("loading file %s" %(outfilename))
				lst = json.load(f)
				results.append([filename, lst])
		else:
			boundarytasks.append([filename, outfilename])

	if args.cpu == '1':
		for filename in matches:
			root = os.path.splitext(filename)[0]
			root = os.path.basename(filename)
			outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
			os.makedirs(outputfolder, exist_ok=True)
			# makedirs(outputfolder)
			outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
			result = processgsf(filename, outfilename, args.step)
			results.append([filename, result])
	else:
		multiprocesshelper.log("New GSF Files to Import: %d" %(len(boundarytasks)))		
		cpu = multiprocesshelper.getcpucount(args.cpu)
		multiprocesshelper.log("Extracting GSF Navigation with %d CPU's" %(cpu))
		pool = mp.Pool(cpu)
		multiprocesshelper.g_procprogress.setmaximum(len(boundarytasks))
		poolresults = [pool.apply_async(processgsf, (task[0], task[1], args.step), callback=multiprocesshelper.mpresult) for task in boundarytasks]
		pool.close()
		pool.join()
		for idx, result in enumerate (poolresults):
			results.append([boundarytasks[idx][0], result._value])
			# print (result._value)

	# now we can read the results files and create the geometry into the SSDM table
	multiprocesshelper.log("Files to Import to geopackage: %d" %(len(results)))		

	multiprocesshelper.g_procprogress.setmaximum(len(results))
	for result in results:
		createTrackLine(result[0], result[1], linestringtable, float(args.step), geo)
		multiprocesshelper.mpresult("")
################################################################################
def mp_processsegy(args, gpkg, geo):

	# boundary = []
	boundarytasks = []
	results = []

	rawfolder = os.path.join(args.inputfolder, ssdmfieldvalue.readvalue("MBES_RAW_FOLDER"))
	if not os.path.isdir(rawfolder):
		rawfolder = args.inputfolder

	# rawfilename = ssdmfieldvalue.readvalue("MBES_RAW_FILENAME")

	matches = fileutils.findFiles2(True, rawfolder, "*.sgy")

	# surveyname = os.path.basename(args.inputfolder) #this folder should be the survey NAME

	#create the linestring table for the trackplot
	type, fields = geopackage_ssdm.createSurveyTracklineSSDM()
	linestringtable = geopackage.vectortable(gpkg.connection, "SurveyTrackLine", args.epsg, type, fields)

	for filename in matches:
		root = os.path.splitext(filename)[0]
		root = os.path.basename(filename)
		outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
		os.makedirs(outputfolder, exist_ok=True)
		# makedirs(outputfolder)
		outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
		if args.reprocess:
			if os.path.exists(outfilename):
				os.unlink(outfilename)
		if os.path.exists(outfilename):
			# the cache file exists so load it
			with open(outfilename) as f:
				# print("loading file %s" %(outfilename))
				lst = json.load(f)
				results.append([filename, lst])
		else:
			boundarytasks.append([filename, outfilename])

	if args.cpu == '1':
		for filename in matches:
			root = os.path.splitext(filename)[0]
			root = os.path.basename(filename)
			outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
			os.makedirs(outputfolder, exist_ok=True)
			# makedirs(outputfolder)
			outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
			result = processsegy(filename, outfilename, args.step)
			results.append([filename, result])
	else:
		multiprocesshelper.log("New sgy Files to Import: %d" %(len(boundarytasks)))		
		cpu = multiprocesshelper.getcpucount(args.cpu)
		multiprocesshelper.log("Extracting SGY Navigation with %d CPU's" %(cpu))
		pool = mp.Pool(cpu)
		multiprocesshelper.g_procprogress.setmaximum(len(boundarytasks))
		poolresults = [pool.apply_async(processsegy, (task[0], task[1], args.step), callback=multiprocesshelper.mpresult) for task in boundarytasks]
		pool.close()
		pool.join()
		for idx, result in enumerate (poolresults):
			results.append([boundarytasks[idx][0], result._value])
			# print (result._value)

	# now we can read the results files and create the geometry into the SSDM table
	multiprocesshelper.log("Files to Import to geopackage: %d" %(len(results)))		

	multiprocesshelper.g_procprogress.setmaximum(len(results))
	for result in results:
		createTrackLine(result[0], result[1], linestringtable, float(args.step), geo)
		multiprocesshelper.mpresult("")

################################################################################
def mp_processjsf(args, gpkg, geo):

	# boundary = []
	boundarytasks = []
	results = []

	rawfolder = os.path.join(args.inputfolder, ssdmfieldvalue.readvalue("MBES_RAW_FOLDER"))
	if not os.path.isdir(rawfolder):
		rawfolder = args.inputfolder

	# rawfilename = ssdmfieldvalue.readvalue("MBES_RAW_FILENAME")

	matches = fileutils.findFiles2(True, rawfolder, "*.jsf")

	# surveyname = os.path.basename(args.inputfolder) #this folder should be the survey NAME

	#create the linestring table for the trackplot
	type, fields = geopackage_ssdm.createSurveyTracklineSSDM()
	linestringtable = geopackage.vectortable(gpkg.connection, "SurveyTrackLine", args.epsg, type, fields)

	for filename in matches:
		root = os.path.splitext(filename)[0]
		root = os.path.basename(filename)
		outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
		os.makedirs(outputfolder, exist_ok=True)
		# makedirs(outputfolder)
		outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
		if args.reprocess:
			if os.path.exists(outfilename):
				os.unlink(outfilename)
		if os.path.exists(outfilename):
			# the cache file exists so load it
			with open(outfilename) as f:
				# print("loading file %s" %(outfilename))
				lst = json.load(f)
				results.append([filename, lst])
		else:
			boundarytasks.append([filename, outfilename])

	multiprocesshelper.log("New jsf Files to Import: %d" %(len(boundarytasks)))		
	cpu = multiprocesshelper.getcpucount(args.cpu)
	multiprocesshelper.log("Extracting JSF Navigation with %d CPU's" %(cpu))
	pool = mp.Pool(cpu)
	multiprocesshelper.g_procprogress.setmaximum(len(boundarytasks))
	# poolresults = [pool.apply_async(processKMALL, (task[0], task[1], args.step)) for task in boundarytasks]
	poolresults = [pool.apply_async(processjsf, (task[0], task[1], args.step), callback=multiprocesshelper.mpresult) for task in boundarytasks]
	pool.close()
	pool.join()
	for idx, result in enumerate (poolresults):
		results.append([boundarytasks[idx][0], result._value])
		# print (result._value)

	# now we can read the results files and create the geometry into the SSDM table
	multiprocesshelper.log("Files to Import to geopackage: %d" %(len(results)))		

	multiprocesshelper.g_procprogress.setmaximum(len(results))
	for result in results:
		createTrackLine(result[0], result[1], linestringtable, float(args.step), geo)
		multiprocesshelper.mpresult("")

################################################################################
def mp_processKMALL(args, gpkg, geo):

	# boundary = []
	boundarytasks = []
	results = []

	rawfolder = os.path.join(args.inputfolder, ssdmfieldvalue.readvalue("MBES_RAW_FOLDER"))
	if not os.path.isdir(rawfolder):
		rawfolder = args.inputfolder

	# rawfilename = ssdmfieldvalue.readvalue("MBES_RAW_FILENAME")

	matches = fileutils.findFiles2(True, rawfolder, "*.kmall")

	# surveyname = os.path.basename(args.inputfolder) #this folder should be the survey NAME

	#create the linestring table for the trackplot
	type, fields = geopackage_ssdm.createSurveyTracklineSSDM()
	linestringtable = geopackage.vectortable(gpkg.connection, "SurveyTrackLine", args.epsg, type, fields)

	for filename in matches:
		root = os.path.splitext(filename)[0]
		root = os.path.basename(filename)
		outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
		os.makedirs(outputfolder, exist_ok=True)
		# makedirs(outputfolder)
		outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
		if args.reprocess:
			if os.path.exists(outfilename):
				os.unlink(outfilename)
		if os.path.exists(outfilename):
			# the cache file exists so load it
			with open(outfilename) as f:
				# print("loading file %s" %(outfilename))
				lst = json.load(f)
				results.append([filename, lst])
		else:
			boundarytasks.append([filename, outfilename])

	if args.cpu == '1':
		for filename in matches:
			root = os.path.splitext(filename)[0]
			root = os.path.basename(filename)
			outputfolder = os.path.join(os.path.dirname(args.outputFilename), "log")
			os.makedirs(outputfolder, exist_ok=True)
			outfilename = os.path.join(outputfolder, root+"_navigation.txt").replace('\\','/')
			result = processKMALL(filename, outfilename, args.step)
			results.append([filename, result])			
	else:
		multiprocesshelper.log("New kmall Files to Import: %d" %(len(boundarytasks)))		
		cpu = multiprocesshelper.getcpucount(args.cpu)
		multiprocesshelper.log("Extracting KMALL Navigation with %d CPU's" %(cpu))
		pool = mp.Pool(cpu)
		multiprocesshelper.g_procprogress.setmaximum(len(boundarytasks))
		# poolresults = [pool.apply_async(processKMALL, (task[0], task[1], args.step)) for task in boundarytasks]
		poolresults = [pool.apply_async(processKMALL, (task[0], task[1], args.step), callback=multiprocesshelper.mpresult) for task in boundarytasks]
		pool.close()
		pool.join()
		for idx, result in enumerate (poolresults):
			results.append([boundarytasks[idx][0], result._value])
			# print (result._value)

	# now we can read the results files and create the geometry into the SSDM table
	multiprocesshelper.log("Files to Import to geopackage: %d" %(len(results)))		

	multiprocesshelper.g_procprogress.setmaximum(len(results))
	for result in results:
		createTrackLine(result[0], result[1], linestringtable, float(args.step), geo)
		multiprocesshelper.mpresult("")

###############################################################################
def createTrackLine(filename, navigation, linestringtable, step, geo, surveyname=""):
	#verified April 2021
	lastTimeStamp = 0
	linestring = []

	timeIDX				= 0
	longitudeIDX		= 1
	latitudeIDX			= 2
	depthIDX 			= 3
	headingIDX 			= 4
	rollIDX 			= 5
	pitchIDX 			= 6

	# navigation = reader.loadnavigation(step)
	totalDistanceRun = 0

	if navigation is None: #trap out empty files.
		return
	print(filename)
	try:
		if len(navigation) == 0: #trap out empty files.
			print("file is empty: %s" % (filename))
			return
	except:
		return
	prevX =  navigation[0][longitudeIDX]
	prevY = navigation[0][latitudeIDX]

	for update in navigation:
		distance = geodetic.est_dist(update[latitudeIDX], update[longitudeIDX], prevY, prevX)
		totalDistanceRun += distance
		prevX = update[longitudeIDX]
		prevY = update[latitudeIDX]

	# compute the brg1 line heading
	# distance, brg1, brg2 = geodetic.calculateRangeBearingFromGeographicals(navigation[0][1], navigation[0][2], navigation[-1][1], navigation[-1][2])
	# create the trackline shape file
	for update in navigation:
		if update[0] - lastTimeStamp >= step:
			x,y = geo.convertToGrid(update[longitudeIDX],update[latitudeIDX])
			linestring.append(x)
			linestring.append(y)
			lastTimeStamp = update[0]
	# now add the very last update
	x,y = geo.convertToGrid(float(navigation[-1][longitudeIDX]),float(navigation[-1][latitudeIDX]))
	linestring.append(x)
	linestring.append(y)
	# print("Points added to track: %d" % (len(line)))
	# now add to the table.
	recDate = from_timestamp(navigation[0][timeIDX]).strftime("%Y%m%d")
	
	###########################
	# write out the FIELDS data
	###########################

	# write out the FIELDS data
	fielddata = []
	fielddata += setssdmarchivefields() # 2 fields
	fielddata += setssdmobjectfields() # 4 fields

	fielddata.append(ssdmfieldvalue.readvalue("LINE_ID"))
	#LINE_NAME
	fielddata.append(os.path.basename(filename))
	#LAST_SEIS_PT_ID
	fielddata.append(int(navigation[-1][timeIDX]))
	#SYMBOLOGY_CODE
	fielddata.append(ssdmfieldvalue.readvalue("TRACK_SYMBOLOGY_CODE"))
	#DATA_SOURCE
	fielddata.append(os.path.basename(filename))
	#CONTRACTOR_NAME
	fielddata.append(ssdmfieldvalue.readvalue("CONTRACTOR_NAME"))
	#LINE_LENGTH
	fielddata.append(totalDistanceRun)
	#FIRST_SEIS_PT_ID
	fielddata.append(int(navigation[0][timeIDX]))
	#HIRES_SEISMIC_EQL_URL
	fielddata.append(ssdmfieldvalue.readvalue("HIRES_SEISMIC_EQL_URL"))
	#OTHER_DATA_URL
	fielddata.append(ssdmfieldvalue.readvalue("OTHER_DATA_URL"))
	#HIRES_SEISMIC_RAP_URL
	fielddata.append(ssdmfieldvalue.readvalue("HIRES_SEISMIC_RAP_URL"))
	#LAYER
	fielddata.append(ssdmfieldvalue.readvalue("TRACK_LAYER"))
	#SHAPE_Length
	fielddata.append(totalDistanceRun)

	# now write the point to the table.
	linestringtable.addlinestringrecord(linestring, fielddata)		
	linestringtable.close()

###############################################################################
def setssdmarchivefields():
	fields = []
	fields.append(datetime.now().date())
	fields.append(os.getenv('username'))
	return fields

###############################################################################
def setssdmobjectfields():
	# featureid 		= 0

	# featureid 		= uuid.UUID()
	objectid 		= None
	featureid 		= str(uuid.uuid4())
	surveyid 		= ssdmfieldvalue.readvalue('survey_id')
	surveyidref 	= ssdmfieldvalue.readvalue('survey_id_ref')
	remarks 		= ssdmfieldvalue.readvalue('remarks')
	
	fields = []
	fields.append(objectid)
	fields.append(featureid)
	fields.append(surveyid)
	fields.append(surveyidref)
	fields.append(remarks)
	return fields
	
###############################################################################
def createsurveyplan(reader, table, geo):
	'''read a survey.mp file and create a SSDM geopackage'''
	totalDistanceRun = 0

	for surveyLine in reader.surveylines:
		line = []
		distance = 0
		linename = surveyLine.name[:254]
		# x,y = geo.convertToGrid(float(wpt.longitude),float(wpt.latitude))
		line.append(surveyLine.x1)
		line.append(surveyLine.y1)
		line.append(surveyLine.x2)
		line.append(surveyLine.y2)

		distance, alpha1Tp2, alpha21 = geodetic.calculateRangeBearingFromGeographicals(surveyLine.x1, surveyLine.y1,  surveyLine.x2,  surveyLine.y2 )
		heading = alpha1Tp2
		# distance += geodetic.est_dist(wpt.latitude, wpt.longitude, prevY, prevX)
		# prevX = wpt.longitude
		# prevY = wpt.latitude
			
		surveyname 		= ssdmfieldvalue.readvalue("SURVEY_NAME")
		lineprefix 		= linename
		# linename 		= surveyname[:20]
		# heading 		= 0

		userName = os.getenv('username')
		# filename = os.path.basename(reader.fileName)
		preparedDate = datetime.now()

		symbologycode 	= "TBA"
		projectname		= surveyname
		surveyblockname	= surveyname

		# write out the FIELDS data
		fielddata = []
		fielddata += setssdmarchivefields()
		fielddata += setssdmobjectfields()

		fielddata.append(surveyname)
		fielddata.append(lineprefix)
		fielddata.append(linename)		
		fielddata.append(heading)
		fielddata.append(symbologycode)

		fielddata.append(projectname)
		fielddata.append(surveyblockname)
		fielddata.append(userName)
		fielddata.append(preparedDate)
		fielddata.append(userName)

		fielddata.append(preparedDate)
		fielddata.append("")
		fielddata.append(distance)

		# now write the point to the table.
		table.addlinestringrecord(line, fielddata)		
	
	table.close()

def from_timestamp(unixtime):
	return datetime(1970, 1 ,1) + timedelta(seconds=unixtime)

###############################################################################
def to_timestamp(recordDate):
	return (recordDate - datetime(1970, 1, 1)).total_seconds()

###############################################################################
def	makedirs(odir):
	if not os.path.isdir(odir):
		os.makedirs(odir, exist_ok=True)
	odirlog = os.path.join(odir, "log").replace('\\','/')
	if not os.path.isdir(odirlog):
		os.makedirs(odirlog)
	return odirlog

###############################################################################
if __name__ == "__main__":
	main()
