#class MrcData
#    def __init__(self.name):
#        self.name = name
#        self.header = dict
#        self.body = ():
import os
import struct
from PIL import Image
from pylab import *
import numpy
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt

from starReader import starRead

# return the lenth of one pixel in mrc file 
def readMrcFile(fileName):
    if not os.path.isfile(fileName):
        print("ERROR:%s is not a valid file."%(fileName))
        return None
    f = open(fileName,"rb")
    data = f.read()
    f.close()
    
    header_fmt = '10i6f3i3f2i100c3f4cifi800c' 
    header = struct.unpack(header_fmt,data[0:1024])
    print header[0],header[1],header[2],header[3]
    n_columns = header[0]
    n_rows = header[1]
    mode = header[3]
    ## 
    print 'mode:',mode
    if mode == 0:
        # signed 8-bit bytes range -128 to 127
        pass
    elif mode == 1:
        # 16-bit halfwords
        pass
    elif mode == 2:
        # 32-bit float
        body_fmt = str(n_columns*n_rows)+"f"
    elif mode == 3:
        # complex 16-bit integers
        pass
    elif mode == 4:
        # complex 32-bit reals
        pass
    elif mode == 6:
        # unsigned 16-bit range 0 to 65535
        pass
    else:
        print("ERROR:mode %s is not a valid value,should be [0|1|2|3|4|6]."%(fileName))
        return None
    
    body = struct.unpack(body_fmt,data[1024:])
    return header,body

# read input data from star format file
# input: 
#	starFileName: star file name
#	particle_size: particle size
# return:
#	positive_particle_array,negative_particle_array
#	positive_particle_array.shape: (total_particle_positive_number,particle_size,particle_size,channel) 
#	negative_particle_array.shape: (total_particle_negative_number,particle_size,particle_size,channel) 
def loadInputStarFile(starFileName,particle_size):
	#particle_size = 60
	#starFileName = "../data/particles_manual.star"
	particle_star = starRead(starFileName)
	table_star = particle_star.getByName('data_')
	mrcfilename_list = table_star.getByName('_rlnMicrographName')
	coordinateX_list = table_star.getByName('_rlnCoordinateX')
	coordinateY_list = table_star.getByName('_rlnCoordinateY')

	# creat a dictionary to store the coordinate
	# the key is the mrc file name
	# the value is a list of the coordinates
	coordinate = {}
	path_star = os.path.split(starFileName)
	for i in range(len(mrcfilename_list)):
    		fileName = mrcfilename_list[i]
    		fileName = os.path.join(path_star[0],fileName)
    		if fileName in coordinate:
        		coordinate[fileName][0].append(int(float(coordinateX_list[i])))
        		coordinate[fileName][1].append(int(float(coordinateY_list[i])))
    		else:
        		coordinate[fileName] = [[],[]]
        		coordinate[fileName][0].append(int(float(coordinateX_list[i])))
        		coordinate[fileName][1].append(int(float(coordinateY_list[i])))

	# read mrc data
	particle_array_positive = None
	particle_array_negative = None
	number_total_particle = 0
	for key in coordinate:
    		print key
    		header,body = readMrcFile(key)
    		n_col = header[0]
    		n_row = header[1]
    		body = list(body)
    		body_2d = array(body,dtype=float32).reshape(n_col,n_row)
    
		# show the micrograph with manually picked particles 	
    		# plot the circle of the particle
		# del the particle around the boundry
		#pil_im = Image.fromarray(uint8(body_2d))
    		fig = plt.figure()
    		ax = fig.add_subplot(111)
    		plt.gray()
    		plt.imshow(body_2d)
		radius = particle_size/2	
		i = 0
		while True:
			if i >= len(coordinate[key][0]):
				break

        		coordinate_x = coordinate[key][0][i]
        		coordinate_y = coordinate[key][1][i]
        		cir1 = Circle(xy = (coordinate_x,coordinate_y), radius = radius, alpha = 0.5, color='g', fill=False)
      			ax.add_patch(cir1)
			# extract the particles
			if coordinate_x < radius and coordinate_y < radius and coordinate_y+radius > n_col and coordinate_x+radius > n_row:
				del coordinate[key][0][i]	
				del coordinate[key][1][i]
			else:
				i = i + 1	
    		#plt.show()
		
		number_particle = len(coordinate[key][0])
		number_total_particle = number_total_particle + number_particle
		print 'number of particles:',number_particle

		# extract the positive particles
		# store the particles in a concated array: particle_array_positive	
		for i in range(number_particle):
        		coordinate_x = coordinate[key][0][i]
        		coordinate_y = coordinate[key][1][i]
			patch = body_2d[(coordinate_y-radius):(coordinate_y+radius),(coordinate_x-radius):(coordinate_x+radius)]
			# nomalize the patch
			max_value = patch.max()
			min_value = patch.min()
			patch = (patch-min_value)/(max_value-min_value)
			mean_value = patch.mean()
			std_value = patch.std()
			patch = (patch-mean_value)/std_value
			if particle_array_positive == None:
				particle_array_positive = patch
			else:
				particle_array_positive = concatenate((particle_array_positive,patch))
		
		# extract the negative particles
		# store the particles in a concated array: particle_array_negative	
		for i in range(number_particle):
			while True:
				isLegal = True
				coor_x = numpy.random.randint(radius, n_row-radius)
				coor_y = numpy.random.randint(radius, n_col-radius)
				for j in range(number_particle):
        				coordinate_x = coordinate[key][0][i]
        				coordinate_y = coordinate[key][1][i]
					distance = ((coor_x-coordinate_x)**2+(coor_y-coordinate_y)**2)**0.5
					if distance < 0.5*particle_size:
						isLegal = False
						break
				if isLegal:
					patch = body_2d[(coor_y-radius):(coor_y+radius),(coor_x-radius):(coor_x+radius)]
					# nomalize the patch
					max_value = patch.max()
					min_value = patch.min()
					patch = (patch-min_value)/(max_value-min_value)
					mean_value = patch.mean()
					std_value = patch.std()
					patch = (patch-mean_value)/std_value
					if particle_array_negative == None:
						particle_array_negative = patch
					else:
						particle_array_negative = concatenate((particle_array_negative,patch))
					break
								
	print 'number_total_particle:',number_total_particle	
	particle_array_positive = particle_array_positive.reshape(number_total_particle,2*radius,2*radius,1)	
	particle_array_negative = particle_array_negative.reshape(number_total_particle,2*radius,2*radius,1)	
	print particle_array_positive.shape, particle_array_positive.dtype
	print particle_array_negative.shape, particle_array_negative.dtype
   	return particle_array_positive,particle_array_negative 


# read input data from star format file
# input: 
#	starFileName: star file name
#	particle_size: particle size
# return:
#	train_particle_array,train_label_array
#	train_particle_array shape (total_particle_number,particle_size,particle_size,channel) 
#	train_label_array.shape (total_particle_number) 
def loadInputData(starFileName="../data/particles_manual.star",particle_size=60,validation_ratio=0.1):    
	particle_array_positive,particle_array_negative = loadInputStarFile(starFileName,particle_size)	
    	fig = plt.figure()
    	ax = fig.add_subplot(111)
    	plt.gray()
    	plt.imshow(particle_array_positive[0:10,...].reshape((10*60,60)))
	#plt.show()
	numpy.random.shuffle(particle_array_positive)	
	numpy.random.shuffle(particle_array_negative)	

	validation_size = int(validation_ratio*particle_array_positive.shape[0])
	train_size = particle_array_positive.shape[0] - validation_size
	print 'train_size',train_size*2
	print 'validation_size',validation_size*2
	validation_data = particle_array_positive[:validation_size, ...]
	validation_data = concatenate((validation_data,particle_array_negative[:validation_size, ...]))
	validation_labels = concatenate((zeros(validation_size,dtype=int64),ones(validation_size,dtype=int64)))

	train_data = particle_array_positive[validation_size:, ...]
	train_data = concatenate((train_data,particle_array_negative[validation_size:, ...]))
	train_labels = concatenate((zeros(train_size,dtype=int64),ones(train_size,dtype=int64)))
	#print train_data[1]
	print train_data.shape,train_data.dtype
	print train_labels.shape,train_labels.dtype
	print validation_data.shape,validation_data.dtype
	print validation_labels.shape,validation_labels.dtype
	return train_data,train_labels,validation_data,validation_labels

loadInputData()
