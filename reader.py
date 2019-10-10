import sys
import numpy as np
'''
 Recursive functions - checking inputs
'''



'''
 Inputs reading
'''
def read_input():
	"""
	Read computing options : model, noise type, method..
	read inputs from input file or command line
	OUTPUT: 
		client
		ec2_server
		gateway_id
		labjack_id
		mat_id
		channel_list
		feature_type
		year
		month
		start_day
		end_day
		
	"""

	# Search for input file
	findinput = str.find(str(sys.argv[len(sys.argv)-1]), '.i')	
	if len(sys.argv) == 3 and str(sys.argv[1]) == '-i' and findinput != -1: 
		print('read input from input file')
		input = np.genfromtxt(str(sys.argv[2]),dtype=str, delimiter=None,comments='#')
	else: 
		print('read input from command line')

	if len(sys.argv) == 3 and str(sys.argv[1]) == '-i' and findinput != -1: # input file case
		client				= input[0].astype('str')
		environment			= input[1].astype('str')
		gateway_id			= input[2].astype('str')
		labjack_id			= input[3].astype('str')
		mat_id				= input[4].astype('str')
		num_channels			= input[5].astype('int')
		if input[6].astype('str')=="[]": 
			temp_channels		=[]
		else: 
			temp_channels		= list(map(int,input[6].replace('[','').replace(']','').split(',')))
		if input[7].astype('str')=="[]": 
			defected_channels	=[]
		else: 
			defected_channels	= list(map(int,input[7].replace('[','').replace(']','').split(',')))
		feature_type		= input[8].astype('str')
		year			= input[9].astype('str')
		start_month		= input[10].astype('str')
		start_day		= input[11].astype('str')
		end_month		= input[12].astype('str')
		end_day			= input[13].astype('str')
		Time_start		= input[14].astype('str')
		Time_end		= input[15].astype('str')
		CPU_pc			= input[16].astype('str')
		

		# TODO check ec2_server, gateway, labjack, mat, feature_type

	else: # command line case
		client				= input()
		environment			= input()
		gateway_id			= input()
		labjack_id			= input()
		mat_id				= input()
		num_channels			= input()
		if input().astype('str')=="[]": 
			temp_channels		=[]
		else: 
			temp_channels		= list(map(int,input().replace('[','').replace(']','').split(',')))
		if input().astype('str')=="[]": 
			defected_channels	=[]
		else: 
			defected_channels	= list(map(int,input().replace('[','').replace(']','').split(',')))
		feature_type			= input()
		year				= input()
		start_month			= input()
		start_day			= input()
		end_month			= input()
		end_day				= input()
		Time_start			= input()
		Time_end			= input()
		CPU_pc				= input(),
		# TODO check


	input_parameters= {'client': client, 'environment': environment, 'gateway_id': gateway_id, 'labjack_id': labjack_id, 'mat_id': mat_id, 				   'num_channels': num_channels, 'temp_channels': temp_channels, 'defected_channels': defected_channels, 
			   'feature_type': feature_type, 'year': year, 'start_month': start_month, 'start_day': start_day,
                           'end_month': end_month,'end_day': end_day,
			   'Time_start': Time_start, 'Time_end': Time_end, 'CPU_pc': CPU_pc
			  } 

	return input_parameters



