'''
Configuration file
'''


# Data and noise parameters
volt_strain		 = 400   # coeff to convert volt to strain 
noise_limit		 = 250
relative_threshold	 = 0.5
relative_noise_threshold = 0.3
minimum_distance	 = 200 #in ms
rolling_size		 = 20
max_axletree		 = 10
min_axletree		 = 1
window			 = 700
time_lag		 = 7200


#--------------------- temperature ----------------
A = 12.858#-51.1#12.858
B = 66.914#-247.2#66.914

# Destination
bucket = 'altaroad-machine-learning'
folder_to_save_results_cor = "sagemaker/diagnostic_sensor/correlation/"
folder_to_save_results_slope = "sagemaker/diagnostic_sensor/slope/"
folder_to_save_results_amp = "sagemaker/diagnostic_sensor/amplitude/"
folder_to_save_results_var = "sagemaker/diagnostic_sensor/std/"
folder_to_save_results_discont = "sagemaker/diagnostic_sensor/discont/major"
folder_to_save_results_discontb = "sagemaker/diagnostic_sensor/discont/minor"
folder_to_save_results_amp_T = "sagemaker/diagnostic_sensor/amplitude_temperature/"

def env_to_ec2(environment):
	if environment=='PROD':
		ec2_server = 'ec2-3-120-139-115.eu-central-1.compute.amazonaws.com:8080'
	elif environment=='DEV':
		ec2_server = 'ec2-52-59-204-91.eu-central-1.compute.amazonaws.com:8080'

	return ec2_server



def client_file(client):
	if client=='voltorbe':
		client_info = {'mass_folder' : 'sagemaker/mass/{}/'.format(client),
			       'delimiter'   : ';',
			       'date_columns': [5,6]
			       }

	return client_info


