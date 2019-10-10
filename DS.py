# library
import boto3
import requests
import pandas as pd
from io import StringIO
import numpy as np
from scipy import signal
import peakutils
from scipy import stats
import os
from matplotlib import colors
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import io
from io import BytesIO
sio = BytesIO()
import datetime
import json
import multiprocessing as mp
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
# Modules
import config
from config import *
import reader
import time



s3 = boto3.resource('s3')
my_bucket = s3.Bucket(bucket)
"""
SIDE FUNCTIONS
"""
def convert_image_bytes(filename):
    img = Image.open(filename, mode='r')
    roiImg = img.crop()
    imgByteArr = io.BytesIO()
    roiImg.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_mass(timestamp,mass_list):
    mass_loc	= mass_list.index.get_loc(timestamp,method='ffill') #ffil to find the PREVIOUS index value if no exact match
    time_diff	= mass_list.index[mass_loc] - timestamp

    return [mass_list['mass'].iloc[mass_loc],time_diff]



def read_mass_file(client, client_info):
	filename = "{}{}.csv".format(client_info['mass_folder'],'mass')
	data_location = 's3://{}/{}'.format(bucket, filename)
	mass_df = pd.read_csv(data_location,sep=client_info['delimiter'],parse_dates=client_info['date_columns'],dayfirst=True) 
	mass_df = mass_df[['Date Entrée','Poids Entrée']]
	mass_df.columns = ['time','mass']
	mass_df['timestamp'] = pd.to_numeric(mass_df['time'].dt.strftime('%s'))
	mass_df=mass_df.set_index('timestamp',verify_integrity =False)[['mass']]
	mass_df=mass_df.sort_index()
	mass_df = mass_df[~mass_df.index.duplicated(keep='first')]
	#mass_df = pd.read_csv(filename,sep=";",parse_dates=[5,6],dayfirst=True,engine='python')

	return mass_df


def feature_calculation(json_object, base_url, input_parameters):
	X_train = pd.DataFrame([])
	for index, item in enumerate(json_object):
		passage_id	= item['passage']
		df		= pd.read_json(passage_data.content,orient='split')
		df		= df.rolling(rolling_size,min_periods=1).sum()
		df		= df.iloc[rolling_size:]
		timestamp	= int(passage_id)
		for channel in range(1,input_parameters['num_channels']):
			channel_name = 'channel_{}'.format(channel)
			if channel < min(input_parameters['temp_channels']):                
				df[channel_name]	= df[channel_name] - df[channel_name].iloc[1]
			if channel == min(input_parameters['temp_channels']):
				temperature	= df[channel_name].mean()
		axletree_list	= calculate_number_of_axletree(df)
		nb_axletree	= len(axletree_list)
		if nb_axletree > min_axletree and nb_axletree < max_axletree:
			mass_info	= get_mass(timestamp+time_lag,mass_df)
			mass		= mass_info[0]
			time_diff	= mass_info[1]
			features0	= eval('features_catalog.'+input_parameters['feature_type']\
					      +'(df,input_parameters,input_data=[axletree_list])')
			features1 	= pd.DataFrame(data=features0,columns=[timestamp])
			features	= features1.transpose()
			X_train 	= pd.concat([X_train,features])

	return X_train


def correlation(df,input_parameters):
	DataPassages = pd.concat(df, ignore_index=True)
	data = DataPassages.data
	frames = []
	for s in data:
		if type(s) == list:
			data = pd.DataFrame(s)

		frames.append(data)
    
	PassageSignalConcat = pd.concat(frames, ignore_index=True)
	PassageSignalConcat = PassageSignalConcat.rename(columns = lambda x : "channel_"+ str(x+1))
	PassageSignalConcat = PassageSignalConcat * volt_strain
	temperature = (PassageSignalConcat["channel_"+str(input_parameters['temp_channels'][0])] / 400).apply(lambda x : x * A + B)
	PassageSignalConcat = PassageSignalConcat.dropna()
	cor0 = []
	for j in range(input_parameters['num_channels']-len(input_parameters['temp_channels'])):##TODO change
		slope, intercept, r_value, p_value, std_err = stats.linregress(temperature,PassageSignalConcat.iloc[:,j])
		cor0.append(r_value)
	X_train = pd.DataFrame([])
	cor1 	= pd.DataFrame(data=cor0)#timesamp date debut
	cor	= cor1.transpose()

	return cor


def discrete_cmap(base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(3)
    return base.from_list(cmap_name, color_list, 3)



'''
General inputs
'''
input_parameters = reader.read_input()

len_line =(input_parameters['num_channels'] - len(input_parameters['temp_channels'])) //4 
# Mat parameters 1
input_parameters['start_first_line_1']	= 1
input_parameters['end_first_line_1']	= len_line 
input_parameters['start_second_line_1']	= len_line + 1
input_parameters['end_second_line_1']	= 2* len_line 

input_parameters['start_first_line_2']	= 2* len_line +1	
input_parameters['end_first_line_2']	= 3* len_line 	
input_parameters['start_second_line_2']	= 3* len_line +1	
input_parameters['end_second_line_2']	= 4* len_line 	

print(input_parameters['start_first_line_1'],input_parameters['end_first_line_1'],input_parameters['start_second_line_1'],input_parameters['end_second_line_1'])

print(input_parameters['start_first_line_2'],input_parameters['end_first_line_2'],input_parameters['start_second_line_2'],input_parameters['end_second_line_2'])

# create passage ids dataframe
input_parameters['ec2_server']=config.env_to_ec2(input_parameters['environment'])

# Hours formatting
HH_start, MM_start =[input_parameters['Time_start'].split('h')[i] for i in range(len(input_parameters['Time_start'].split('h')))] 
HH_end, MM_end =[input_parameters['Time_end'].split('h')[i] for i in range(len(input_parameters['Time_end'].split('h')))]

day_list=np.arange(input_parameters['start_day'].astype('int'),input_parameters['end_day'].astype('int'))
base_url = 'http://'+input_parameters['ec2_server']+'/api/v1/passage/'+input_parameters['gateway_id']+'-'+input_parameters['labjack_id']+'-'+input_parameters['mat_id']
print(day_list)

# read mass file
mode='continuous'
s3 = boto3.resource('s3')
my_bucket_source	= s3.Bucket(bucket)

#correlation
X_cor = pd.DataFrame([])
date=[]
prefix_source=folder_to_save_results_cor+input_parameters['mat_id']+'/'
for obj in my_bucket_source.objects.filter(Prefix=prefix_source):
	data_location = 's3://{}/{}'.format(obj.bucket_name, obj.key)
	if obj.key.endswith('_T'+input_parameters['Time_start']+'_T'+input_parameters['Time_end']+'.csv'):
		print(obj.key)
		#print(obj.key[72:74])
		if obj.key[72:74] in str(day_list):
			print(obj.key[72:74])
			l=obj.key.split('_')
			date.append(l[5]+'/'+l[4]+'/'+l[3])
			df= pd.read_csv(data_location,sep=',',dayfirst=True)
			X_cor = pd.concat([X_cor,df])
X_cor = X_cor.dropna()
X_cor = X_cor.iloc[:,1:]

#slope
X_slope = pd.DataFrame([])
date=[]
prefix_source=folder_to_save_results_slope+input_parameters['mat_id']+'/'
for obj in my_bucket_source.objects.filter(Prefix=prefix_source):
	data_location = 's3://{}/{}'.format(obj.bucket_name, obj.key)
	if obj.key.endswith('_T'+input_parameters['Time_start']+'_T'+input_parameters['Time_end']+'.csv'):
		print(obj.key)
		if obj.key[60:62] in str(day_list):
			print(obj.key[60:62])
			l=obj.key.split('_')
			date.append(l[5]+'/'+l[4]+'/'+l[3])
			df= pd.read_csv(data_location,sep=',',dayfirst=True)
			X_slope = pd.concat([X_slope,abs(df)])
X_slope = X_slope.dropna()
X_slope = X_slope.iloc[:,1:]

#amplitude
'''X_amp = pd.DataFrame([])
date=[]
prefix_source=folder_to_save_results_amp+input_parameters['mat_id']+'/'
for obj in my_bucket_source.objects.filter(Prefix=prefix_source):
	data_location = 's3://{}/{}'.format(obj.bucket_name, obj.key)
	if obj.key.endswith('_T'+input_parameters['Time_start']+'_T'+input_parameters['Time_end']+'.csv'):
		print(obj.key)
		if obj.key[68:70] in str(day_list):
			print(obj.key[68:70])
			l=obj.key.split('_')
			date.append(l[5]+'/'+l[4]+'/'+l[3])
			print(date)
			df= pd.read_csv(data_location,sep=',',dayfirst=True)

			print('shape',df2)
			X_amp = pd.concat([X_amp,abs(df.iloc[:,0])])
			print('xamp',X_amp)
X_amp = X_amp.dropna()
X_amp = X_amp.iloc[:,1:]'''

X_amp = pd.DataFrame([])
date=[]
prefix_source=folder_to_save_results_amp+input_parameters['mat_id']+'/'
for obj in my_bucket_source.objects.filter(Prefix=prefix_source):
	data_location = 's3://{}/{}'.format(obj.bucket_name, obj.key)
	if obj.key.endswith('_T'+input_parameters['Time_start']+'_T'+input_parameters['Time_end']+'.csv'):
		print(obj.key)
		if obj.key[68:70] in str(day_list):
			print(obj.key[68:70])
			l=obj.key.split('_')
			date.append(l[5]+'/'+l[4]+'/'+l[3])
			#print(date)
			df= pd.read_csv(data_location,sep=',',dayfirst=True)
			#print(df.shape)
			#print('df', df)
			#print(df.columns[0])
			#data = []
			line = pd.DataFrame([df.columns])
			line.columns = df.columns
			line.iloc[:,1] = float(line.iloc[:,1])
			# always inserting new rows at the first position - last row will be always on top    
			df = pd.concat([line, df]).reset_index(drop=True)
			df.columns=['0','1']
			#print(df)
			#df = pd.concat([pd.DataFrame(data), df[:]]).reset_index(drop = True) 
			df=df.drop(columns='0')
			#print(df.transpose().reset_index(drop=True))
			#df=df.transpose()
			df=df.transpose().reset_index(drop=True)
			print(df)
			#print('shape',df2)
			X_amp = pd.concat([X_amp,abs(df)])
			print('xamp',X_amp)
X_amp = X_amp.dropna()
X_amp = X_amp.iloc[:,1:]
			

# dist slope cor
slope_wk=pd.DataFrame([])
cor_wk = pd.DataFrame([])
slope_j5k=pd.DataFrame([])
cor_j5k = pd.DataFrame([])
channel_list = set(range(1,input_parameters['num_channels']+1))
j5k_channel = set([16,27,33,54,43,49])  
channel_list = channel_list - j5k_channel - set(input_parameters['temp_channels'])
for ch in channel_list:
	ch_name='channel_{}'.format(str(ch))
	slope_wk = pd.concat((slope_wk, X_slope[ch_name]), axis=0)
	cor_wk = pd.concat((cor_wk, X_cor[ch_name]), axis=0)
for ch in j5k_channel:
	ch_name='channel_{}'.format(str(ch))
	slope_j5k = pd.concat((slope_j5k, X_slope[ch_name]), axis=0)
	cor_j5k = pd.concat((cor_j5k, X_cor[ch_name]), axis=0)
plt.plot(slope_wk, cor_wk, 'b', marker='o',linewidth=0, label='WK sensors')
plt.plot(slope_j5k, cor_j5k, 'r', marker='o',linewidth=0, label='J5K sensors')
plt.xlabel('slope')
plt.ylabel('correlation')
plt.legend()
output_file='cor_vs_slope_'+input_parameters['mat_id']+'_'+input_parameters['start_month']+'_'+input_parameters['start_day']+'_'+input_parameters['end_month']+'_'+input_parameters['end_day']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.eps'
key = folder_to_save_results_cor + output_file
output_location = 's3://{}/{}'.format(bucket, key)
plt.savefig(output_file);plt.close()

# Plots correlation
# color map
N=input_parameters['num_channels'] - len(input_parameters['temp_channels'])
pos= np.arange(0.5, N,1.0)
posx= np.arange(0.5, len(date),1.0)
labels	= ['channel_{}'.format(str(j)) for j in range(1,input_parameters['num_channels']-len(input_parameters['temp_channels'])+1)]
diverging_colors = sns.color_palette("RdBu", 10)
sns.palplot(diverging_colors)
sns.set_palette("bright")
grid_kws = {"height_ratios": (.9, .01), "hspace": 0.1, "wspace": 0.01}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(len(date)*4,N), gridspec_kw=grid_kws)# figsize=(0.5*N,N), figsize=(8,8)
ax = sns.heatmap(X_cor.transpose(), ax=ax,vmin=0,vmax=1,cbar_ax=cbar_ax,cmap="RdBu",cbar_kws={"orientation": "horizontal"},linewidths=.3)#, annot=True)
ax.set_yticks(pos)
ax.set_yticklabels(labels,fontsize=25)
ax.set_xticks(posx)
ax.set_xticklabels(date,fontsize=23)
form		= "{:.2f}"
for (j,i), x in np.ndenumerate(X_cor.transpose()):
    if x<0.8:
    	ax.annotate(form.format(x), xy=(i+0.5,j+0.5), color='r', ha="center", va="center"),#,bbox=dict(color=None, alpha=30))
    else:
    	ax.annotate(form.format(x), xy=(i+0.5,j+0.5),color='w', ha="center", va="center"),#,bbox=dict(color=None, alpha=30))
output_file='correlation_colormap_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.eps'
key = folder_to_save_results_cor + output_file
output_location = 's3://{}/{}'.format(bucket, key)
fig.savefig(output_file);plt.close()
cmap = colors.ListedColormap(['red', 'blue'])
bounds=[0.0,0.8,1.0]
norm = colors.BoundaryNorm(bounds, cmap.N)

# correlation and slope values
diverging_colors = sns.color_palette("RdBu", 10)
sns.palplot(diverging_colors)
sns.set_palette("bright")
grid_kws = {"height_ratios": (.9, .01), "hspace": 0.1, "wspace": 0.01}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(len(date)*4,N), gridspec_kw=grid_kws)# figsize=(0.5*N,N), figsize=(8,8)
ax = sns.heatmap(X_cor.transpose(), ax=ax,vmin=0,vmax=1,cbar_ax=cbar_ax,cmap=cmap,norm=norm,cbar_kws={"ticks":[0.0, 0.8,1.0],"orientation": "horizontal"},linewidths=.3)#,annot=True)
ax.set_yticks(pos)
ax.set_yticklabels(labels,fontsize=25)
ax.set_xticks(posx)
ax.set_xticklabels(date,fontsize=23)
form		= "{:.2f}"
for (j,i), x in np.ndenumerate(X_slope.transpose()):
    if x>100 :
    	ax.annotate(form.format(x), xy=(i+0.5,j+0.5), color='r', ha="center", va="center"),#,bbox=dict(color=None, alpha=30))
    else:
    	ax.annotate(form.format(x), xy=(i+0.5,j+0.5),color='w', ha="center", va="center"),#,bbox=dict(color=None, alpha=30))
output_file='correlation_slope_colormap_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.eps'
key = folder_to_save_results_cor + output_file
output_location = 's3://{}/{}'.format(bucket, key)
fig.savefig(output_file);plt.close()



# Plot slope
N=input_parameters['num_channels'] - len(input_parameters['temp_channels'])
pos= np.arange(0.5, N,1.0)
posx= np.arange(0.5, len(date),1.0)
labels	= ['channel_{}'.format(str(j)) for j in range(1,input_parameters['num_channels']-len(input_parameters['temp_channels'])+1)]
diverging_colors = sns.color_palette("RdBu", 10)
sns.palplot(diverging_colors)
sns.set_palette("bright")
grid_kws = {"height_ratios": (.9, .01), "hspace": 0.1, "wspace": 0.01}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(len(date)*4,N), gridspec_kw=grid_kws)# figsize=(0.5*N,N), figsize=(8,8)
ax = sns.heatmap(X_slope.transpose(), ax=ax,cbar_ax=cbar_ax,cmap="RdBu_r",cbar_kws={"orientation": "horizontal"},linewidths=.3)
ax.set_yticks(pos)
ax.set_yticklabels(labels,fontsize=25)
ax.set_xticks(posx)
ax.set_xticklabels(date,fontsize=23)
form		= "{:.2f}"
for (j,i), x in np.ndenumerate(X_slope.transpose()):
    if x>100:
    	ax.annotate(form.format(x), xy=(i+0.5,j+0.5), color='r', ha="center", va="center"),#,bbox=dict(color=None, alpha=30))
    else:
    	ax.annotate(form.format(x), xy=(i+0.5,j+0.5),color='w', ha="center", va="center"),#,bbox=dict(color=None, alpha=30))
output_file='slope_colormap_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.eps'
key = folder_to_save_results_slope + output_file
output_location = 's3://{}/{}'.format(bucket, key)
fig.savefig(output_file);plt.close()

# Plot amplitude
print('xamp', X_amp)
diverging_colors = sns.color_palette("RdBu", 10)
sns.palplot(diverging_colors)
sns.set_palette("bright")
grid_kws = {"height_ratios": (.9, .01), "hspace": 0.1, "wspace": 0.01}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(len(date)*4,N), gridspec_kw=grid_kws)# figsize=(0.5*N,N), figsize=(8,8)
ax = sns.heatmap(X_amp.iloc[:,:75].transpose(),ax=ax,cbar_ax=cbar_ax,cmap="RdBu_r",cbar_kws={"orientation": "horizontal"},linewidths=.3, annot=True)
ax.set_yticks(pos)
ax.set_yticklabels(labels,fontsize=25,rotation='horizontal')
ax.set_xticks(posx)
ax.set_xticklabels(date,fontsize=23)
output_file='amplitude_colormap_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.eps'
key = folder_to_save_results_amp + output_file
output_location = 's3://{}/{}'.format(bucket, key)
fig.savefig(output_file);plt.close()


# histogramme slope variation
channel_list = set(range(1,input_parameters['num_channels']+1))
defected_channel = set([20,38, 51, 53,54, 55, 58,  59, 60])
fig,ax = plt.subplots(1, figsize=(10,10))
slope_b =[]
slope_a = []
slope_b2 =[]
slope_a2 = []
channel_list = channel_list - defected_channel- set(input_parameters['temp_channels'])
for ch in channel_list:
	ch_name='channel_{}'.format(str(ch))
	slope_b.append(X_slope[ch_name].iloc[0])
	slope_a.append(X_slope[ch_name].iloc[-1])
	slope_b2.append(X_slope[ch_name].iloc[2])
	slope_a2.append(X_slope[ch_name].iloc[10])
# the histogram of the data
fig,ax = plt.subplots(1, figsize=(10,15))
n, bins, patches = plt.hist(np.array(slope_a)-np.array(slope_b), 10)
plt.xlabel('Slope variation : 06/07 - 08/07  VINCI')
output_file='histo_slope_vinci_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.png'
plt.savefig(output_file);plt.close()

fig,ax = plt.subplots(1, figsize=(10,15))
n, bins, patches = plt.hist(np.array(slope_a2)-np.array(slope_b2), 10)
plt.xlabel('Slope variation : 22/07 - 29/07 VEOLIA')
output_file='histo_slope_veolia_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.png'
plt.savefig(output_file);plt.close()

fig,ax = plt.subplots(1, figsize=(20,15))
diff = X_slope.iloc[0,:] -X_slope.iloc[10,:]
print(diff)
sns.barplot(x = diff.index.tolist(), y = diff.tolist() )
plt.ylabel('Slope variation per channel')
ax.set_xticklabels(np.arange(1,len(channel_list)+1),fontsize=15, rotation='vertical')
output_file='histo_slope_per_channel_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.png'

plt.savefig(output_file);plt.close()

#temperature
print('amp', X_amp)
fig,ax = plt.subplots(1, figsize=(10,15))
ax.set_xticks(posx)
ax.set_xticklabels(date,fontsize=15, rotation='vertical')
plt.plot(np.arange(len(date)),X_amp[str(input_parameters['temp_channels'][0]-1)], marker='o', markersize=1, linewidth=0.5)
plt.ylabel('Temperature °C')
plt.savefig('temperature.png')
plt.close()


channel_list = set(range(0,input_parameters['num_channels']))- set(input_parameters['temp_channels'])
temperature = np.argsort(X_amp[str(input_parameters['temp_channels'][0]-1)].tolist()) 
print('temp',temperature)
print(len(temperature))
print(X_amp[str(input_parameters['temp_channels'][0]-1)].values.shape)
print(X_amp[str(ch)].values.shape)
#print(X_amp[str(ch)].values[temperature])
print(np.sort(X_amp[str(input_parameters['temp_channels'][0]-1)].values))

channel_list=np.arange(0,76)
print('amp', X_amp)
for ch in channel_list:
	#amp=X_amp[str(ch)].values
	fig,ax = plt.subplots(1,2, figsize=(10,15))
	ax[0].set_xticks(posx)
	ax[0].set_xticklabels(date,fontsize=15, rotation='vertical')
	ax[0].plot(np.arange(len(date)),X_amp['76'], marker='o', markersize=2, linewidth=0.5, label='Temperature')#X_amp[str(input_parameters['temp_channels'][0]-1)], marker='o', markersize=2, linewidth=0.5, label='Temperature')
	ax[0].set_xlabel('Time')
	ax[0].set_ylabel('Temperature')
	ax[1].set_xticks(posx)
	ax[1].set_xticklabels(date,fontsize=15, rotation='vertical')
	ax[1].plot(np.arange(len(date)),X_amp[str(ch)], marker='o', markersize=2, linewidth=0.5, label='Amplitude')
	ax[1].set_xlabel('Time')
	ax[1].set_ylabel('Amplitude')
	#plt.ylabel('Amplitude (mu strain)')
	#plt.savefig('temperature.png')
	#plt.close()
	fig_name = "Temperature_channel_"+str(ch)
	sio = BytesIO()
	plt.savefig(sio, format="png")
	my_bucket.put_object(Key = folder_to_save_results_amp_T + fig_name + ".png", 
                     Body = convert_image_bytes(sio))
	plt.close()

'''
# discontinuities
print('discontinuities')
X_discont = pd.DataFrame([])
date=[]
print('discontinuities_{}'.format(input_parameters['mat_id']))
prefix_source=folder_to_save_results_discont+input_parameters['mat_id']+'/'
print(prefix_source)
for obj in my_bucket_source.objects.filter(Prefix=prefix_source):
	data_location = 's3://{}/{}'.format(obj.bucket_name, obj.key)
	print(data_location)
	if obj.key.endswith('_T'+input_parameters['Time_start']+'_T'+input_parameters['Time_end']+'.csv'):#obj.key.startswith('dicontinuities'#):_{}'.format(input_parameters['mat_id'])):#obj.key.endswith('_T'+input_parameters['Time_start']+'_T'+input_parameters['Time_end']+'.csv') and 
		print(obj.key)
		l=obj.key.split('_')
		date.append(l[5]+'/'+l[4]+'/'+l[3])
		print(l)
		df= pd.read_csv(data_location,sep=',',dayfirst=True)
		print(df)
		X_discont = pd.concat([X_discont,abs(df)])

#X = X_train.drop(columns="Unnamed: 0")
#print(X['channel_62'])
#print(X)
X_discont = X_discont.dropna()
X_discont = X_discont.iloc[:,1:]
N=input_parameters['num_channels'] - len(input_parameters['temp_channels'])
#print(X.iloc[1":N+1])
pos= np.arange(0.5, N,1.0)
posx= np.arange(0.5, len(date),1.0)
print(posx)
#posx= np.arange(1, 2*len(date),2.0)
labels	= ['channel_{}'.format(str(j)) for j in range(1,input_parameters['num_channels']-len(input_parameters['temp_channels'])+1)]
print(labels)


diverging_colors = sns.color_palette("RdBu", 10)
sns.palplot(diverging_colors)
sns.set_palette("bright")
grid_kws = {"height_ratios": (.9, .01), "hspace": 0.1, "wspace": 0.01}
fig, (ax, cbar_ax) = plt.subplots(2, figsize=(len(date)*4,N), gridspec_kw=grid_kws)# figsize=(0.5*N,N), figsize=(8,8)
ax = sns.heatmap(X_discont.transpose(), ax=ax,cbar_ax=cbar_ax,cmap="coolwarm",cbar_kws={"orientation": "horizontal"},linewidths=.3)

ax.set_yticks(pos)
ax.set_yticklabels(labels,fontsize=25)
#days=np.arange(input_parameters['start_day'].astype('int'),input_parameters['end_day'].astype('int'))
#date=[str(days[i])+'/'+input_parameters['month']+'/'+input_parameters['year'] for i in range(len(days))]
print(date)
ax.set_xticks(posx)
ax.set_xticklabels(date,fontsize=23)
form		= "{:.2f}"

output_file='discontinuity_color_tresholds_'+input_parameters['mat_id']+'_'+input_parameters['Time_start']+'_'+input_parameters['Time_end']+'.eps'

key = folder_to_save_results_discont + output_file
output_location = 's3://{}/{}'.format(bucket, key)
fig.savefig(output_file);plt.close()'''




print('out')



#exit()


