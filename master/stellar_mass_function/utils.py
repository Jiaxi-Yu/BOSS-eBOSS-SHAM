import logging
import os
import time
import json
import scipy
import numpy
from scipy import constants,stats,spatial,interpolate
import matplotlib

###############################################################################
# Physical quantities
###############################################################################	

def depth_to_flux(x):
	return 5./scipy.sqrt(x)

def flux_to_depth(x):
	return (5./x)**2

def flux_to_mag(x):
	return -2.5*(scipy.log10(x)-9.)

def mag_to_flux(x):
	return 10.**(x/(-2.5)+9.)

def depth_to_mag(x):
	return flux_to_mag(depth_to_flux(x))

def mag_to_depth(x):
	return flux_to_depth(mag_to_flux(x))
	
###############################################################################
# General algorithms
###############################################################################

def extrapolate_linear(xnew,x,y):
	"""Basic linear rule."""
	return (y[1]-y[0])/(x[1]-x[0])*(xnew-x[0]) + y[0]

def interpolate_linear(x,y):
	"""Return a function performing the linear interpolation (x,y)."""
	fun = interpolate.interp1d(x,y,kind='linear',assumed_sorted=True)
	
	def call(xnew,extrapolate='linear'):
		toret = fun(xnew)
		mask_min,mask_max = xnew<x[0],xnew>x[-1]
		if extrapolate == 'linear':
			toret[mask_min] = extrapolate_linear(xnew[mask_min],x[:2],y[:2])
			toret[mask_max] = extrapolate_linear(xnew[mask_max],x[-2:],y[-2:])
		elif extrapolate == 'nan':
			toret[mask_min] = scipy.nan
			toret[mask_max] = scipy.nan
		else:
			toret[mask_min] = y[0]
			toret[mask_max] = y[-1]
		return toret
	
	return call

def interpolate_spline(x,y):
	"""Return a function performing the spline (akima1d) interpolation (x,y)."""
	fun = interpolate.Akima1DInterpolator(x,y)
	
	def call(xnew,extrapolate='spline'):
		if extrapolate == 'spline':
			return fun(xnew,extrapolate=True)
		else:
			toret = fun(xnew,extrapolate=False)
			mask_min,mask_max = xnew<x[0],xnew>x[-1]
			if extrapolate == 'linear':
				toret[mask_min] = extrapolate_linear(xnew[mask_min],x[:2],y[:2])
				toret[mask_max] = extrapolate_linear(xnew[mask_max],x[-2:],y[-2:])
			elif extrapolate == 'nan':
				toret[mask_min] = scipy.nan
				toret[mask_max] = scipy.nan
			else:
				toret[mask_min] = y[0]
				toret[mask_max] = y[-1]
			return toret

	return call

def interpolate_bin(edges,y):
	"""Return a function perfomring the binning interpolation (edges,y)."""
	def call(xnew,extrapolate='lim'):
		ibin = scipy.digitize(xnew,edges,right=False)-1
		nbins = len(edges)-1
		mask_min,mask_max = ibin<0,ibin>=nbins
		ibin[mask_min] = 0
		ibin[mask_max] = nbins-1
		toret = y[ibin]
		if extrapolate == 'nan':
			toret[mask_min] = scipy.nan
			toret[mask_max] = scipy.nan
		return toret
	
	return call

def calc_density(z,weight=None,edges=None,area=None,cosmo=None,volume=None,extrapolate=None):
	"""Calculate density in redshift bins.

	Parameters
	----------
	z : array
		redshifts.
	weight : array, optional
		weights.
	edges : array
		redshift bins to use.
	area : float, optional
		area in steradians.
	cosmo : cosmology, optional
		must have a 'comoving_distance' attribute.
	volume : array, optional
		to use instead of area.
	extrapolate : str, optional
		whether to extrapolate (zmean,nz) at bins[0],bins[-1].
		If 'linear', a linear extrapolation (limited by 0) is performed.
		Else, the nearest binned value is taken.

	Returns
	-------
	zmean : array
		the mean redshift
	nz : array
		the corresponding density

	"""
	if weight is None: weight = scipy.ones_like(z)
	sumweight = stats.binned_statistic(z,weight,statistic='sum',bins=edges)[0]
	zmean = stats.binned_statistic(z,z*weight,statistic='sum',bins=edges)[0]/sumweight
	isnan = scipy.isnan(zmean)
	zmean[isnan] = ((edges[1:]+edges[:-1])/2.)[isnan] #empty bins
	if isnan.any(): logger.warning('Some redshift bins are empty.')
	if volume is not None:
		nz = sumweight/volume
	else:
		edges_dist = cosmo.comoving_distance(edges).value*cosmo.h
		nz = sumweight/area*3./(edges_dist[1:]**3-edges_dist[:-1]**3)
	if extrapolate:
		zmin,zmax = edges[:1],edges[-1:]
		if extrapolate == 'linear':
			nz_start,nz_end = extrapolate_linear(zmin[0],zmean[:2],nz[:2]),extrapolate_linear(zmax[0],zmean[-2:],nz[-2:])
			nz_start,nz_end = max(nz_start,0.),max(nz_end,0.)
			nz = scipy.concatenate([[nz_start],nz,[nz_end]])
		else:
			nz = scipy.concatenate([[nz[0]],nz,[nz[-1]]])
		zmean = scipy.concatenate([zmin,zmean,zmax])

	return zmean,nz

def cartesian_to_sky(position,wrap=True,degree=True):
	"""Transform cartesian coordinates into distance, RA, Dec.

	Parameters
	----------
	position : array of shape (N,3)
		position in cartesian coordinates.
	wrap : bool, optional
		whether to wrap ra into [0,2*pi]
	degree : bool, optional
		whether RA, Dec are in degree (True) or radian (False).

	Returns
	-------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.

	"""
	dist = scipy.sqrt(scipy.sum(position**2,axis=-1))
	ra = scipy.arctan2(position[:,1],position[:,0])
	if wrap: ra %= 2.*constants.pi
	dec = scipy.arcsin(position[:,2]/dist)
	if degree: return dist,ra/constants.degree,dec/constants.degree
	return dist,ra,dec

def sky_to_cartesian(dist,ra,dec,degree=True,dtype=None):
	"""Transform distance, RA, Dec into cartesian coordinates.

	Parameters
	----------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.
	degree : bool
		whether RA, Dec are in degree (True) or radian (False).
	dtype : dtype, optional
		return array dtype.

	Returns
	-------
	position : array
		position in cartesian coordinates; of shape (len(dist),3).

	"""
	conversion = 1.
	if degree: conversion = constants.degree
	position = [None]*3
	cos_dec = scipy.cos(dec*conversion)
	position[0] = dist*cos_dec*scipy.cos(ra*conversion)
	position[1] = dist*cos_dec*scipy.sin(ra*conversion)
	position[2] = dist*scipy.sin(dec*conversion)
	return scipy.asarray(position,dtype=dtype).T

def fiber_collision_group(groups,mask_hasfiber,mask_target=None,return_imatch=False,return_counts=False):
	"""Resolve fiber collisions based on the collision group id.

	Parameters
	----------
	groups : array
		collision group ids.
	mask_hasfiber : boolean array
		targets which received a fiber.
	mask_target : boolean array, optional
		mask for targets.
	return_imatch : bool, optional
		whether to return imatch.
	return_counts : bool, optional
		whether to return the group multiplicity.

	Returns
	-------
	weight_cp : array
		the weight to correct for fiber collisions, is ntargets/nfibers within each collision group.
	imach : array
		12 if not in mask_target
		0 if the target has no fiber and there is no fiber in its collision group.
		1 if the target has received a fiber.
		3 if the target has been resolved, i.e. in a collision group with at least one fiber.
	counts : array
		the collision group multiplicity.

	"""
	if mask_target is None: mask_target = scipy.ones_like(groups,dtype=scipy.bool_)
	mask_hasfiber = mask_target & mask_hasfiber
	counts_target = digitized_statistics(groups,values=mask_target)
	counts_hasfiber = digitized_statistics(groups,values=mask_hasfiber)

	with numpy.warnings.catch_warnings():
		numpy.warnings.filterwarnings('ignore','divide by zero encountered in divide')
		weight_cp = counts_target*1./counts_hasfiber
	weight_cp[~mask_hasfiber] = 0. # weight = 0 for targets that did not receive a fiber	
	imatch = scipy.zeros_like(mask_target,dtype=scipy.int16)
	imatch[~mask_target] = 12
	imatch[mask_target & (counts_hasfiber > 0.)] = 3
	imatch[mask_hasfiber] = 1
	mask = mask_target & (imatch != 0) & (imatch != 12)

	logger.info('Resolved fibers = {:d} among all fibers = {:d}.'.format(int(weight_cp[mask_hasfiber].sum()),mask_target.sum()))

	if (return_imatch and return_counts):
		return weight_cp,imatch,scipy.rint(counts_target).astype(scipy.int16),scipy.rint(counts_hasfiber).astype(scipy.int16)
	if return_imatch:
		return weight_cp,imatch

def get_ra_dec_tree(radec):
	"""Return scipy.spatial.cKDTree for the given radec (shape (2,N))."""
	angular = sky_to_cartesian(1.,radec[0],radec[1])
	return spatial.cKDTree(angular,leafsize=16,compact_nodes=True,copy_data=False,balanced_tree=True)

def match_ra_dec(radec1,radec2=None,nn=1,return_sep2d=False,return_sep3d=False,distance_upper_bound=180.,tree2=None,degree=True,**kwargs):
	"""Match objects according to their RA/Dec, using scipy.spatial.cKDTree

	Parameters
	----------
	radec1 : array of shape (2,n1)
		RA/Dec of the sample 1.
	radec2 : array of shape (2,n2), optional
		RA/Dec of the sample 2. If not provided, tree2 must be.
	nn : int
		the number of nearest neighbors to consider.
	return_sep2d : bool, optional
		whether to return the angular separation.
	return_sep3d : bool, optional
		whether to return the 3d separation (on the unit sphere).
	distance_upper_bound : float, optional
		the maximum angular separation to consider.
	tree2 : scipy.spatial.cKDTree
		must be provided if radec2 is not.
	degree : bool, optional
		whether RA, Dec are in degree (True) or radian (False).
	kwargs : dict, optional
		other parameters to be provided to cKDTree.query

	Returns
	-------
	index2 : array of shape (n1,nn) (squeezed)
		indices in sample 2 matching sample 1. Missing neighbors are indicated with n2.
	sep2d : array of shape (n1,nn) (squeezed)
		angular seperation. Missing neighbors are indicated with scipy.inf.
	sep3d : array of shape (n1,nn) (squeezed)
		3d separation (on the unit sphere). Missing neighbors are indicated with scipy.inf.

	"""
	conversion = 1.
	if degree: conversion = constants.degree
	angular1 = sky_to_cartesian(1.,radec1[0],radec1[1],degree=degree)
	if tree2 is None: tree2 = get_ra_dec_tree(radec2)
	distance_upper_bound = 2.*scipy.sin(distance_upper_bound*conversion/2.)
	sep3d,index2 = tree2.query(angular1,k=nn,p=2,distance_upper_bound=distance_upper_bound,**kwargs) # warning: 3d distance
	sep2d = scipy.full_like(sep3d,scipy.inf)
	mask = ~scipy.isinf(sep3d)
	sep2d[mask] = 2.*scipy.arcsin(sep3d[mask]/2.)/conversion

	if (return_sep2d and return_sep3d):
		return index2,sep2d,sep3d
	if return_sep2d:
		return index2,sep2d
	if return_sep3d:
		return index2,sep3d
	return index2
	
###############################################################################
# Convenient functions
###############################################################################

def isnaninf(array):
	"""Is nan or inf."""
	return scipy.isnan(array) | scipy.isinf(array)

def isnotnaninf(array):
	"""Is not nan nor inf."""
	return ~isnaninf(array)

def digitized_statistics(indices,values=None,statistic='sum'):
	"""Return the array of same shape as indices, filled with the required statistics."""
	if not isinstance(indices,scipy.integer):
		uniques,inverse = scipy.unique(indices,return_inverse=True)
		uniques = scipy.arange(len(uniques))
		indices = uniques[inverse]
	else:
		uniques = scipy.unique(indices)
	edges = scipy.concatenate([uniques,[uniques[-1]+1]])
	if values is None: values = scipy.ones(len(indices),dtype='f8')
	statistics,_,binnumber = scipy.stats.binned_statistic(indices,values,statistic=statistic,bins=edges)
	return statistics[binnumber-1]

def digitized_interp(ind1,ind2,val2,fill):
	"""Return the array such that values of indices ind1 match val2 if ind1 in ind2, fill with fill otherwise."""
	val2 = scipy.asarray(val2)
	unique1,indices1,inverse1 = scipy.unique(ind1,return_index=True,return_inverse=True)
	unique2,indices2 = scipy.unique(ind2,return_index=True) #reduce ind2, val2 to uniqueness
	inter1,inter2 = overlap(unique1,unique2)
	tmp2 = val2[indices2]
	tmp1 = scipy.full(unique1.shape,fill_value=fill,dtype=type(fill))
	tmp1[inter1] = tmp2[inter2] #fill with val2 corresponding to matching ind1 and ind2
	return tmp1[inverse1]

def interp_digitized_statistics(new,indices,fill,values=None,statistic='sum'):
	"""Return the array of same shape as new, filled with the required statistics."""
	stats = digitized_statistics(indices,values=values,statistic=statistic)
	return digitized_interp(new,indices,stats,fill)

def overlap(a,b):
	"""Returns the indices for which a and b overlap.
	Warning: makes sense if and only if a and b elements are unique.
	Taken from https://www.followthesheep.com/?p=1366.
	"""
	a1=scipy.argsort(a)
	b1=scipy.argsort(b)
	# use searchsorted:
	sort_left_a=a[a1].searchsorted(b[b1], side='left')
	sort_right_a=a[a1].searchsorted(b[b1], side='right')
	#
	sort_left_b=b[b1].searchsorted(a[a1], side='left')
	sort_right_b=b[b1].searchsorted(a[a1], side='right')

	# # which values are in b but not in a?
	# inds_b=(sort_right_a-sort_left_a==0).nonzero()[0]
	# # which values are in b but not in a?
	# inds_a=(sort_right_b-sort_left_b==0).nonzero()[0]

	# which values of b are also in a?
	inds_b=(sort_right_a-sort_left_a > 0).nonzero()[0]
	# which values of a are also in b?
	inds_a=(sort_right_b-sort_left_b > 0).nonzero()[0]

	return a1[inds_a], b1[inds_b]

def radec_to_healpix(ra,dec,degree=True):
	"""Transform RA, Dec to healpix theta, phi.

	Parameters
	----------
	ra : array
		RA.
	dec : array
		Dec.
	degree : bool, optional
		whether RA, Dec are in degree (True) or radian (False).

	Returns
	-------
	theta : array
		theta.
	phi : array
		phi.

	"""
	conversion = 1.
	if degree: conversion = constants.degree
	theta_rad = dec*conversion
	phi_rad = ra*conversion
	theta_rad = constants.pi/2. - theta_rad
	phi_rad %= 2.*constants.pi

	return theta_rad,phi_rad

def healpix_to_radec(theta,phi,degree=True):
	"""Transform healpix theta, phi to RA, Dec.

	Parameters
	----------
	theta : array
		theta.
	phi : array
		phi.
	degree : bool, optional
		whether RA, Dec are in degree (True) or radian (False).

	Returns
	-------
	ra : array
		RA.
	dec : array
		Dec.

	"""
	dec = constants.pi/2. - theta
	ra = phi.copy()
	ra %= 2.*constants.pi
	if degree:
		dec /= constants.degree
		ra /= constants.degree

	return ra,dec
	
def list_to_string(list_,sep=''):
	"""Concatenate list of strings, inserting sep, or just return list_ if already a string."""
	if isinstance(list_,str):
		return list_
	return sep.join(map(str,list_))

def dict_to_string(keys,shorts,values,mainsep='_',auxsep='-'):
	li = []
	for key,short in zip(keys,shorts):
		if (key in values):
			li += [short + auxsep + list_to_string(values[key],sep=auxsep) if short else list_to_string(values[key],sep=auxsep)]
	li = list_to_string(li,sep=mainsep)
	return li

###############################################################################
# IOs
###############################################################################

def field_to_legend(field):

	txt = field.lower()
	if 'lognh1' in txt:
		return '$\\log{\\mathrm{HI}}$'
	if 'debv' in txt:
		return 'Lenz+17 - SFD $\\Delta$E(B-V) [$\\mathrm{mag}$]'
	if 'ebv' in txt:
		return 'galactic extinction E(B-V) [$\\mathrm{mag}$]'
	if 'stardens' in txt:
		return 'stellar density [$\\mathrm{deg}^{-2}$]'
	if 'decalsdens' in txt:
		return 'DECaLS objects density [$\\mathrm{deg}^{-2}$]'
	if 'psfsize' in txt:
		return txt[-1] + '-band PSF FWHM [$\\mathrm{arscec}$]'
	if 'psfdepth' in txt:
		return txt[-1] + '-band 5sig. psfdepth [$\\mathrm{mag}$]'
	if 'galdepth' in txt:
		return txt[-1] + '-band 5sig. galdepth [$\\mathrm{mag}$]'
	if 'nobs' in txt:
		return 'number of observations in {}-band'.format(txt[-1])
	if 'ra' in txt:
		return 'right ascension [$\\mathrm{deg}$]'
	if 'dec' in txt:
		return 'declination [$\\mathrm{deg}$]'
	if 'skyflux' in txt:
		return 'Sky flux'
	if 'airmass' in txt:
		return 'Air mass'
	return field

def array_to_txt(data,columns,rows,tab=20,width=60,hline='-',fmt=None):
	"""Convert array to txt."""
	numcolumns = len(columns)
	numrows = len(rows)
	output = ''
	for label in columns: output += '{:<{tab}}'.format(label,tab=tab)
	output += '\n'
	if hline is not None: output += hline*width + '\n'
	#Write data lines
	for i in range(numrows):
		if fmt is not None: strrows = [format(val,fmt) for val in data[i]]
		else: strrows = [format(val) for val in data[i]]
		output += '{:<{tab}}'.format(rows[i],tab=tab)
		for val in strrows: output +=  '{:<{tab}}'.format(val,tab=tab)
		output += '\n'
		if hline is not None: output += hline*width + '\n'
	return output

def array_to_latex(data,columns,rows,alignment='c',fmt=None):
	"""Convert array to latex."""
	numcolumns = len(columns)
	numrows = len(rows)
	output = ''
	fmtcolumns = '{}|{}'.format(alignment,alignment*numcolumns)
	#Write header
	output += '\\begin{{tabular}}{{{}}}\n'.format(fmtcolumns)
	labelcolumns = ['{}'.format(label) for label in columns]
	output += '{}\\\\\n\\hline\n'.format(' & '.join(labelcolumns))
	#Write data lines
	for i in range(numrows):
		if fmt is not None: strrows = [format(val,fmt) for val in data[i]]
		else: strrows = [format(val) for val in data[i]]
		output += '{} & {}\\\\\n'.format(rows[i], ' & '.join(strrows))
	#Write footer
	output += '\\end{tabular}\n'
	return output

def save_data_statistics(stats,list_survey,list_stats,path=None,tab=25,width=None,fmt='txt'):
	"""Save stats file to path.

	Parameters
	----------
	stats : dict
		each key must be a survey, and each value its statistics.
	list_survey : list, optional
		list of surveys for which to print stats. If not provided, take stats.keys().
	list_stats : list, optional
		list of stats. If not provided, take stats.values()[0].keys().
	path : str
		path to the stats file. If not provided, returns the char array.
	tab : int, optional
		seperation between columns.
	width : int, optional
		total with. If not provided, take tab*(len(list_survey)+1).
	fmt : str, list, optional
		in ['txt','tex']: export format.

	"""
	if list_survey is None:
		list_survey = stats.keys()
	if list_stats is None:
		list_stats = stats.values()[0].keys()
	if width is None: width = tab*(len(list_survey)+1)
	data = []; rows = []
	for key in list_stats:
		if key.startswith('N'):
			N = [int(scipy.rint(stats[survey][key])) for survey in list_survey]
			if fmt == 'tex':
				data.append(['${:,d}$'.format(n) for n in N])
				rows.append('$N_{{\mathrm{{{}}}}}$'.format(key[1:]))
			else:
				data.append(['{:,d}'.format(n) for n in N])
				rows.append(key)
		else:
			N = [stats[survey][key] for survey in list_survey]
			if fmt == 'tex':
				data.append(['${:,.4f}$'.format(n) for n in N])
				rows.append(key.replace('deg^2','$\deg^{2}$').replace('deg^-2','$\deg^{-2}$'))
			else:
				data.append(['{:,.4f}'.format(n) for n in N])
				rows.append(key)
	columns = [''] + list_survey
	
	if fmt == 'tex':
		output = array_to_latex(data,columns,rows,alignment='l',fmt=None)
	else:
		output = log_header('Sample statistics',width=width,beg='') + '\n'
		output += array_to_txt(data,columns,rows,tab=tab,width=width,hline='-',fmt=None)
	
	if path is not None:
		with open(path,'w') as file:
			file.write(output)
			file.close()
		logger.info('Saving statistics to {}.'.format(path))
	else:
		return output

def save_density(density,list_survey=[],path='density.txt',tab=20):
	"""Save density file to path.

	Parameters
	----------
	density : dict
		each key must be a survey, and each value a dictionary with 'Z', 'NZ', and 'area'.
	list_survey : list, optional
		list of list_survey for which to print density. If not provided, take density.keys().
	path : str
		path to the density file.
	tab : int, optional
		seperation between columns.

	"""
	if not list_survey:
		list_survey = density.keys()
	columns = ['Mean z','n(z) [(h/Mpc)^3]']
	width_survey = tab*len(columns)
	width = width_survey*len(list_survey)
	with open(path,'w') as file:
		file.write(log_header('Redshift density',width=width,beg='') + '\n')
		file.write('{:<{tab}}'.format('#'+list_survey[0],tab=width_survey))
		for survey in list_survey[1:]: file.write('{:<{tab}}'.format(survey,tab=width_survey))
		file.write('\n{:<{tab}}'.format('#Eff. area(deg^2)',tab=tab))
		for survey in list_survey: file.write('{:<{tab}.4g}'.format(density[survey]['area'],tab=width_survey))
		file.write('\n{:<{tab}}'.format('#'+columns[0],tab=tab))
		for field in columns[1:]: file.write('{:<{tab}}'.format(field,tab=tab))
		for survey in list_survey[1:]:
			for field in columns: file.write('{:<{tab}}'.format(field,tab=tab))
		file.write('\n')
		for iline in range(len(density[list_survey[0]]['Z'])):
			for survey in list_survey: file.write('{:<{tab}.5g}{:<{tab}.5g}'.format(density[survey]['Z'][iline],density[survey]['NZ'][iline],tab=tab))
			file.write('\n')
		file.close()
	logger.info('Saving density to {}.'.format(path))


def log_header(txt,width=80,char='#',beg='\n'):
	"""Make log header from header txt."""
	header = beg + char*width + '\n'
	header += '{txt:{char}{align}{tab}}'.format(txt=' '+txt+' ',char=char,align='^',tab=width) + '\n'
	header += char*width + '\n'
	return header

def load(save,comments='#'):
	"""Load json file, ignoring comments."""
	logger.info('Loading {}.'.format(save))
	with open(save,'r') as ff:
		fixed_json = ''.join(line for line in ff if not line.startswith(comments))
		return json.loads(fixed_json,cls=JSONDecoder)
              
def save(save,array,comments=''):
	"""Save json file, adding comments."""
	logger.info('Saving {}.'.format(save))
	with open(save,'w') as ff:
		if comments: ff.write(comments+'\n')
		json.dump(array,ff,cls=JSONEncoder)

def mkdir(path):
	"""Make directory if does not exist already."""
	path = os.path.abspath(path)
	if not os.path.isdir(path): os.makedirs(path)

def savefig(path,*args,**kwargs):
	"""Save matplotlib figure."""
	mkdir(os.path.dirname(path))
	logger.info('Saving figure to {}.'.format(path))
	matplotlib.pyplot.savefig(path,*args,**kwargs)
	matplotlib.pyplot.close(matplotlib.pyplot.gcf())

def suplabel(axis,label,shift=0,labelpad=5,ha='center',va='center',**kwargs):
	"""Add super ylabel or xlabel to the figure. Similar to matplotlib.suptitle.
	Taken from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots.

	Parameters
	----------
	axis : str
		'x' or 'y'.
	label : str
		label.
	shift : float, optional
		shift.
	labelpad : float, optional
		padding from the axis.
	ha : str, optional
		horizontal alignment.
	va : str, optional
		vertical alignment.
	kwargs : dict
		kwargs for matplotlib.pyplot.text

	"""
	fig = matplotlib.pyplot.gcf()
	xmin = []
	ymin = []
	for ax in fig.axes:
		xmin.append(ax.get_position().xmin)
		ymin.append(ax.get_position().ymin)
	xmin,ymin = min(xmin),min(ymin)
	dpi = fig.dpi
	if axis.lower() == 'y':
		rotation = 90.
		x = xmin - float(labelpad)/dpi
		y = 0.5 + shift
	elif axis.lower() == 'x':
		rotation = 0.
		x = 0.5 + shift
		y = ymin - float(labelpad)/dpi
	else:
		raise Exception('Unexpected axis: x or y')
	matplotlib.pyplot.text(x,y,label,rotation=rotation,transform=fig.transFigure,ha=ha,va=va,**kwargs)

def sorted_alpha_numerical_order(li):
	import re
	"""Sort the given iterable in the way that humans expect.
	Taken from https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python.

	""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))] 
	return sorted(li,key = alphanum_key)

_logging_handler = None

def setup_logging(log_level="info"):
    """
    Turn on logging, with the specified level.
    Taken from nbodykit: https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py.

    Parameters
    ----------
    log_level : 'info', 'debug', 'warning'
        the logging level to set; logging below this level is ignored.
    
    """

    # This gives:
    #
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Nproc = [2, 1, 1]
    # [ 000000.43 ]   0: 06-28 14:49  measurestats    INFO     Rmax = 120

    levels = {
            "info" : logging.INFO,
            "debug" : logging.DEBUG,
            "warning" : logging.WARNING,
            }

    logger = logging.getLogger();
    t0 = time.time()


    class Formatter(logging.Formatter):
        def format(self, record):
            s1 = ('[ %09.2f ]: ' % (time.time() - t0))
            return s1 + logging.Formatter.format(self, record)

    fmt = Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M ')

    global _logging_handler
    if _logging_handler is None:
        _logging_handler = logging.StreamHandler()
        logger.addHandler(_logging_handler)

    _logging_handler.setFormatter(fmt)
    logger.setLevel(levels[log_level])

#setup_logging()
logger = logging.getLogger('Utils')

class JSONEncoder(json.JSONEncoder):
	"""
	A subclass of :class:`json.JSONEncoder` that can also handle numpy arrays,
	complex values.
	Taken from nbodykit: https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py.

	"""
	def default(self, obj):

		# numpy arrays
		if isinstance(obj, numpy.ndarray):
			value = obj
			dtype = obj.dtype
			d = {
				'__dtype__' : dtype.str if dtype.names is None else dtype.descr,
				'__shape__' : value.shape,
				'__data__': value.tolist(),
			}
			return d
			
		# explicity convert numpy data types to python types
		# see: https://bugs.python.org/issue24313
		if isinstance(obj, numpy.floating):
			return float(obj)
			
		if isinstance(obj, numpy.integer):
			return int(obj)

		return json.JSONEncoder.default(self, obj)
		


class JSONDecoder(json.JSONDecoder):
	"""
	A subclass of :class:`json.JSONDecoder` that can also handle numpy arrays,
	complex values.
	Taken from nbodykit: https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py.

	"""
	@staticmethod
	def hook(value):
		def fixdtype(dtype):
			if isinstance(dtype, list):
				true_dtype = []
				for field in dtype:
					if len(field) == 3:
						true_dtype.append((str(field[0]), str(field[1]), field[2]))
					if len(field) == 2:
						true_dtype.append((str(field[0]), str(field[1])))
				return true_dtype
			return dtype

		def fixdata(data, N, dtype):
			if not isinstance(dtype, list):
				return data

			# for structured array,
			# the last dimension shall be a tuple
			if N > 0:
				return [fixdata(i, N - 1, dtype) for i in data]
			else:
				assert len(data) == len(dtype)
				return tuple(data)

		if '__dtype__' in value:
			dtype = fixdtype(value['__dtype__'])
			shape = value['__shape__']
			a = fixdata(value['__data__'], len(shape), dtype)
			return numpy.array(a, dtype=dtype)

		return value

	def __init__(self, *args, **kwargs):
		kwargs['object_hook'] = JSONDecoder.hook
		json.JSONDecoder.__init__(self, *args, **kwargs)
