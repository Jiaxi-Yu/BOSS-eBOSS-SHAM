import logging
import os
import textwrap
import scipy
from scipy import stats,spatial,constants
from astropy import cosmology
from astropy.coordinates import SkyCoord
from catalogue import Catalogue
from catalogueELG import ELGCatalogue,Header,DataModel
import photometric_correction as photo
import spectroscopic_correction as spectro
import utils

logger = logging.getLogger('MkELG')

def set_parameters(params):
	global parameters
	parameters = params
	photo.set_parameters(params)
	spectro.set_parameters(params)

###############################################################################
# IOs
###############################################################################

def options_to_string(options):
	OPTIONS = {'keys':['option_los','option_mask','option_photo','option_cp','option_noz','option_rand_radec','option_rand_z'],'shorts':['los','mask','photo','cp','noz','randradec','randz','alpha','mean','norm'],'default':'none'}
	string = utils.dict_to_string(OPTIONS['keys'],OPTIONS['shorts'],options,mainsep='_',auxsep='-')
	return string if string else OPTIONS['default']

def dir_catalogues(experiment,catalogue,version,options):
	return os.path.join(os.getenv('ELGCAT'),'{experiment}_{catalogue}_{version}_{options}/'.format(experiment=experiment,
			catalogue=catalogue,version=version,options=options_to_string(options)))

def path_catalogues(experiment,catalogue,version,options,survey,type_data,num=None,argsdir=None):
	if argsdir is None: argsdir = [experiment,catalogue,version,options]
	if num is not None: return os.path.join(dir_catalogues(*argsdir),'{experiment}_ELG_{catalogue}_{survey}_{version}.{type_data}.{num}.fits'.format(experiment=experiment,
		catalogue=catalogue,version=version,survey=survey,type_data=type_data,num=num))
	return os.path.join(dir_catalogues(*argsdir),'{experiment}_ELG_{catalogue}_{survey}_{version}.{type_data}.fits'.format(experiment=experiment,
		catalogue=catalogue,version=version,survey=survey,type_data=type_data))

def path_mask_sky_lines(line,width):
	path_sky_lines = os.path.join(os.getenv('ELGCAT'),'skylines')
	return os.path.join(path_sky_lines,'redshift_mask_{line}_width-{width:.1f}.txt'.format(line=line,width=width))

def path_geometry(chunk):
	return os.path.join(os.getenv('ELGCAT'),'geometry','geometry-{}.ply'.format(chunk))

def save_log(path='log.txt'):
	"""Write log file.

	Parameters
	----------
	path : str, optional
		path to log file.

	"""
	options = parameters['options']	
	header = Header(**parameters['fits']['header'])
	datamodel = DataModel(parameters['fits']['data_model'])
	tab = datamodel.tabstr
	width = datamodel.widthstr
	widthbody = 200
	list_options = ['option_mask','option_photo','option_cp','option_noz','option_rand_radec','option_rand_z']

	with open(path,'w') as file:

		file.write(utils.log_header('Introduction',width=width,beg='') + '\n')
		file.write(header.tostr(tab=tab))
		file.write('\n')		

		wrapper = textwrap.TextWrapper(initial_indent='',subsequent_indent=' '*tab,width=widthbody)	
		file.write(utils.log_header('Used options',width=width) + '\n')
		for field in list_options:
			file.write(wrapper.fill('{:<{tab}}{}'.format(field,utils.list_to_string(options[field],sep='-'),tab=tab))+'\n')
		file.write('\n')

		wrapper = textwrap.TextWrapper(initial_indent='',subsequent_indent=' '*tab*(len(DataModel.DESCRIPTORS)-1),width=widthbody)
		file.write(utils.log_header('Data model',width=width) + '\n')
		file.write('\n'.join(map(wrapper.fill,datamodel.tostr(tab=tab,width=width).split('\n'))))
		file.write('\n')		

		wrapper = textwrap.TextWrapper(initial_indent='',subsequent_indent=' '*tab,width=widthbody)	
		file.write(utils.log_header('Description of options',width=width) + '\n')
		for field in list_options:
			file.write('{0:={align}{tab}}\n'.format(' ' + field + ' ',align='^',tab=2*tab))
			options = [opt for opt in parameters[field] if 'description' in parameters[field][opt]]
			options.sort(key=lambda item: (len(item), item))
			for opt in options: file.write(wrapper.fill('{:<{tab}}{}'.format(opt,parameters[field][opt]['description'],tab=tab))+'\n')
			file.write('\n')

		file.close()
	
	logger.info('Saving log to {}.'.format(path))
	
def save_data_model(datamodel,path='datamodel.txt'):
	"""Write data model.

	Parameters
	----------
	datamodel : DataModel
		the data model to save.
	path : str, optional
		path to log file.

	"""
	header = Header(**parameters['fits']['header'])
	datamodel = DataModel(datamodel)
	with open(path,'w') as file: 
		file.write(str(datamodel))

	logger.info('Saving data model to {}.'.format(path))

def make_data_statistics(path_data,path_randoms,fmt='txt',path='stats'):
	"""Write statistics file, following the same format as in Reid et al. 2015.
	Geom. area is the sum of the area of sectors filled with randoms.
	Unvetoed area is the sum of the randoms, divided by their original density (includes all veto masks).
	Effective area is Tot. area weighted by the BOSS completeness.
	
	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the randoms catalogue.
	fmt : str, list, optional
		in ['txt','tex']: export format
	path : str, optional
		path to the statistics file.

	"""
	logger.info(utils.log_header('Computing statistics'))
	warnings = scipy.seterr(divide='ignore',invalid='ignore')
	
	### Load data ###
	data = ELGCatalogue.load(path_data)
	
	### Preliminary cuts (remove duplicates) ###
	mask_target = data.subsample(parameters['target_subsample'],txt='target_subsample')

	list_survey = ELGCatalogue.LIST_CHUNK + ELGCatalogue.LIST_CAP + ['ALL']
	list_stats = ['Ntarg','Nobs','Nmissed','Ncp','Ngal','Nstar','Nfail','Nused','Mean TSR','Mean comp.',
				'Mean SSR','Geom. area (deg^2)','Photo. area (deg^2)','Unvetoed area (deg^2)','Effective area (deg^2)','Targets (deg^-2)']
	stats = {survey:{} for survey in list_survey}

	### Completeness ###
	mask_hasfiber = mask_target & data.subsample(parameters['fiber_subsample'],txt='fiber_subsample')
	imatch = data['IMATCH'] #_,imatch = utils.fiber_collision_group(groups=data['ELG_INGROUP'],mask_hasfiber=mask_hasfiber,mask_target=mask_target,return_imatch=True)
	mask_resolved = mask_target & (imatch >= 1) & (imatch != 14)
	assert (mask_target == (imatch != 12)).all()
	sector_target = utils.digitized_statistics(data['ELG_sector'],values=mask_target)
	data['sector_TSR'] = utils.digitized_statistics(data['ELG_sector'],values=mask_hasfiber)*1./sector_target
	data['COMP_BOSS'] = utils.digitized_statistics(data['ELG_sector'],values=mask_resolved)*1./sector_target
	
	### Reliable redshift ###
	mask_spectro_subsample = mask_hasfiber & data.subsample(parameters['spectro_subsample'],txt='spectro_subsample')
	mask_reliable_redshift = mask_spectro_subsample & data.subsample(parameters['reliable_redshift'],txt='reliable_redshift')
	data['sector_SSR'] = utils.digitized_statistics(data['ELG_sector'],values=mask_reliable_redshift)*1./utils.digitized_statistics(data['ELG_sector'],values=mask_spectro_subsample)

	### Selection cuts ###
	used = mask_reliable_redshift & data.subsample(parameters['final_subsample'],txt='final_subsample')
	
	### stats ###
	for survey in list_survey:
		mask_survey = data.survey(survey) & mask_target
		stats[survey]['Ntarg'] = scipy.sum(mask_survey)
		stats[survey]['Nobs'] = scipy.sum(mask_survey & mask_hasfiber)
		stats[survey]['Nmissed'] = scipy.sum(mask_survey & ((imatch == 0) | (imatch == 14)))
		stats[survey]['Ncp'] = scipy.sum(mask_survey & (imatch == 3))
		stats[survey]['Ngal'] = scipy.sum(mask_survey & (imatch == 1))
		stats[survey]['Nstar'] = scipy.sum(mask_survey & (imatch == 4))
		stats[survey]['Nfail'] = scipy.sum(mask_survey & (imatch == 7))
		stats[survey]['Nused'] = scipy.sum(mask_survey & used)

		assert stats[survey]['Ntarg'] == stats[survey]['Ngal'] + stats[survey]['Nstar'] + stats[survey]['Nfail'] + stats[survey]['Ncp'] + stats[survey]['Nmissed']
		assert stats[survey]['Nobs'] == stats[survey]['Ngal'] + stats[survey]['Nstar'] + stats[survey]['Nfail']
	
	### Load randoms ###
	randoms = ELGCatalogue.load(path_randoms)
	
	for key in ['COMP_BOSS','sector_TSR','sector_SSR']:
		randoms[key] = utils.digitized_interp(randoms['ELG_sector'],data['ELG_sector'],data[key],fill=0.)

	### All cuts ###
	mask_randoms_target = randoms.subsample(parameters['randoms_target_subsample'])
	mask_randoms_final = randoms.all_randoms_cuts(parameters,exclude=['Z'])
	
	weights = randoms['COMP_BOSS']
		
	### stats ###
	for survey in list_survey:
		mask_survey = randoms.survey(survey)
		stats[survey]['Geom. area (deg^2)'] = scipy.sum(randoms['sector_area'][mask_survey][scipy.unique(randoms['ELG_sector'][mask_survey],return_index=True)[-1]])
		stats[survey]['Photo. area (deg^2)'] = scipy.sum(mask_survey & mask_randoms_target)/parameters['density']['density_randoms']
		stats[survey]['Targets (deg^-2)'] = stats[survey]['Ntarg']/stats[survey]['Photo. area (deg^2)']
		stats[survey]['Unvetoed area (deg^2)'] = scipy.sum(mask_survey & mask_randoms_final)/parameters['density']['density_randoms']
		stats[survey]['Effective area (deg^2)'] = scipy.sum(weights[mask_survey & mask_randoms_final])/parameters['density']['density_randoms']
		stats[survey]['Mean TSR'] = scipy.mean(randoms['sector_TSR'][mask_survey & mask_randoms_final])
		stats[survey]['Mean comp.'] = scipy.mean(randoms['COMP_BOSS'][mask_survey & mask_randoms_final])
		stats[survey]['Mean SSR'] = scipy.mean(randoms['sector_SSR'][mask_survey & mask_randoms_final])

	### Writes output ###
		
	if isinstance(fmt,(str,unicode)):
		fmts = [fmt]
	else:
		fmts = fmt
	for fmt in fmts:
		path_ = '{}.{}'.format(path,fmt) if fmt != 'txt' else path
		utils.save_data_statistics(stats,list_survey,list_stats,path=path_,fmt=fmt)

	scipy.seterr(**warnings)
	logger.info('Computing statistics completed.')

def fits_to_rdzw(path_fits,path_rdzw):
	"""Export a fits catalogue to an ascii file, in the format:
	RA, Dec, Z, Weight.

	Parameters
	----------
	path_fits : str
		path to the fits catalogue.
	path_rdzw : str
		path to the ascii catalogue.

	"""
	logger.info(utils.log_header('Exporting to rdzw format'))
	
	catalogue = ELGCatalogue.load(path_fits)
	catalogue = condense_catalogue(catalogue)

	catalogue.save(path_rdzw)

	logger.info('Exporting to rdzw format completed.')

	
###############################################################################
# Catalogue construction
###############################################################################


def to_standard_fits_format(type_data='data',path_output='test.fits',path_data_model=None):
	"""Make the conversion from Anand's catalogues in parameters['paths'][type_data] to standard format.
	It adds units to columns and descriptions to header, following parameters['fits']['data_model'].
	The new fits file is saved to path_output and the file data model to path_data_model if provided.
	Essentially, redrock 'rr_' fields are converted to standard ones, and masking (e.g. maskbits, gaiamask) can be applied.

	Parameters
	----------
	type_data : str
		in ['data','randoms']; parameters['paths'][type_data] is Anand's catalogue.
	path_output : str
		path to the output catalogue.
	path_data_model : str, optional
		if provided, save data model in path_data_model.

	"""
	logger.info(utils.log_header('Exporting to standard fits format'))
	option_mask = parameters['options']['option_mask']

	### Open input fits ###
	catalogue = ELGCatalogue.load(parameters['paths'][type_data])
	
	catalogue.set_header(parameters['fits']['header'])
	catalogue.set_datamodel(parameters['fits']['data_model'])

	### Some additionnal conversion or precomputation stuff ###
	catalogue['ELG_sector'] = catalogue.ELG_sector
	keep = None
	
	### Redrock ###
	if (type_data == 'data'):
		logger.info('Converting redrock names.')
		catalogue.redrock_to_standard()
	
	### Masking ###
	mask = mask_catalogue(catalogue,option_mask,type_data=type_data)
	if not mask.all(): catalogue = catalogue[mask]

	mask = mask_maskbits(catalogue,type_data=type_data)
	catalogue['mskbit'] += (~mask)*2**8 # adding 2**8 for discrepancy between mskbit and anymask
	mask = mask_centerpost(catalogue)
	catalogue['mskbit'] += (~mask)*2**9 # adding 2*9 for centerpost masking
	mask = mask_tdss_fes(catalogue)
	catalogue['mskbit'] += (~mask)*2**10 # adding 2*10 for tdss_fes masking
	mask = mask_bad_exposures(catalogue)
	catalogue['mskbit'] += (~mask)*2**11 # adding 2*11 for bad calibration masking
	#mask = mask_nobs(catalogue,type_data=type_data)
	#catalogue['mskbit'] += (~mask)*2**11 # adding 2*11 for nobs masking
	#mask = mask_tycho2blob(catalogue)
	#catalogue['mskbit'] += (~mask)*2**4 # adding 2*4 for tycho2blob masking
	
	if type_data == 'data':
		catalogue = catalogue[~catalogue['isdupl']]
		catalogue['ELG_INGROUP'] = catalogue.ELG_INGROUP
		add_imatch(catalogue,key_imatch='IMATCH')
		#add_des(catalogue)
	
	if type_data == 'randoms':
		keep = parameters['fits']['fields_randoms']
		data = ELGCatalogue.load(parameters['paths']['data'])
		index_data = utils.match_ra_dec([catalogue['RA'].astype(scipy.float64),catalogue['DEC'].astype(scipy.float64)],
					[data['RA'].astype(scipy.float64),data['DEC'].astype(scipy.float64)],nn=1)
		for key in ['nobs_{}'.format(b) for b in ELGCatalogue.LIST_BAND]:
			catalogue[key] = data[key][index_data]	# adds nobs to the randoms based on the nearest neighbor in the data (not used)

	catalogue.save(path_output,path_data_model=path_data_model,keep=keep)

	logger.info('Exporting to standard fits format completed.')


def make_data(path_data,path_output,import_maps=False,import_healpix=False,option_photo=None):
	"""Make the data catalogue, applying completeness weights:
	* photometric weights (WEIGHT_SYSTOT), as provided in the healpix map parameters['paths']['healpix_density'],
	* collision pair weights (WEIGHT_CP),
	* redshift failures weights (WEIGHT_NOZ).
	Options for this weights are given by 'option_photo', 'option_cp', 'option_noz' in parameters['options'],
	respectively.
	No cut is performed at this stage: output is full catalogue.

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_output : str
		path to the output catalogue.
	import_maps : bool, optional
		whether to import photometric parameters provdided in external (dust)maps.
	import_healpix : bool, optional
		whether to import healpix photometric parameters (hpgaldepth, etc.).
	option_photo : str, optional
		if provided, overrides the value given in parameters['options']['option_photo'].

	"""
	logger.info(utils.log_header('Making data catalogue'))

	if option_photo is None: option_photo = parameters['options']['option_photo']
	option_cp = parameters['options']['option_cp']
	option_noz = parameters['options']['option_noz']

	warnings = scipy.seterr(divide='ignore',invalid='ignore')

	### Load data ###
	catalogue = ELGCatalogue.load(path_data)
	
	### Import photo maps ###
	if import_maps: photo.add_maps_values(catalogue)
	
	### Import Healpix ###
	if import_healpix: photo.add_healpix_values(catalogue)

	### Preliminary cuts ###
	mask_target = catalogue.subsample(parameters['target_subsample'],txt='target_subsample')

	### Collision pairs ###
	logger.info('Correction for collision pairs: {}.'.format(option_cp))

	mask_hasfiber = mask_target & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')
	catalogue['WEIGHT_CP'][:] = 1.
	catalogue['MASK_CP'] = mask_target.copy()
	if option_cp == 'no': pass
	elif option_cp == 'colgroup': add_weight_collision_group(catalogue,key_weight='WEIGHT_CP')
	else: logger.warning('Correction for collision pairs {} is not available. WEIGHT_CP set to 1.'.format(option_cp))
	catalogue.fill_default_value('WEIGHT_CP',~mask_hasfiber)

	### Update of the Tiling Success Rate (TSR) ###
	sector_target = utils.digitized_statistics(catalogue['ELG_sector'],values=mask_target)
	catalogue['sector_TSR'] = utils.digitized_statistics(catalogue['ELG_sector'],values=mask_hasfiber)*1./sector_target
	mask_resolved = mask_target & catalogue['MASK_CP']
	catalogue['COMP_BOSS'] = utils.digitized_statistics(catalogue['ELG_sector'],values=mask_resolved)*1./sector_target
	for comp in ['sector_TSR','COMP_BOSS']: logger.info('{} range: {:.4f} - {:.4f}.'.format(comp,catalogue[comp][mask_target].min(),catalogue[comp][mask_target].max()))

	### Cuts to get the subsample of spectroscopic objects ###
	mask_spectro_subsample = mask_hasfiber & catalogue.subsample(parameters['spectro_subsample'],txt='spectro_subsample')

	### Reliable redshift ###
	mask_reliable_redshift = mask_spectro_subsample & catalogue.subsample(parameters['reliable_redshift'],txt='reliable_redshift')

	### Update of the Spectroscopic Success Rate (SSR) ###
	catalogue['sector_SSR'] = utils.digitized_statistics(catalogue['ELG_sector'],values=mask_reliable_redshift)*1./utils.digitized_statistics(catalogue['ELG_sector'],values=mask_spectro_subsample)
	catalogue['plate_MJD'],catalogue['plate_MJD_SSR'],catalogue['plate_MJD_SN_MEDIAN_ALL'] = spectro.get_plate_mjd_stats(catalogue,mask=None,key_plate='plate_MJD',key_sn='SN_MEDIAN_ALL')
	for comp in ['sector_SSR','plate_MJD_SSR']: logger.info('{} range: {:.4f} - {:.4f}.'.format(comp,catalogue[comp][mask_spectro_subsample].min(),catalogue[comp][mask_spectro_subsample].max()))

	### Correction for redshift failures ###
	logger.info('Correction for redshift failures: {}.'.format(option_noz))
	
	catalogue['WEIGHT_NOZ'][:] = 1.
	if (option_noz == 'no') or option_noz.startswith('ran-'): pass
	elif option_noz == 'sector':
		catalogue['WEIGHT_NOZ'] = 1./catalogue['sector_SSR']
	elif option_noz == 'plate': 
		catalogue['WEIGHT_NOZ'] = 1./catalogue['plate_MJD_SSR']
	elif option_noz == 'fiberid': add_weight_noz_fiberid(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ')
	elif option_noz == 'platefiberid': add_weight_noz_platefiberid(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ')
	elif option_noz == 'platexyfocal': add_weight_noz_platexyfocal(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ')
	elif option_noz == 'nearestneighbor': add_weight_noz_nearestneighbor(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ')
	elif option_noz == 'fitplatesnfiberid': add_weight_noz_fitplatesnfiberid(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ',path_plate_ssr=parameters['paths'].get('fit_plate_ssr',None))
	#elif option_noz == 'fitplatesnfitfiberid': catalogue['WEIGHT_NOZ'] = 1./(catalogue['zfail_platesn']*catalogue['zfail_fiberid']) # when Anand provided zfail_platesn, zfail_fiberid
	elif option_noz == 'fitplatesnfitfiberid': add_weight_noz_fitplatesnfitfiberid(catalogue,key_weight='WEIGHT_NOZ',path_plate_ssr=parameters['paths'].get('fit_plate_ssr',None),path_fiberid_ssr=parameters['paths'].get('fit_fiberid_ssr',None))
	elif option_noz == 'fitplatesnfitxyfocal': add_weight_noz_fitplatesnfitxyfocal(catalogue,key_weight='WEIGHT_NOZ',path_plate_ssr=parameters['paths'].get('fit_plate_ssr',None),path_xyfocal_ssr=parameters['paths'].get('fit_xyfocal_ssr',None))
	else: logger.warning('Correction for redshift failures {} is not available. WEIGHT_NOZ set to 1.'.format(option_noz))
	if 'mag' in option_noz: add_weight_noz_magcorrection(catalogue,mask_resolved,mask_reliable_redshift,key_weight='WEIGHT_NOZ')
	catalogue.fill_default_value('WEIGHT_NOZ',~mask_reliable_redshift)

	### Photometric systematic weights ###
	logger.info('Correction for photometry: {}.'.format(option_photo))
	catalogue['WEIGHT_SYSTOT'][:] = 1.
	if option_photo == 'no': pass
	elif option_photo.startswith('hp'): photo.add_weight_healpix(catalogue,key_weight_pixel='hpweight_{}'.format(option_photo.replace('hp','')),key_redshift='Z',key_weight='WEIGHT_SYSTOT',to_weight='data',mask=mask_target)
	elif option_photo.startswith('obj'): photo.add_weight_object(catalogue,key_redshift='Z',key_weight='WEIGHT_SYSTOT',to_weight='data',import_healpix=parameters['fit_photo']['params'],mask=mask_target)
	else: logger.warning('Correction for photometry {} is not available. WEIGHT_SYSTOT set to 1.'.format(option_photo))
		
	normalize_data_weights(catalogue)	

	catalogue.save(path_output)
	
	scipy.seterr(**warnings)
	logger.info('Making data completed.')

def make_randoms(path_data,path_randoms,path_output_data,path_output_randoms,import_maps=False,import_healpix=False,only_radec=False):
	"""Make the random catalogue:
	* weights are controlled by options parameters['options']['option_photo'] and parameters['options']['option_noz'].
	* random RA/Dec can be taken from the data RA/Dec distribution, following parameters['options']['option_rand_radec'].
	* random redshifts are drawn from the data redshift distribution, following parameters['options']['option_rand_z'].
	* density is computed for both data and random catalogues, following parameters['density'].
	The area considered in the density estimation is the effective area, i.e. the unvetoed area times the BOSS completeness.
	You shall not trust the power spectrum normalization given by:
	.. math::
			A = \sum w_{\mathrm{comp}} \bar{n} w_{\mathrm{fkp}}^{2}.
	See: https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py
	No cut is performed: output is full catalogue.

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the random catalogue.
	path_output_data : str
		path to the output data catalogue.
	path_output_randoms : str
		path to the output random catalogue.
	import_maps : bool, optional
		whether to import photometric parameters provdided in external (dust)maps.
	import_healpix : bool, optional
		whether to import healpix photometric parameters (hpgaldepth, etc.).
	only_radec : bool, optional
		whether to stop the random catalogue production after RA/DEC angular weights.

	"""
	logger.info(utils.log_header('Making random catalogue'))

	option_noz = parameters['options']['option_noz']
	option_photo = parameters['options']['option_photo']
	option_rand_radec = parameters['options']['option_rand_radec']
	option_rand_z = parameters['options']['option_rand_z']

	warnings = scipy.seterr(divide='ignore',invalid='ignore')

	### Load data and randoms ###
	data = ELGCatalogue.load(path_data)
	randoms = ELGCatalogue.load(path_randoms)
	
	### Import photo maps ###
	if import_maps: photo.add_maps_values(randoms)

	### Import Healpix ###
	if import_healpix: photo.add_healpix_values(randoms)
	
	for key in ['COMP_BOSS','sector_TSR','sector_SSR']:
		randoms[key] = utils.digitized_interp(randoms['ELG_sector'],data['ELG_sector'],data[key],fill=scipy.nan)

	plate_mjd_data = data.plate_mjd
	plate_mjd_randoms = randoms.plate_mjd
	for key in ['plate_MJD_SSR','plate_MJD_SN_MEDIAN_ALL']:
		randoms[key] = utils.digitized_interp(plate_mjd_randoms,plate_mjd_data,data[key],fill=scipy.nan)
	
	if only_radec:
		randoms.save(path_output_randoms)
		scipy.seterr(**warnings)
		logger.info('Making randoms completed.')
		return

	### Randoms RA/DEC ###
	list_survey = parameters['cut_surveys']['list']
	if option_rand_radec == 'final':
		mask_data = data.all_data_cuts(parameters)
		for survey in list_survey:
			mask_data_survey = mask_data & data.survey(survey)
			mask_randoms_survey = randoms.survey(survey)
			shuffle_radec(data,randoms,mask_data=mask_data_survey,mask_randoms=mask_randoms_survey,seed=parameters['option_rand_radec']['final'].get('seed',None))

	### Randoms Z ###
	prob_data = data.weight_object
	add_random_redshifts(data,randoms,option_rand_z=option_rand_z,prob_data=prob_data,mask_data=data.all_data_cuts(parameters,exclude=['Z']),mask_randoms=randoms.all_randoms_cuts(parameters,exclude=['Z']),key_redshift='Z',key_model_z='comb_galdepth',path_spectro=parameters['paths'].get('fit_spectro',None))
	
	### Correction for photometry ###
	mask_randoms = randoms.subsample(parameters['randoms_target_subsample'],txt='randoms_target_subsample')
	randoms['WEIGHT_SYSTOT'][:] = 1.
	if option_photo.startswith('ran-'):
		option_photo = option_photo.replace('ran-','')
		if option_photo.startswith('hp'): photo.add_weight_healpix(randoms,key_weight_pixel='hpweight_{}'.format(option_photo.replace('hp','')),key_redshift='Z',key_weight='WEIGHT_SYSTOT',mask=mask_randoms)
		elif option_photo.startswith('obj'): photo.add_weight_object(randoms,key_redshift='Z',key_weight='WEIGHT_SYSTOT',import_healpix=['hpgaiastardens'],clip=None,interpolate='spline',extrapolate='lim',flatten=False,mask=mask_randoms)
		else: logger.warning('Correction for photometry {} is not available. WEIGHT_SYSTOT set to 1.'.format(option_rand_z))
	
	list_survey = parameters['cut_surveys']['list']
	for survey in list_survey:
		mask_survey = randoms.survey(survey)
		factor = 1./scipy.mean(randoms['WEIGHT_SYSTOT'][mask_randoms & mask_survey])
		randoms['WEIGHT_SYSTOT'][mask_survey] *= factor
		logger.info('Renormalizing WEIGHT_SYSTOT in {} by {:.4f} such that mean(WEIGHT_SYSTOT) = 1.'.format(survey,factor))
	
	### Correction for Tiling Success Rate (TSR) ###
	randoms['WEIGHT_SYSTOT'] *= randoms['COMP_BOSS']
	randoms['WEIGHT_SYSTOT'][utils.isnaninf(randoms['WEIGHT_SYSTOT'])] = 0.
	randoms['WEIGHT_CP'][:] = 1.

	### Correction for redshift failures ###
	logger.info('Correction for redshift failures: {}.'.format(option_noz))
	randoms['WEIGHT_NOZ'][:] = 1.
	if option_noz == 'no': pass
	elif option_noz == 'ran-sector':
		randoms['WEIGHT_NOZ'][:] = randoms['sector_SSR'][:]
		randoms['WEIGHT_NOZ'][utils.isnaninf(randoms['WEIGHT_NOZ'])] = 0.
	else: logger.warning('Correction for redshift failures {} is not available for the randoms. WEIGHT_NOZ set to 1.'.format(option_noz))
	
	if option_photo.startswith('obj'):
		zedges = scipy.linspace(randoms['Z'].min(),randoms['Z'].max()+1e-9,1000)
		mask_zdata = data.all_data_cuts(parameters,exclude=['Z'])
		mask_zrandoms = randoms.all_randoms_cuts(parameters,exclude=['Z'])
		for ichunkz in scipy.unique(data['chunk_z'][mask_zdata]):
			logger.info('Renormalizing redshift density in chunk_z {:d}.'.format(ichunkz))
			mask_data_survey = mask_zdata & (data['chunk_z'] == ichunkz)
			mask_randoms_survey = randoms['chunk_z'] == ichunkz
			if not (mask_zrandoms & mask_randoms_survey).any():
				logger.info('No randoms in chunk_z {:d}. Skipping.'.format(ichunkz))
				continue
			normalize_redshift_density(data,randoms,mask_data=mask_data_survey,mask_randoms=mask_zrandoms,mask_randoms_survey=mask_randoms_survey,key_redshift='Z',key_weight='WEIGHT_SYSTOT',edges=zedges)
	
	### Density ###
	cosmo = cosmology.wCDM(**parameters['density']['cosmology'])
	edges = parameters['density']['edges']
	density = {}
	mask_zdata = data.all_data_cuts(parameters,exclude=['Z'])
	mask_zrandoms = randoms.all_randoms_cuts(parameters,exclude=['Z'])
	for survey in list_survey:
		density[survey] = {}
		density[survey]['Z'],density[survey]['NZ'],density[survey]['area'] = add_redshift_density(data,randoms,mask_data=mask_zdata,mask_randoms=mask_zrandoms,mask_data_survey=data.survey(survey),mask_randoms_survey=randoms.survey(survey),cosmo=cosmo,edges=edges,key_redshift='Z',key_density='NZ')

	path_density = parameters['paths'].get('density',None)
	if path_density is not None: utils.save_density(density,list_survey=list_survey,path=path_density)
	data['WEIGHT_FKP'] = data.weight_fkp(key_density='NZ',P0=parameters['option_fkp']['P0'])
	randoms['WEIGHT_FKP'] = randoms.weight_fkp(key_density='NZ',P0=parameters['option_fkp']['P0'])
	
	data.save(path_output_data)
	
	if option_rand_radec == 'pixel':
		mask_data = data.all_data_cuts(parameters)
		mask_randoms = randoms.all_randoms_cuts(parameters)
		for survey in list_survey:
			mask_data_survey = mask_data & data.survey(survey)
			mask_randoms_survey = mask_randoms & randoms.survey(survey)
			for ichunkz in scipy.unique(data['chunk_z'][mask_data_survey]):
				mask_data_chunk_z = mask_data_survey & (data['chunk_z'] == ichunkz)
				mask_randoms_chunk_z = mask_randoms_survey & (randoms['chunk_z'] == ichunkz)
				params = parameters['option_rand_radec']['pixel']
				pixel_radec(data,randoms,mask_data=mask_data_chunk_z,mask_randoms=mask_randoms_chunk_z,nside=params['nside'],nest=params['nest'])

	normalize_random_weights(data,randoms)

	randoms.save(path_output_randoms)
	
	scipy.seterr(**warnings)
	logger.info('Making randoms completed.')

def add_imatch(catalogue,key_imatch='IMATCH'):
	"""Add imatch, as defined in
	https://trac.sdss.org/wiki/eBOSS/QGC/LSScats/DR16#Objectclasses (as of Feb. 2019).

	Parameters
	----------
	catalogue : ELGCatalogue
		data catalogue.
	key_imatch : str, optional
		the catalogue field to save imatch in.

	"""
	mask_target = catalogue.subsample(parameters['target_subsample'],txt='target_subsample')
	mask_hasfiber = mask_target & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')
	_,imatch = utils.fiber_collision_group(groups=catalogue['ELG_INGROUP'],mask_hasfiber=mask_hasfiber,mask_target=mask_target,return_imatch=True)
	mask_spectro_subsample = mask_hasfiber & catalogue.subsample(parameters['spectro_subsample'],txt='spectro_subsample')
	mask_reliable_redshift = mask_spectro_subsample & catalogue.subsample(parameters['reliable_redshift'],txt='reliable_redshift')
	imatch[mask_hasfiber & mask_reliable_redshift] = 1
	imatch[mask_hasfiber & (catalogue['SPECTYPE'] == 'STAR')] = 4
	#imatch[mask_hasfiber & (catalogue['SPECTYPE'] == 'QSO')] = 9 # QSOs are ELGs
	imatch[mask_hasfiber & mask_spectro_subsample & (~mask_reliable_redshift)] = 7
	imatch[mask_target & (imatch != 3) & (catalogue['hasfiber'] == -1)] = 14
	assert scipy.all((imatch == 0) | (imatch == 1) | (imatch == 3) | (imatch == 4) | (imatch == 7) | (imatch == 12) | (imatch == 14))
	
	catalogue[key_imatch] = imatch

def normalize_data_weights(catalogue,mask=None):

	if mask is None: mask = catalogue.trues()
	mask_target = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample')
	
	list_survey = parameters['cut_surveys']['list']
	for survey in list_survey:
		mask_survey = catalogue.survey(survey)
		factor = 1./scipy.mean(catalogue['WEIGHT_SYSTOT'][mask_target & mask_survey])
		catalogue['WEIGHT_SYSTOT'][mask_survey] *= factor
		logger.info('Renormalizing WEIGHT_SYSTOT in {} by {:.4f} such that mean(WEIGHT_SYSTOT) = 1.'.format(survey,factor))

	for weight in ['WEIGHT_CP','WEIGHT_NOZ']: catalogue[weight][~mask_target] = 0.
	mask_hasfiber = mask_target & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')
	mask_resolved = mask_target & catalogue['MASK_CP']
	mask_spectro_subsample = mask_hasfiber & catalogue.subsample(parameters['spectro_subsample'],txt='spectro_subsample')
	mask_reliable_redshift = mask_spectro_subsample & catalogue.subsample(parameters['reliable_redshift'],txt='reliable_redshift')
	catalogue['WEIGHT_NOZ'][mask_target & ~mask_hasfiber] = 0. # convention
	catalogue['WEIGHT_NOZ'][mask_hasfiber & ~mask_spectro_subsample] = 1. # convention
	for survey in list_survey:
		mask_survey = catalogue.survey(survey)
		mask_survey_reliable = mask_survey & mask_hasfiber & mask_reliable_redshift
		mask_survey_stars = mask_survey & mask_hasfiber & ~mask_spectro_subsample # ~mask_spectro_subsample are stars
		factor = (scipy.sum(catalogue['WEIGHT_SYSTOT'][mask_survey & mask_resolved])-scipy.sum(catalogue.weight_object[mask_survey_stars]))/scipy.sum(catalogue.weight_object[mask_survey_reliable])
		logger.info('Renormalizing WEIGHT_NOZ in {} by factor {:.4f} such that sum(WEIGHT_SYSTOT*WEIGHT_CP*WEIGHT_NOZ)/reliable redshifts or stars = sum(WEIGHT_SYSTOT)/resolved fibers.'.format(survey,factor))
		catalogue['WEIGHT_NOZ'][mask_survey_reliable] *= factor
	
	catalogue['WEIGHT_CP'][mask & ~mask_hasfiber] = 0.
	catalogue['WEIGHT_NOZ'][mask_hasfiber & ~mask_reliable_redshift] = 0.
	mask_final = mask_hasfiber & mask_reliable_redshift
	for weight in ['WEIGHT_SYSTOT','WEIGHT_CP','WEIGHT_NOZ']: logger.info('{} range: {:.4f} - {:.4f}.'.format(weight,catalogue[weight][mask_final].min(),catalogue[weight][mask_final].max()))

def add_redshift_in_window(data,randoms,mask_data=None,mask_randoms=None,key_redshift='Z',rng=None,seed=None,prob_data=None,key_sort='comb_galdepth',step_fraction_key_sort=0.01,fraction_below=0.1,fraction_above=0.1):
	"""Fill random catalogue with shuffled data redshifts.
	Data redshifts are first sorted according to key_sort. An array of key_sort,
	subsampled by step_fraction_key_sort, is produced. Then, for each random object, its nearest
	neighbor in the subsampled array of key_sort is determined. The subsample of data
	containing this key_sort value, with the fraction fraction_below of data below
	and fraction_above above is taken as a the pool of redshifts to take the random redshift from.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.
	key_redshift : str, optional
		catalogue field to save the redshift values in.
	rng : RandomState, optional
		numpy RandomState.
	seed : int, optional
		random seed.
	prob_data : array, optional
		probability associated with the data redshift.
		If not provided, the normalized total data completeness weight is taken as probablity.
	key_sort : str, optional
		key to sort data redshift (e.g. comb_galdepth).
	step_fraction_key_sort : float, optional
		the subsampling fraction of key_sort (used to speed up calculations).
	fraction_below : float
		the fraction of data below key_sort to pick random redshifts from.
	fraction_above : float
		the fraction of data above key_sort to pick random redshifts from.

	"""
	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()
	if rng is None: rng = scipy.random.RandomState(seed=seed)
	
	### Let's sort redshifts and prob by increasing values of key_sort (= comb_galdepth) ###
	sortdata = scipy.argsort(data[mask_data][key_sort])
	zdata = data[key_redshift][mask_data][sortdata]
	probdata = prob_data[mask_data][sortdata] if prob_data is not None else data.weight_object[mask_data][sortdata]
	zmodelstep = max(int(step_fraction_key_sort*mask_data.sum()),1)
	zmodeldata = data[key_sort][mask_data][sortdata][::zmodelstep]	# subsample key_sort to speed above later calculations; nan are at the end
	
	tmpz = scipy.nan*scipy.ones_like(randoms[key_redshift][mask_randoms])
	treedata = spatial.cKDTree(zmodeldata[:,None],leafsize=16,compact_nodes=True,copy_data=False,balanced_tree=True)
	_,indexdata = treedata.query(randoms[key_sort][mask_randoms][:,None],k=1,eps=0,p=1,distance_upper_bound=scipy.inf,n_jobs=8)	# fetch the closed comb_galdepth in the data; nan are at the end
	percentile_above = int(fraction_above*len(zmodeldata))
	percentile_below = int(fraction_below*len(zmodeldata))
	for ival,val in enumerate(zmodeldata):	# loop over the different (subsampled) values of key_sort
		above = min(ival+percentile_above,zmodeldata.shape[-1])*zmodelstep	# below:above is the interval in the data where we will pick random redshifts
		below = max(ival-percentile_below,0)*zmodelstep
		ivalmask = (indexdata == ival)
		p = scipy.copy(probdata[below:above])
		nrand = ivalmask.sum()
		tmpz[ivalmask] = rng.choice(zdata[below:above],size=nrand,replace=True,p=p/p.sum())	# let's pick random redshifts into the data, with probability p prop. to WEIGHT_SYSTOT*WEIGHT_CP*WEIGHT_NOZ/COMP_BOSS
	
	randoms[key_redshift][mask_randoms] = tmpz[:]
	
	if scipy.any(mask_randoms & randoms.bad_value(key_redshift)):	
		logger.warning('{} has incorrect ({}) values.'.format(key_redshift,ELGCatalogue.bad_values(key_redshift)))


def add_redshift_in_bin(data,randoms,mask_data=None,mask_randoms=None,key_redshift='Z',rng=None,seed=None,prob_data=None,bins=[{}]):
	"""Fill randoms catalogue with data redshifts in bins.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.
	key_redshift : str, optional
		catalogue field to save the redshift values in.
	rng : RandomState, optional
		numpy RandomState.
	seed : int, optional
		random seed.
	prob_data : array, optional
		probability associated with the data redshift.
		If not provided, the normalized total data completeness weight is taken as probablity.
	bins : list of dict, optional
		the different subsamples to take the random redshifts from.
		Dictionaries must be of the form {field1: value, field2: [low, high]}, etc.
		as e.g. in parameters['spectro_subsample'].

	Returns
	-------
	ibin_data : array
		data bins corresponding to each quantile.
	ibin_randoms : array
		randoms bins corresponding to each quantile.

	"""
	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()
	if rng is None: rng = scipy.random.RandomState(seed=seed)
	probdata = prob_data if prob_data is not None else data.weight_object
	ibin_data = data.zeros(dtype='int8')
	ibin_randoms = randoms.zeros(dtype='int8')

	for ibin,bin in enumerate(bins):
		mask_data_bin = mask_data & data.subsample(bin,txt='bin')
		if not scipy.any(mask_data_bin):
			logger.warning('Skipping empty bin {:d}: {}.'.format(ibin,bin))
			continue
		mask_randoms_bin = mask_randoms & randoms.subsample(bin,txt='bin')
		p = probdata[mask_data_bin]
		zdata = data[key_redshift][mask_data_bin]
		nrand = mask_randoms_bin.sum()
		randoms[key_redshift][mask_randoms_bin] = rng.choice(zdata,size=nrand,replace=True,p=p/p.sum())
		ibin_data[mask_data_bin] = ibin + 1
		ibin_randoms[mask_randoms_bin] = ibin + 1

	if scipy.any(mask_randoms & randoms.bad_value(key_redshift)):	
		logger.warning('{} has incorrect ({}) values.'.format(key_redshift,ELGCatalogue.bad_values(key_redshift)))

	return ibin_data,ibin_randoms

def add_redshift_in_quantile(data,randoms,mask_data=None,mask_randoms=None,mask_data_survey=None,mask_randoms_survey=None,key_redshift='Z',rng=None,seed=None,prob_data=None,key_quantile='comb_galdepth',quantiles=[0.,1.]):
	"""Fill randoms catalogue with data redshifts in quantiles.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.
	mask_data_survey : boolean array, optional
		if provided, survey mask applied on the data. Redshifts are taken in mask_data & mask_data_survey.
	mask_randoms_survey : boolean array, optional
		if provided, survey mask applied on the randoms. Quantiles will be taken in mask_randoms & mask_randoms_survey.
	key_redshift : str, optional
		catalogue field to save the redshift values in.
	rng : RandomState, optional
		numpy RandomState.
	seed : int, optional
		random seed.
	prob_data : array, optional
		probability associated with the data redshift.
		If not provided, the normalized total data completeness weight is taken as probablity.
	key_quantile : str, optional
		key to split data and randoms in quantiles.
	quantiles : list, optional
		the different quantiles to take the random redshifts from.
		
	Returns
	-------
	ibin_data : array
		data bins corresponding to each quantile.
	ibin_randoms : array
		randoms bins corresponding to each quantile.

	"""
	if mask_data_survey is None: mask_data_survey = data.trues()
	if mask_randoms_survey is None: mask_randoms_survey = randoms.trues()
	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()
	mask_data = mask_data & mask_data_survey
	mask_randoms = mask_randoms & mask_randoms_survey
	
	if rng is None: rng = scipy.random.RandomState(seed=seed)
	probdata = prob_data if prob_data is not None else data.weight_object

	ibin_data = data.zeros(dtype='int8')
	ibin_randoms = randoms.zeros(dtype='int8')
	for iquantile,quantile in enumerate(zip(quantiles[:-1],quantiles[1:])):
		low,up = scipy.percentile(randoms[key_quantile][mask_randoms],scipy.array(quantile)*100.)
		if quantile[0] <= 0.: low = min(low,data[key_quantile][mask_data_survey].min()) - 1. #to include first element
		if quantile[-1] >= 1.: up = max(up,data[key_quantile][mask_data_survey].max()) + 1. #to include last element
		if quantile[0] <= 0.: low = min(low,randoms[key_quantile][mask_randoms_survey].min()) - 1. #to include first element
		if quantile[-1] >= 1.: up = max(up,randoms[key_quantile][mask_randoms_survey].max()) + 1. #to include last element
		mask_data_quantile = mask_data_survey & (data[key_quantile] >= low) & (data[key_quantile] < up)
		mask_randoms_quantile = mask_randoms_survey & (randoms[key_quantile] >= low) & (randoms[key_quantile] < up)
		p = probdata[mask_data & mask_data_quantile]
		zdata = data[key_redshift][mask_data & mask_data_quantile]
		nrand = mask_randoms_quantile.sum()
		randoms[key_redshift][mask_randoms_quantile] = rng.choice(zdata,size=nrand,replace=True,p=p/p.sum())
		ibin_data[mask_data_quantile] = iquantile + 1
		ibin_randoms[mask_randoms_quantile] = iquantile + 1

	if scipy.any(mask_randoms & randoms.bad_value(key_redshift)):	
		logger.warning('{} has incorrect ({}) values.'.format(key_redshift,ELGCatalogue.bad_values(key_redshift)))

	return ibin_data,ibin_randoms

def add_model_z(data,randoms,mask_data=None,mask_randoms=None,key_redshift='Z',key_model_z='comb_galdepth',path_spectro=None):
	results = spectro._fit_redshift_(data,key_redshift=key_redshift,mask=mask_data)
	if path_spectro is not None: utils.save(path_spectro,results)
	spectro._add_model_z_(data,results,key_model_z=key_model_z,mask=mask_data)
	spectro._add_model_z_(randoms,results,key_model_z=key_model_z,mask=mask_randoms)

def add_random_redshifts(data,randoms,option_rand_z='bin',prob_data=None,mask_data=None,mask_randoms=None,key_redshift='Z',key_model_z='comb_galdepth',path_spectro=None):
	"""Fill randoms catalogue with redshifts.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	option_rand_z : str, optional
		option for randoms redshifts.
	prob_data : array, optional
		probability associated with the data redshift.
		If not provided, the normalized ratio of the total data completeness weight to the BOSS completeness
		(that weights randoms) is taken as probablity.
	mask_data : boolean array, optional
		if provided, mask applied on the data. Redshifts are taken in this selection.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms. Quantiles will be taken in this selection.
	key_redshift : str, optional
		catalogue field to save the redshift values in.
	key_model_z : str, optional
		name of the model_z variable used to select subsamples of data redshifts.
	path_spectro : str, optional
		path to z = f(photo) fit result.
		
	Returns
	-------
	ibin_data : array
		data bins corresponding to each quantile.
	ibin_randoms : array
		randoms bins corresponding to each quantile.

	"""
	logger.info('Option for random redshifts: {}.'.format(option_rand_z))
	
	if prob_data is None: prob_data = data.weight_object
	if mask_data is None: mask_data = data.all_data_cuts(parameters,exclude=['Z'])
	if mask_randoms is None: mask_randoms = randoms.trues()
	
	data['chunk_z'][:] = 0
	randoms['chunk_z'][:] = 0

	if 'window' in option_rand_z:
		params = parameters['option_rand_z']['window']
		survey_subsample = params['survey_subsample']
		if key_model_z and (key_model_z in [params[survey]['key_sort'] for survey in survey_subsample]):
			add_model_z(data,randoms,key_redshift=key_redshift,key_model_z=key_model_z,path_spectro=path_spectro)
		mask_data = mask_data & data.subsample(params.get('data_subsample',{}),txt='window_subsample')
		rng = scipy.random.RandomState(seed=params.get('seed',None))
		list_survey = utils.sorted_alpha_numerical_order(survey_subsample.keys())
		for survey in list_survey:
			mask_data_survey = mask_data & data.subsample(survey_subsample[survey])
			mask_randoms_survey = randoms.subsample(survey_subsample[survey])
			add_redshift_in_window(data,randoms,mask_data=mask_data_survey,mask_randoms=mask_randoms_survey,key_redshift=key_redshift,rng=rng,prob_data=prob_data,**params[survey])
			ichunkz = max(data['chunk_z'].max(),randoms['chunk_z'].max())
			data['chunk_z'][mask_data_survey] = ichunkz + 1
			randoms['chunk_z'][mask_randoms_survey] = ichunkz + 1
	
	elif 'bin' in option_rand_z:
		params = parameters['option_rand_z']['bin']
		survey_subsample = params['survey_subsample']
		if key_model_z in [key for survey in survey_subsample for bins in params[survey] for key in bins]:
			add_model_z(data,randoms,key_redshift=key_redshift,key_model_z=key_model_z,path_spectro=path_spectro)
		mask_data = mask_data & data.subsample(params.get('data_subsample',{}),txt='bin_subsample')
		rng = scipy.random.RandomState(seed=params.get('seed',None))
		list_survey = utils.sorted_alpha_numerical_order(survey_subsample.keys())
		for survey in list_survey:
			mask_data_survey = mask_data & data.subsample(survey_subsample[survey])
			mask_randoms_survey = randoms.subsample(survey_subsample[survey])
			ibin_data,ibin_randoms = add_redshift_in_bin(data,randoms,mask_data=mask_data_survey,mask_randoms=mask_randoms_survey,key_redshift=key_redshift,rng=rng,prob_data=prob_data,bins=params[survey])
			ichunkz = max(data['chunk_z'].max(),randoms['chunk_z'].max())
			data['chunk_z'][mask_data_survey] = ibin_data[mask_data_survey] + ichunkz
			randoms['chunk_z'][mask_randoms_survey] = ibin_randoms[mask_randoms_survey] + ichunkz
	
	elif 'quantile' in option_rand_z:
		params = parameters['option_rand_z']['quantile']
		survey_subsample = params['survey_subsample']
		if key_model_z in [params[survey]['key_quantile'] for survey in survey_subsample]:
			add_model_z(data,randoms,key_redshift=key_redshift,key_model_z=key_model_z,path_spectro=path_spectro)
		mask_data = mask_data & data.subsample(params.get('data_subsample',{}),txt='quantile_subsample')
		rng = scipy.random.RandomState(seed=params.get('seed',None))
		list_survey = utils.sorted_alpha_numerical_order(survey_subsample.keys())
		for survey in list_survey:
			mask_data_survey = mask_data & data.subsample(survey_subsample[survey])
			mask_randoms_survey = randoms.subsample(survey_subsample[survey])
			ibin_data,ibin_randoms = add_redshift_in_quantile(data,randoms,mask_data=mask_data,mask_randoms=mask_randoms,mask_data_survey=mask_data_survey,mask_randoms_survey=mask_randoms_survey,key_redshift=key_redshift,rng=rng,prob_data=prob_data,**params[survey])
			ichunkz = max(data['chunk_z'].max(),randoms['chunk_z'].max())
			data['chunk_z'][mask_data_survey] = ibin_data[mask_data_survey] + ichunkz
			randoms['chunk_z'][mask_randoms_survey] = ibin_randoms[mask_randoms_survey] + ichunkz
				
	
	else:
		if option_rand_z != 'no':
			logger.warning('Correction for photometric dependence of redshifts {} is not available. Random redshifts are drawn from full list_survey.'.format(option_rand_z))
		params = parameters['option_rand_z']['no']
		survey_subsample = params['survey_subsample']
		mask_data = mask_data & data.subsample(params.get('data_subsample',{}),txt='shuffled_subsample')
		rng = scipy.random.RandomState(seed=params.get('seed',None))
		list_survey = utils.sorted_alpha_numerical_order(survey_subsample.keys())
		for survey in list_survey:
			mask_data_survey = mask_data & data.subsample(survey_subsample[survey])
			mask_randoms_survey = randoms.subsample(survey_subsample[survey])
			add_redshift_in_bin(data,randoms,mask_data=mask_data_survey,mask_randoms=mask_randoms_survey,key_redshift=key_redshift,rng=rng,prob_data=prob_data,bins=[{}])
			ichunkz = max(data['chunk_z'].max(),randoms['chunk_z'].max())
			data['chunk_z'][mask_data_survey] = ichunkz + 1
			randoms['chunk_z'][mask_randoms_survey] = ichunkz + 1
	

	if 'shiftz' in option_rand_z:
		list_chunk_z = scipy.unique(randoms['chunk_z'])
		for ichunkz in list_chunk_z:
			logger.info('Processing chunkz {}.'.format(ichunkz))
			mask_data_chunkz = data['chunk_z'] == ichunkz
			mask_randoms_chunkz = randoms['chunk_z'] == ichunkz
			key_model_z_chunkz = '{}_{}'.format(key_model_z,'chunkz')
			path_spectro_chunkz = path_spectro.replace('.json','chunkz{:d}.json'.format(ichunkz))
			add_model_z(data,randoms,mask_data=mask_data_chunkz,mask_randoms=mask_randoms_chunkz,key_redshift=key_redshift,key_model_z=key_model_z_chunkz,path_spectro=path_spectro_chunkz)
			tmp = randoms[key_model_z_chunkz][mask_randoms_chunkz]
			tmp -= scipy.mean(tmp)
			logger.info('Std redshift shift in {}: {:.4g}.'.format(ichunkz,scipy.std(tmp)))
			randoms['Z'][mask_randoms_chunkz] += tmp
	"""
	if 'shiftz' in option_rand_z:
		list_chunk_z = scipy.unique(randoms['chunk_z'])
		x,y = [],[]
		for ichunkz in list_chunk_z:
			logger.info('Processing chunkz {}.'.format(ichunkz))
			mask_data_chunkz = data['chunk_z'] == ichunkz
			x.append(scipy.mean(data[key_model_z][mask_data_chunkz]))
			y.append(scipy.mean(data['Z'][mask_data_chunkz]))
		a,b,r = stats.linregress(x,y)[:3]
		logger.info('Regression (a,b,r) = ({:.4g},{:.4g},{:.4g}).'.format(a,b,r))
		for ichunkz in list_chunk_z:
			logger.info('Processing chunkz {}.'.format(ichunkz))
			mask_data_chunkz = data['chunk_z'] == ichunkz
			mask_randoms_chunkz = randoms['chunk_z'] == ichunkz
			tmp = a*randoms[key_model_z][mask_randoms_chunkz]+b
			tmp -= scipy.mean(tmp)
			logger.info('Std redshift shift in {}: {:.4g}.'.format(ichunkz,scipy.std(tmp)))
			randoms['Z'][mask_randoms_chunkz] += tmp
	"""

def normalize_redshift_density(data,randoms,mask_data=None,mask_randoms=None,mask_randoms_survey=None,key_redshift='Z',key_weight='WEIGHT_SYSTOT',edges=None,nbins=1000):
	"""Normalize random redshift distribution.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.
	mask_randoms_survey : boolean array, optional
		if provided, survey mask applied on the randoms. Quantiles will be taken in mask_randoms & mask_randoms_survey.
	key_redshift : str, optional
		field to the redshift values.
	key_weight : str, optional
		field to the random weight to be updated.
	edges : array, optional
		the redshift edges used to bin data/randoms.
	nbins : int, optional
		the number of bins used to bin data/randoms.

	"""
	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()
	mask_randoms = mask_randoms & mask_randoms_survey
	if edges is None: edges = scipy.linspace(randoms[key_redshift][mask_randoms].min(),randoms[key_redshift][mask_randoms].max()+1e-9,nbins+1)

	values = mask_randoms*randoms.weight_object
	values[~mask_randoms] = 0.
	counts_randoms,edges,ibin_randoms = stats.binned_statistic(randoms[key_redshift][mask_randoms_survey],values=values[mask_randoms_survey],statistic='sum',bins=edges)
	counts_data = stats.binned_statistic(data[key_redshift][mask_data],values=data.weight_object[mask_data],statistic='sum',bins=edges)[0]
	tmp = (counts_data/counts_randoms)[ibin_randoms-1]
	randoms[key_weight][mask_randoms_survey] *= tmp/scipy.mean(tmp)

def normalize_random_weights(data,randoms,mask_data=None,mask_randoms=None):
	"""Normalize random weights.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.

	"""
	if mask_data is None: mask_data = data.all_data_cuts(parameters)
	if mask_randoms is None: mask_randoms = randoms.all_randoms_cuts(parameters)

	weight_data = data['WEIGHT_FKP']*data.weight_object
	weight_randoms = randoms['WEIGHT_FKP']*randoms.weight_object
	alpha_global = weight_data[mask_data].sum()/weight_randoms[mask_randoms].sum()
	list_survey = parameters['cut_surveys']['list']
	for survey in list_survey:
		mask_data_survey = mask_data & data.survey(survey)
		for ichunkz in scipy.unique(data['chunk_z'][mask_data_survey]):
			mask_data_chunk_z = mask_data_survey & (data['chunk_z'] == ichunkz)
			mask_randoms_chunk_z = randoms.survey(survey) & (randoms['chunk_z'] == ichunkz)
			factor = weight_data[mask_data_chunk_z].sum()/weight_randoms[mask_randoms & mask_randoms_chunk_z].sum()/alpha_global
			randoms['WEIGHT_SYSTOT'][mask_randoms_chunk_z] *= factor
			logger.info('Renormalizing WEIGHT_SYSTOT in {} and chunk_z {:d} by {:.4f} such that mean(WEIGHT_SYSTOT) = 1.'.format(survey,ichunkz,factor))

	logger.info('Number of objects after cut: {:d}/{:d}.'.format(mask_randoms.sum(),len(mask_randoms)))
	for key_weight in ['WEIGHT_SYSTOT','WEIGHT_CP','WEIGHT_NOZ']: logger.info('{} range: {:.4f} - {:.4f}.'.format(key_weight,randoms[key_weight][mask_randoms].min(),randoms[key_weight][mask_randoms].max()))

def shuffle_radec(data,randoms,mask_data=None,mask_randoms=None,rng=None,seed=None):
	"""Fill randoms catalogue with data RA/Dec.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.
	rng : RandomState, optional
		numpy RandomState.
	seed : int, optional
		random seed.
	
	"""
	logger.info('Shuffling random RA/Dec.')

	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()
	if rng is None: rng = scipy.random.RandomState(seed=seed)
	
	indices = rng.choice(mask_data.sum(),size=mask_randoms.sum(),replace=True)
	for key in randoms.fields:
		if key in data: randoms[key][mask_randoms] = data[key][mask_data][indices] # all data fields are transferred to the randoms

def pixel_radec(data,randoms,mask_data=None,mask_randoms=None,nest=False,nside=32):
	"""Apply pixelization.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.
	nest : bool, optional
		nested scheme for healpix?
	nside : int, optional
		what nside for healpix?
	
	"""
	logger.info('Option for random RA/Dec: pixel with nside = {:d}.'.format(nside))

	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()

	pixel_data = data.pixelize(key_ra='RA',key_dec='DEC',degree=True,mask=mask_data,**params_healpix)
	pixel_randoms = randoms.pixelize(key_ra='RA',key_dec='DEC',degree=True,mask=mask_randoms,**params_healpix)
	
	weight_data = (data['WEIGHT_FKP']*data.weight_object)[mask_data]
	weight_randoms = (randoms['WEIGHT_FKP']*randoms.weight_object)[mask_randoms]
	
	sum_randoms = utils.digitized_statistics(pixel_randoms,values=weight_randoms)
	sum_data = utils.interp_digitized_statistics(pixel_randoms,pixel_data,values=weight_data,fill=0.)
	
	randoms['WEIGHT_SYSTOT'][mask_randoms] *= sum_data/sum_randoms

def add_random_radec(data,randoms,option_rand_radec='no',mask_data=None,mask_randoms=None):
	"""Fill randoms catalogue with RA/Dec.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	option_rand_radec : str, optional
		option for randoms RA/Dec.
	mask_data : boolean array, optional
		if provided, mask applied on the data.
	mask_randoms : boolean array, optional
		if provided, mask applied on the randoms.
	
	"""
	### Randoms RA/DEC ###
	logger.info('Option for random RA/Dec: {}.'.format(option_rand_radec))
	
	if mask_data is None: mask_data = data.all_data_cuts(parameters)
	if mask_randoms is None: mask_randoms = randoms.all_randoms_cuts(parameters)
	
	list_survey = parameters['cut_surveys']['list']
	if option_rand_radec == 'final':
		for survey in list_survey:
			mask_data_survey = mask_data & data.survey(survey)
			for ichunkz in scipy.unique(data['chunk_z'][mask_data_survey]):
				mask_data_chunk_z = mask_data_survey & (data['chunk_z'] == ichunkz)
				mask_randoms_chunk_z = randoms.survey(survey) & (randoms['chunk_z'] == ichunkz)
				shuffle_radec(data,randoms,mask_data=mask_data_chunk_z,mask_randoms=mask_randoms_chunk_z,seed=parameters['option_rand_radec']['final'].get('seed',None))
	
	elif option_rand_radec == 'pixel':
		for survey in list_survey:
			mask_data_survey = mask_data & data.survey(survey)
			mask_randoms_survey = mask_randoms & randoms.survey(survey)
			for ichunkz in scipy.unique(data['chunk_z'][mask_data_survey]):
				mask_data_chunk_z = mask_data_survey & (data['chunk_z'] == ichunkz)
				mask_randoms_chunk_z = mask_randoms_survey & (randoms['chunk_z'] == ichunkz)
				params = parameters['option_rand_radec']['pixel']
				pixel_radec(data,randoms,mask_data=mask_data_chunk_z,mask_randoms=mask_randoms_chunk_z,nside=params['nside'],nest=params['nest'])

	elif option_rand_radec != 'no':
		logger.warning('Option for random RA/Dec {} is not available. Random RA/Dec are kept uniform on the sky, downsampled by all photometric masks and weighted by COMP_BOSS.'.format(option_rand_radec))


def add_redshift_density(data,randoms,mask_data=None,mask_randoms=None,mask_data_survey=None,mask_randoms_survey=None,key_redshift='Z',key_density='NZ',cosmo=None,edges=None,area=None):
	"""Fill data and randoms density, and return Z, NZ and the area.

	Parameters
	----------
	data : ELGCatalogue
		data catalogue.
	randoms : ELGCatalogue
		randoms catalogue.
	mask_data : boolean array, optional
		if provided, veto mask applied on the data. Density is measured in this selection.
	mask_randoms : boolean array, optional
		if provided, veto mask applied on the randoms. Area is measured in this selection.
	mask_data_survey : boolean array, optional
		if provided, survey mask applied on the data.
	mask_randoms_survey : boolean array, optional
		if provided, survey mask applied on the randoms.
	key_redshift : str, optional
		field to the redshift values.
	key_density : str, optional
		catalogue field to save the density values in.
	cosmo : cosmology
		must have a 'comoving_distance' attribute.
	edges : array
		the bin edges to use to bin redshifts.
	area : float, optional
		the effective area to be used

	Returns
	-------
	meanz : array
		the mean redshift
	nz : array
		the corresponding density
	area : float
		the effective area used

	"""
	if mask_data_survey is None: mask_data_survey = data.trues()
	if mask_randoms_survey is None: mask_randoms_survey = randoms.trues()
	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()
	mask_data = mask_data & mask_data_survey
	mask_randoms = mask_randoms & mask_randoms_survey

	if area is None: area = scipy.sum(randoms['COMP_BOSS'][mask_randoms])/parameters['density']['density_randoms']
	logger.info('Effective area: {:.4g} square degrees'.format(area))
	
	meanz,nz = utils.calc_density(data[key_redshift][mask_data],weight=data.weight_object[mask_data],edges=edges,area=constants.degree**2*area,cosmo=cosmo,extrapolate='linear')
	data[key_density][mask_data_survey] = scipy.interp(data[key_redshift][mask_data_survey],meanz,nz)
	randoms[key_density][mask_randoms_survey] = scipy.interp(randoms[key_redshift][mask_randoms_survey],meanz,nz)
	
	return meanz,nz,area


def condense_catalogue(catalogue,key_weight='WEIGHT'):
	"""Condense catalogue, i.e. keep only RA, DEC, Z and key_weight.
	key_weight is the total weight, completeness weight times WEIGHT_FKP.

	Parameters
	----------
	catalogue : ELGCatalogue
		catalogue
	key_weight : str, optional
		catalogue field to save the total weight in.

	Returns
	-------
	catalogue : ELGCatalogue
		condensed catalogue.

	"""

	catalogue.add_field({'field':key_weight,'description':'Total weight','format':'float64'})
	catalogue[key_weight] = catalogue['WEIGHT_FKP']*catalogue.weight_object
	catalogue.keep_field(*['RA','DEC','Z',key_weight])

	return catalogue


def cut_catalogues(path_data,path_randoms,path_output_data,path_output_randoms=None,condensed=False):
	"""Apply all cuts to data and random catalogues, to obtain clustering catalogues.
	Catalogues are cut into different chunks.

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the random catalogue.
	path_output_data : dict
		paths to the output data catalogue; each key is a chunk.
	path_output_randoms : dict
		paths to the output random catalogue; each key is a chunk.
	condensed : bool, optional
		whether to output condensed catalogues.

	"""
	logger.info(utils.log_header('Cutting catalogues'))
	
	### Cutting data ###
	catalogue = ELGCatalogue.load(path_data)
	list_survey = parameters['cut_surveys']['list']
	
	mask = catalogue.all_data_cuts(parameters)
			
	ndata = {}
	for survey in list_survey:
		logger.info('Cutting survey {} in data.'.format(survey))
		mask_survey = catalogue.survey(survey) & mask
		ndata[survey] = mask_survey.sum()
		if condensed: condense_catalogue(catalogue[mask_survey]).save(path_output_data[survey])
		else: catalogue[mask_survey].save(path_output_data[survey])

	if not path_output_randoms:
		logger.info('Randoms output path not provided.')
		logger.info('Cutting catalogues completed.')
		return	

	### Cutting randoms ###
	catalogue = ELGCatalogue.load(path_randoms)
	
	invalpha = parameters['cut_surveys']['nrand/ndata']
	mask = catalogue.all_randoms_cuts(parameters)
	
	rng = scipy.random.RandomState(seed=parameters['cut_surveys'].get('seed',None))
	for survey in list_survey:
		logger.info('Cutting survey {} in randoms.'.format(survey))
		mask_survey = catalogue.survey(survey) & mask
		nrand = mask_survey.sum()
		nrandsub = ndata[survey]*invalpha
		cut = scipy.ones(nrand,dtype=scipy.bool_)
		if nrandsub < nrand:
			logger.info('Subsampling randoms: 1/alpha = {:.4g}, data/alpha = {:d} < #randoms = {:d}'.format(invalpha,nrandsub,nrand))
			#mask = rng.choice(nrand,size=nrandsub,replace=False)
			mask = rng.uniform(low=0.,high=1.,size=nrand) < nrandsub*1./nrand
			cut[:] = False
			cut[mask] = True
		mask_survey[mask_survey] = cut

		if condensed: condense_catalogue(catalogue[mask_survey]).save(path_output_randoms[survey])
		else: catalogue[mask_survey].save(path_output_randoms[survey])

	logger.info('Cutting catalogues completed.')

def merge_catalogues(list_path_data,list_path_randoms,path_output_data,path_output_randoms,condensed=False):
	"""Merge data and random chunks into caps, making sure alpha = sum(weight_dat)/sum(weight_ran)
	are the same within each chunk.

	Parameters
	----------
	list_path_data : list
		paths to the data chunk catalogues.
	list_path_randoms : list
		paths to the random chunk catalogues.
	path_output_data : str
		path to the output data catalogue.
	path_output_randoms : str
		path to the output random catalogue.
	condensed : bool, optional
		whether to produce condensed catalogues.
	
	"""
	logger.info(utils.log_header('Merging catalogues'))

	assert len(list_path_data) == len(list_path_randoms)
	
	list_sum_data = []
	list_sum_randoms = []

	### Loading data ###
	list_catalogue = []
	for path_data in list_path_data:
		catalogue = ELGCatalogue.load(path_data)
		list_sum_data.append(scipy.sum(catalogue['WEIGHT_FKP']*catalogue.weight_object))
		if condensed: list_catalogue.append(condense_catalogue(catalogue))
		else: list_catalogue.append(catalogue)

	### Merging data ###
	logger.info('Merging data {}.'.format(path_output_data))
	sum(list_catalogue).save(path_output_data)

	### Loading random ###
	list_catalogue = []
	for path_randoms in list_path_randoms:	
		catalogue = ELGCatalogue.load(path_randoms)
		list_sum_randoms.append(scipy.sum(catalogue['WEIGHT_FKP']*catalogue.weight_object))
		list_catalogue.append(catalogue)
	
	list_alpha = scipy.true_divide(list_sum_data,list_sum_randoms)
	alpha_global = scipy.sum(list_sum_data)*1./scipy.sum(list_sum_randoms)
	list_ratio_alpha = list_alpha/alpha_global
	logger.info('Rescaling alphas by {}.'.format(list_ratio_alpha))

	for icat,(catalogue,ratio) in enumerate(zip(list_catalogue,list_ratio_alpha)):
		catalogue['WEIGHT_SYSTOT'] *= ratio
		if condensed: list_catalogue[icat] = condense_catalogue(catalogue)

	### Merging random ###
	logger.info('Merging randoms {}.'.format(path_output_randoms))
	sum(list_catalogue).save(path_output_randoms)
	
	logger.info('Merging catalogues completed.')


def make_photo(path_data,path_randoms,path_output_data,path_output_randoms):
	"""Make photometric catalogues.
	Apply photometric weights to data following parameters['options']['option_photo'].
	Fill unrelevant fields with default values.

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the random catalogue.
	path_output_data : str
		path to the output data catalogue.
	path_output_randoms : str
		path to the output random catalogue.

	"""
	logger.info(utils.log_header('Making photometric catalogues'))

	option_photo = parameters['options']['option_photo']
	keep = ['RA','DEC','WEIGHT_SYSTOT','chunk']

	### Loading data ###	
	catalogue = ELGCatalogue.load(path_data)

	### Preliminary cuts (removes duplicates) ###
	#catalogue['mskbit'][:] = 1
	mask = catalogue.subsample(parameters['target_subsample'],txt='target_subsample')

	### Masking ###
	#mask &= mask_catalogue(catalogue,option_mask)

	### Photometric systematic weights ###
	logger.info('Correction for photometry: {}.'.format(option_photo))
	catalogue['WEIGHT_SYSTOT'][:] = 1.
	if option_photo == 'no': pass
	elif 'hp' in option_photo: photo.add_weight_healpix(catalogue,key_weight_pixel='hpweight_{}'.format(option_photo.replace('hp','')))
	elif 'obj' in option_photo: photo.add_weight_object(catalogue)
	else: logger.warning('Correction for photometry {} is not available. WEIGHT_SYSTOT to 1.'.format(option_photo))

	### Default values for non relevant fields ###
	for weight in ['WEIGHT_CP','WEIGHT_NOZ','WEIGHT_FKP']:
		if weight in catalogue: catalogue[weight][:] = 1.
	
	for weight in ['WEIGHT_SYSTOT','WEIGHT_CP','WEIGHT_NOZ']:
		if weight in catalogue: logger.info('{} range: {:.4f} - {:.4f}'.format(weight,catalogue[weight][mask].min(),catalogue[weight][mask].max()))
	
	### Writing fits ###
	catalogue[mask].save(path_output_data,keep=keep)

	### Randoms ###
	catalogue = ELGCatalogue.load(path_randoms)
	
	### Preliminary cuts ###
	#catalogue['mskbit'][:] = 1
	mask = catalogue.subsample(parameters['randoms_target_subsample'],txt='randoms_target_subsample')

	### Masking ###
	#mask &= mask_catalogue(catalogue,option_mask)

	### Default values for non relevant fields ###
	catalogue['WEIGHT_SYSTOT'][:] = 1.
	for weight in ['WEIGHT_CP','WEIGHT_NOZ','WEIGHT_FKP']:
		if weight in catalogue: catalogue[weight][:] = 1.
	
	for weight in ['WEIGHT_SYSTOT','WEIGHT_CP','WEIGHT_NOZ']:
		if weight in catalogue: logger.info('{} range: {:.4f} - {:.4f}'.format(weight,catalogue[weight][mask].min(),catalogue[weight][mask].max()))

	catalogue[mask].save(path_output_randoms,keep=keep)

	logger.info('Making photo completed.')
	

###############################################################################
# Options-related functions
###############################################################################

def mask_maskbits(catalogue,type_data='data',threshold=0.1):
	"""Compute maskbits, a posteriori reconstruction of anymask.
	Uses parameters['paths']['maskbits_pixels'] and parameters['paths']['maskbits_targets'].

	Parameters
	----------
	catalogue : ELGCatalogue
		data/randoms catalogue.
	type_data : str, optional
		in ['data','randoms'].
	threshold : float
		threshold to remove pixels where maskbits fails to recover anymask.

	Returns
	-------
	mask : boolean array
		mask.

	"""

	data = Catalogue.load(parameters['paths']['maskbits_pixels'])
	
	### Targets flagged by Tractor, not in maskbits ###
	
	mask = data.falses()
	for key in ['grz','xybug','any','t2b','bright']:
		notflag = data['{}_nnotflagok'.format(key)] + data['{}_nnotflagfail'.format(key)]
		tmp = (notflag > 0) & (data['{}_nnotflagfail'.format(key)] > threshold*notflag)
		mask |= tmp
		logger.debug('I will mask {:d} pixel(s) (flagged by Tractor {}, not by maskbits).'.format(tmp.sum(),key))
	mask_pixel = data['hpind'][mask]
	logger.info('I will mask {:d} pixel(s) (flagged by Tractor, not by maskbits): {}.'.format(len(mask_pixel),mask_pixel))
	assert set(mask_pixel) == set([2981667,3464728,3514005,3645255,4546075,4685432,5867869,5933353,6031493,6072514,6080368,6092477,6301369,6408277,6834661,
	2907700,3583785,3587880,4067035,4669088,6007074,6186688,6190785,6199270,6371066,6547876,6551972,6645991,6711673,6735965,6744444,6744445,6748540,6752636,6769023,6773119,6781133])
	
	nside = 1024; nest = False
	pix = catalogue.pixelize(key_ra='RA',key_dec='DEC',degree=True,nside=nside,nest=nest)

	mask = scipy.in1d(pix,mask_pixel)
	toret = ~mask
	if type_data == 'data':
		logger.info('I have masked {:d} (unique: {:d}) objects (flagged by Tractor, not by maskbits).'.format(mask.sum(),(mask & ~catalogue['isdupl']).sum()))
	else:
		logger.info('I have masked {:d} objects (flagged by Tractor, not by maskbits).'.format(mask.sum()))
	
	### Targets not flagged by Tractor, in maskbits ###
	
	if type_data == 'data-v4':
		
		data = Catalogue.load(parameters['paths']['maskbits_targets'])
		mask = data.falses()
		for key,bit in zip(['grz','xybug','any','t2b','bright'],[2,4,8,16,32]):
			tmp = (data['mskbit'] & bit) > 0
			mask |= tmp
			logger.debug('I will mask {:d} decals_uniqid(s) (flagged by maskbits {}, not by Tractor).'.format(tmp.sum(),key))
		mask_id = data['decals_uniqid'][mask]
		logger.info('I will mask {:d} decals_uniqid(s) (flagged by maskbits, not by Tractor).'.format(len(mask_id)))
		#logger.info('I will mask {:d} objects (flagged by maskbits, not by Tractor): {}.'.format(len(mask_id),' '.join(mask_id)))
		mask = scipy.in1d(catalogue['decals_uniqid'],mask_id)
		toret &= ~mask
		logger.info('I have masked {:d} (unique: {:d}) objects (flagged by maskbits, not by Tractor).'.format(mask.sum(),(mask & ~catalogue['isdupl']).sum()))

	return toret

def mask_gaia(catalogue):
	"""Compute Gaia mask around stars.
	Uses parameters['paths']['mask_gaia'].

	Parameters
	----------
	catalogue : ELGCatalogue
		data/random catalogue.

	Returns
	-------
	mask : boolean array
		mask.

	"""
	import pymangle
	
	path = parameters['paths']['mask_gaia']

	logger.info('Loading mangle mask {}.'.format(path))
	mask_mng = pymangle.Mangle(parameters['paths']['mask_gaia'])
	mask = mask_mng.polyid(catalogue['RA'],catalogue['DEC']) != -1
	if 'isdupl' in catalogue.data:
		logger.info('I have masked {:d} (unique: {:d}) objects(s) falling in Gaia mask (area: {:.4g} square degrees).'.format(mask.sum(),(mask & ~catalogue['isdupl']).sum(),mask_mng.area))
	else:
		logger.info('I have masked {:d} objects(s) falling in Gaia mask (area: {:.4g} square degrees).'.format(mask.sum(),mask_mng.area))
	
	return ~mask

def mask_bad_exposures(catalogue):
	"""Compute bad exposure mask, to remove DECam exposures where
	the reflexion of the can of the camera is really strong.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/random catalogue.

	Returns
	-------
	mask : boolean array
		mask.

	"""
	import pymangle
	
	mask = catalogue.falses()
	for band,path in parameters['paths']['mask_bad_exposures'].items():
		logger.info('Loading mangle mask {}.'.format(path))
		mask_mng = pymangle.Mangle(path)
		mask_band = mask_mng.polyid(catalogue['RA'],catalogue['DEC']) != -1
		if 'isdupl' in catalogue.data:
			logger.info('I have masked {:d} (unique: {:d}) objects(s) falling in {}-band bad exposures mask (area: {:.4g} square degrees).'.format(mask_band.sum(),(mask_band & ~catalogue['isdupl']).sum(),band,mask_mng.area))
		else:
			logger.info('I have masked {:d} objects(s) falling in {}-band bad exposures mask (area: {:.4g} square degrees).'.format(mask_band.sum(),band,mask_mng.area))
		mask |= mask_band
	if 'isdupl' in catalogue.data:
		logger.info('I have masked {:d} (unique: {:d}) objects(s) falling in bad exposures mask.'.format(mask.sum(),(mask & ~catalogue['isdupl']).sum()))
	else:
		logger.info('I have masked {:d} objects(s) falling in bad exposures mask.'.format(mask.sum()))
	
	return ~mask


def mask_tdss_fes_separation(catalogue,masking_radius=62/60./60.):
	"""Compute TDSS FES mask.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/random catalogue.
	masking_radius : float
		masking radius.

	Returns
	-------
	mask : boolean array
		mask.

	"""
	mask = catalogue.trues()
	for survey,path in parameters['paths']['targets'].items():
		mask_survey = catalogue.survey(survey)
		coords = SkyCoord(ra=catalogue['RA'][mask_survey],dec=catalogue['DEC'][mask_survey],unit='deg',frame='icrs')
		targets = ELGCatalogue.load(path)
		sourcetype = scipy.unique(targets['SOURCETYPE']).tolist() # tolist to avoid bugs due to trailing spaces...
		list_tdss_fes = [st for st in sourcetype if st.startswith('TDSS_FES')]
		mask_targets = scipy.in1d(targets['SOURCETYPE'],list_tdss_fes)
		logger.info('Masking around {:d} TDSS targets in survey {}.'.format(mask_targets.sum(),survey))
		mask_tmp = mask[mask_survey]
		for ra,dec in zip(targets['RA'][mask_targets],targets['DEC'][mask_targets]):
			mask_tmp &= coords.separation(SkyCoord(ra=ra,dec=dec,unit='deg',frame='icrs')).deg > masking_radius
		mask[mask_survey] = mask_tmp
	
	if 'isdupl' in catalogue.data:
		logger.info('Masking {:d}/{:d} (unique {:d}/{:d}) objects around TDSS FES targets.'.format((~mask).sum(),len(catalogue),(~mask & ~catalogue['isdupl']).sum(),(~catalogue['isdupl']).sum()))
	else:
		logger.info('Masking {:.4f}% of objects around TDSS FES targets.'.format((~mask).sum()*100./len(mask)))

	return mask

def mask_tdss_fes(catalogue):
	"""Compute TDSS FES mask.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/random catalogue.

	Returns
	-------
	mask : boolean array
		mask.

	"""
	import pymangle

	path = parameters['paths']['mask_tdss_fes']

	logger.info('Loading mangle mask {}.'.format(path))
	mask_mng = pymangle.Mangle(path)
	mask = mask_mng.polyid(catalogue['RA'],catalogue['DEC']) != -1

	if 'isdupl' in catalogue.data:
		logger.info('Masking {:d}/{:d} (unique {:d}/{:d}) objects around TDSS FES targets.'.format(mask.sum(),len(catalogue),(mask & ~catalogue['isdupl']).sum(),(~catalogue['isdupl']).sum()))
	else:
		logger.info('Masking {:.4f}% of objects around TDSS FES targets.'.format(mask.sum()*100./len(mask)))

	return ~mask

def mask_centerpost(catalogue):
	"""Compute centerpost mask.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/random catalogue.

	Returns
	-------
	mask : boolean array
		mask.

	"""
	import pymangle

	path = parameters['paths']['mask_centerpost']

	logger.info('Loading mangle mask {}.'.format(path))
	mask_mng = pymangle.Mangle(path)
	mask = mask_mng.polyid(catalogue['RA'],catalogue['DEC']) != -1

	if 'isdupl' in catalogue.data:
		logger.info('Masking {:d}/{:d} (unique {:d}/{:d}) objects in centerposts.'.format(mask.sum(),len(catalogue),(mask & ~catalogue['isdupl']).sum(),(~catalogue['isdupl']).sum()))
	else:
		logger.info('Masking {:.4f}% of objects in centerposts.'.format(mask.sum()*100./len(mask)))

	return ~mask

def mask_nobs(catalogue):
	"""Compute nobs mask.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/random catalogue.

	Returns
	-------
	mask : boolean array
		mask.

	"""

	nside = 1024; nest = False
	pix = catalogue.pixelize(key_ra='RA',key_dec='DEC',degree=True,nside=nside,nest=nest)
	healpix = ELGCatalogue.load(parameters['paths']['mask_nobs'])
	photo = {}
	for field in ['hpnobs_{}'.format(b) for b in ELGCatalogue.LIST_BAND]:
		photo[field] = utils.digitized_interp(pix,healpix['hpind'],healpix[field],fill=-scipy.inf)
	mask = catalogue.trues()
	for survey in ELGCatalogue.LIST_CAP:
		mask_survey = catalogue.survey(survey)
		if survey == 'SGC': threshold = 2
		else: threshold = 1
		mask_tmp = mask[mask_survey]
		for field in photo: mask_tmp &= photo[field][mask_survey] >= threshold
		mask[mask_survey] = mask_tmp
		if 'isdupl' in catalogue.data:
			isdupl = catalogue['isdupl'][mask_survey]
			logger.info('Masking {:d}/{:d} (unique {:d}/{:d}) objects with nobs >= {:d} in {}.'.format((~mask_tmp).sum(),len(mask_tmp),(~mask_tmp & ~isdupl).sum(),(~isdupl).sum(),threshold,survey))
		else:
			logger.info('Masking {:.4f}% of objects with nobs >= {:d} in {}.'.format((~mask_tmp).sum()*100./len(mask_tmp),threshold,survey))

	return mask

def mask_tycho2blob(catalogue):
	"""Compute tycho2blob mask.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/random catalogue.

	Returns
	-------
	mask : boolean array
		mask.

	"""
	import pymangle

	path = parameters['paths']['mask_tycho2blob']

	logger.info('Loading mangle mask {}.'.format(path))
	mask_mng = pymangle.Mangle(path)
	mask = mask_mng.polyid(catalogue['RA'],catalogue['DEC']) != -1
	mask &= (catalogue['mskbit'] & 2**4) == 0

	if 'isdupl' in catalogue.data:
		logger.info('Masking {:d}/{:d} (unique {:d}/{:d}) objects in tycho2blob.'.format(mask.sum(),len(catalogue),(mask & ~catalogue['isdupl']).sum(),(~catalogue['isdupl']).sum()))
	else:
		logger.info('Masking {:.4f}% of objects in tycho2blob.'.format(mask.sum()*100./len(mask)))

	return ~mask

def mask_sky_lines(catalogue,option_mask):
	"""Compute sky line masks.
	Credit to Johan Comparat.

	Parameters
	----------
	catalogue : ELGCatalogue
		data catalogue.
	option_mask : list
		list of lines to mask: O2a, O2b, O3b, Hb.
	type_data : str
		in ['data','randoms'].

	Returns
	-------
	mask : boolean array
		mask.

	"""
	logger.info('Masking sky lines.')

	selection = parameters['option_mask']['skylines']
	redshift_rounded = (catalogue['Z']*10000).astype(scipy.int64)
	mask = catalogue.trues()
	for line in option_mask:
		mask_sky_lines = (scipy.loadtxt(path_mask_sky_lines(line,selection[line]['width']))*10000).astype(scipy.int64)
		mask &= scipy.in1d(redshift_rounded,mask_sky_lines)
	logger.info('Sky line masking removes {:.4g}% of data.'.format(mask.sum()*100./len(catalogue)))

	return ~mask


def mask_catalogue(catalogue,option_mask,type_data='data'):
	"""Compute all masks, following option_mask.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/randoms catalogue.
	option_mask : list
		list of masking options.
	type_data : str
		in ['data','randoms'].

	Returns
	-------
	mask : boolean array
		mask.

	"""
	logger.info('Masking: {}.'.format(option_mask))
	mask = catalogue.trues()
	if not isinstance(option_mask,list): option_mask = [option_mask]
	
	for option in option_mask:
		
		if option == 'no':
			continue
		
		if option == 'maskbits':
			mask &= catalogue['mskbit'] == 1
			mask &= mask_maskbits(catalogue,type_data)
		
		elif option == 'gaia':
			mask &= mask_gaia(catalogue)
		
		elif option == 'badexp':
			mask &= mask_bad_exposures(catalogue)
		
		elif 'tycho2blob' in option:
			mask &= mask_tycho2blob(catalogue)
			
		elif option == 'hasfiber2':
			catalogue['hasfiber'][catalogue['hasfiber']==2] = 1
			
		elif option == 'hasfiber3':
			catalogue['hasfiber'][catalogue['hasfiber']==3] = 1
		
		elif option == 'tdss-fes':
			 mask &= mask_tdss_fes(catalogue)

		elif 'field' in option:
			circles = parameters['option_mask'][option]
			for circle in circles:
				center = SkyCoord(ra=circle['RA'],dec=circle['DEC'],unit='deg',frame='icrs')
				coords = SkyCoord(ra=catalogue['RA'],dec=catalogue['DEC'],unit='deg',frame='icrs')
				mask &= coords.separation(center).deg > circle['radius']
			
		elif option in parameters['option_mask']:
			selection = parameters['option_mask'][option]
			for mode in selection:
				if selection[mode]:
					mask_ = catalogue.subsample(selection[mode],txt='{} {}'.format(mode,name))
					if mode == 'outs': mask_=~mask_
				mask &= mask_
		
		elif 'sky' in option: #option should be of the form sky-O2a-O2b
			lines = options.replace('sky','').split('-')
			mask &= mask_sky_lines(catalogue,lines)
		else:
			logger.warning('Masking {} is not available. No masking applied.'.format(option))

	return mask


def add_des(catalogue):

	des = Catalogue.load(parameters['paths']['des'])
	index_des,sep2d = utils.match_ra_dec([catalogue['RA'],catalogue['DEC']],[des['RA'],des['DEC']],return_sep2d=True,return_sep3d=False,distance_upper_bound=1./3600.,nn=1)
	mask = ~ scipy.isinf(sep2d)
	logger.info('No match found with DES for {:d}/{:d} objects.'.format((~mask).sum(),len(mask)))
	
	des['g'] = des['mag_auto_g_dered']
	des['gr'] = des['mag_auto_g_dered'] - des['mag_auto_r_dered']
	des['rz'] = des['mag_auto_r_dered'] - des['mag_auto_z_dered']
	for b in ['g','gr','rz']:
		field = 'd{}'.format(b)
		catalogue[field][mask] = catalogue[b][mask] - des[b][index_des[mask]]
		catalogue[field][~mask] = 0.

def add_weight_collision_group(catalogue,key_weight='WEIGHT_CP',mask=None):
	"""Add fiber collision weights, based on collision groups.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	key weight : str, optional
		catalogue field to save fiber collisions weight in.
	mask : boolean array, optional
		optional mask.
	"""
	if mask is None: mask = catalogue.trues()
	mask_target = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample')
	mask_hasfiber = mask_target & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')

	catalogue['ELG_INGROUP'] = catalogue.ELG_INGROUP
	# Correction
	catalogue[key_weight],catalogue['MASK_CP'] = utils.fiber_collision_group(catalogue['ELG_INGROUP'],mask_hasfiber,mask_target=mask_target,return_imatch=True)
	catalogue['MASK_CP'] = mask_target & (catalogue['MASK_CP'] >= 1.)
	logger.info('Mean {} of valid fibers: {:.4f}.'.format(key_weight,scipy.mean(catalogue[key_weight][mask_hasfiber])))
	

def add_weight_noz_fiberid(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ'):
	"""Add redshift failures weights.
	It calculates a fiberid dependent SSR weight WEIGHT_NOZ.
	WEIGHT_NOZ is given by 1./fiberid_SSR.
	fiberid_SSR is calculated using fidedges bins in the different survey_subsample.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	mask_spectro_subsample : boolean array
		mask corresponding to the spectroscopic subsample (e.g. SPECTYPE != STAR).
	mask_reliable_redshift : boolean array
		mask corresponding to reliable redshifts (e.g. Z_ok).
	key weight : str, optional
		catalogue field to save redshift failures weights in.

	"""
	fidedges = parameters['option_noz']['fiberid'].get('fidedges',1000)
	survey_subsample = parameters['option_noz']['fiberid'].get('survey_subsample',{survey:{'survey':survey} for survey in ELGCatalogue.LIST_CHUNK})

	for survey in survey_subsample:

		mask_survey = mask_spectro_subsample & catalogue.subsample(survey_subsample[survey])
		counts,edges,binnumber = stats.binned_statistic(catalogue['FIBERID'][mask_survey],mask_reliable_redshift[mask_survey],statistic='count',bins=fidedges)
		reliable = stats.binned_statistic(catalogue['FIBERID'][mask_survey],mask_reliable_redshift[mask_survey],statistic='sum',bins=edges)[0]
		assert ((binnumber.min() == 1) and (binnumber.max() == len(edges)-1))
		fiberssr = reliable[binnumber-1]/counts[binnumber-1]
		catalogue[key_weight][mask_survey] = 1./fiberssr

def add_weight_noz_platefiberid(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ'):
	"""Add redshift failures weights.
	It calculates a plate and fiberid dependent SSR weight WEIGHT_NOZ.
	WEIGHT_NOZ is given by 1./plate_SSR/fiberid_SSR.
	fiberid_SSR is calculated using fidedges bins in the different survey_subsample.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	mask_spectro_subsample : boolean array
		mask corresponding to the spectroscopic subsample (e.g. SPECTYPE != STAR).
	mask_reliable_redshift : boolean array
		mask corresponding to reliable redshifts (e.g. Z_ok).
	key weight : str, optional
		catalogue field to save redshift failures weights in.

	"""
	fidedges = parameters['option_noz']['platefiberid'].get('fidedges',1000)
	survey_subsample = parameters['option_noz']['platefiberid'].get('survey_subsample',{survey:{'survey':survey} for survey in ELGCatalogue.LIST_CHUNK})

	for survey in survey_subsample:

		mask_survey = mask_spectro_subsample & catalogue.subsample(survey_subsample[survey])
		counts,edges,binnumber = stats.binned_statistic(catalogue['FIBERID'][mask_survey],mask_reliable_redshift[mask_survey],statistic='count',bins=fidedges)
		weight = 1./catalogue['plate_MJD_SSR'][mask_survey]
		weight[utils.isnaninf(weight)] = 0.
		reliable = stats.binned_statistic(catalogue['FIBERID'][mask_survey],mask_reliable_redshift[mask_survey]*weight,statistic='sum',bins=edges)[0]
		assert ((binnumber.min() == 1) and (binnumber.max() == len(edges)-1))
		fiberssr = reliable[binnumber-1]/counts[binnumber-1]
		catalogue[key_weight][mask_survey] = weight/fiberssr


def add_weight_noz_platexyfocal(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ'):
	"""(Old) Add redshift failures weights.
	It calculates a XYFOCAL dependent SSR weight WEIGHT_NOZ. Two steps: 
		1) The SSR is calculated in XYFOCAL bins (their number being given by nxyedges).
		A subsample of plates can be chosen with good_plate to perform this operation. 
		2) The weight WEIGHT_NOZ is given by the inverse SSR binned in XYFOCAL times the inverse plate_SSR.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	mask_spectro_subsample : boolean array
		mask corresponding to the spectroscopic subsample (e.g. SPECTYPE != STAR).
	mask_reliable_redshift : boolean array
		mask corresponding to reliable redshifts (e.g. Z_ok).
	key weight : str, optional
		catalogue field to save redshift failures weights in.

	"""
	option_good_plate = parameters['option_noz']['platexyfocal'].get('good_plate',{})
	xyedges = parameters['option_noz']['platexyfocal'].get('xyedges',15)
	survey_subsample = parameters['option_noz']['platexyfocal'].get('survey_subsample',{survey:{'survey':survey} for survey in ELGCatalogue.LIST_CHUNK})

	mask_good_plate = mask_spectro_subsample.copy()
	mask_good_plate &= catalogue.subsample(option_good_plate,txt='good_plate')
	catalogue.fill_default_value(key_weight)

	for survey in survey_subsample:
		mask_survey = mask_spectro_subsample & catalogue.subsample(survey_subsample[survey])
		counts,xedges,yedges,binnumber = stats.binned_statistic_2d(catalogue['XFOCAL'][mask_survey],catalogue['YFOCAL'][mask_survey],values=mask_good_plate[mask_survey],statistic='sum',bins=xyedges,expand_binnumbers=True)
		weight = 1./catalogue['plate_MJD_SSR'][mask_survey]
		weight[utils.isnaninf(weight)] = 0.
		reliable = stats.binned_statistic_2d(catalogue['XFOCAL'][mask_survey],catalogue['YFOCAL'][mask_survey],values=(mask_good_plate & mask_reliable_redshift)[mask_survey]*weight,statistic='sum',bins=[xedges,yedges])[0]
		ssr = (counts*1./reliable)[tuple(binnumber-1)]
		catalogue[key_weight][mask_survey] = weight/ssr

def add_weight_noz_nearestneighbor(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ'):
	"""Add redshift failures weights.
	Each redshift failure is corrected for by adding its weight to its nearest neighbor with reliable redshift.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	mask_spectro_subsample : boolean array
		mask corresponding to the spectroscopic subsample (e.g. SPECTYPE != STAR).
	mask_reliable_redshift : boolean array
		mask corresponding to reliable redshifts (e.g. Z_ok).
	key weight : str, optional
		catalogue field to save redshift failures weights in.

	"""
	mask_unreliable_redshift = mask_spectro_subsample & (~mask_reliable_redshift)
	index_nn_reliable_redshift = utils.match_ra_dec([catalogue['RA'][mask_unreliable_redshift],catalogue['DEC'][mask_unreliable_redshift]],[catalogue['RA'][mask_reliable_redshift],catalogue['DEC'][mask_reliable_redshift]],nn=1)
	weight_noz_reliable_redshift = scipy.ones((mask_reliable_redshift.sum()),dtype=scipy.float64)
	scipy.add.at(weight_noz_reliable_redshift,index_nn_reliable_redshift,catalogue['WEIGHT_CP'][mask_unreliable_redshift])
	catalogue[key_weight][mask_reliable_redshift] = weight_noz_reliable_redshift
	logger.info('Resolved redshifts = {:.2f}, all redshifts = {:.2f}.'.format(scipy.sum((catalogue['WEIGHT_NOZ']+catalogue['WEIGHT_CP']-1.)[mask_reliable_redshift]),scipy.sum(catalogue['WEIGHT_CP'][mask_spectro_subsample])))
	catalogue[key_weight] = (catalogue['WEIGHT_CP']+catalogue['WEIGHT_NOZ']-1.)/catalogue['WEIGHT_CP'] #final weighting is WEIGHT_SYSTOT*WEIGHT_NOZ*WEIGHT_CP

def add_weight_noz_fitplatesnfiberid(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ',key_plate_ssr='zfail_platesn',path_plate_ssr=None):
	"""Add redshift failures weights.
	It calculates a plate and fiberid dependent SSR weight WEIGHT_NOZ.
	WEIGHT_NOZ is given by 1./zfail_platesn/fiberid_SSR.
	zfail_platesn results from a fit zfail_platesn = f(plate_SN_MEDIAN_ALL).
	fiberid_SSR is calculated using fidedges bins in the different survey_subsample.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	mask_spectro_subsample : boolean array
		mask corresponding to the spectroscopic subsample (e.g. SPECTYPE != STAR).
	mask_reliable_redshift : boolean array
		mask corresponding to reliable redshifts (e.g. Z_ok).
	key weight : str, optional
		catalogue field to save redshift failures weights in.
	key_plate_ssr : str, optional
		catalogue field to save SSR = f(plate_SN) relation in.
	path_plate_ssr : str, optional
		path to SSR = f(plate_SN) fit result.

	"""
	fidedges = parameters['option_noz']['fitplatesnfiberid'].get('fidedges',1000)
	survey_subsample = parameters['option_noz']['fitplatesnfiberid'].get('survey_subsample',{survey:{'survey':survey} for survey in ELGCatalogue.LIST_CHUNK})

	results = spectro._fit_plate_ssr_(catalogue)
	if path_plate_ssr is not None: utils.save(path_plate_ssr,results)
	spectro._add_model_plate_ssr_(catalogue,results,key_model_ssr=key_plate_ssr,normalize=False)
	catalogue.fill_default_value(key_weight)
	
	for survey in survey_subsample:

		mask_survey = mask_spectro_subsample & catalogue.subsample(survey_subsample[survey])
		counts,edges,binnumber = stats.binned_statistic(catalogue['FIBERID'][mask_survey],mask_reliable_redshift[mask_survey],statistic='count',bins=fidedges)
		weight = 1./catalogue[key_plate_ssr][mask_survey]
		weight[utils.isnaninf(weight)] = 0.
		reliable = stats.binned_statistic(catalogue['FIBERID'][mask_survey],mask_reliable_redshift[mask_survey]*weight,statistic='sum',bins=edges)[0]
		assert ((binnumber.min() == 1) and (binnumber.max() == len(edges)-1))
		fiberssr = reliable[binnumber-1]/counts[binnumber-1]
		catalogue[key_weight][mask_survey] = weight/fiberssr


def add_weight_noz_fitplatesnfitfiberid(catalogue,key_weight='WEIGHT_NOZ',key_plate_ssr='zfail_platesn',key_fiberid_ssr='zfail_fiberid',path_plate_ssr=None,path_fiberid_ssr=None):
	"""Add redshift failures weights.
	It calculates a plate and fiberid dependent SSR weight WEIGHT_NOZ.
	WEIGHT_NOZ is given by 1./zfail_platesn/zfail_fiberid.
	zfail_platesn and zfail_fiberid result from a fit zfail_platesn = f(plate_SN_MEDIAN_ALL)
	and zfail_fiberid = f(FIBERID) respectively.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	key weight : str, optional
		catalogue field to save redshift failures weights in.
	key_plate_ssr : str, optional
		catalogue field to save SSR = f(plate_SN) relation in.
	key_fiberid_ssr : str, optional
		catalogue field to save SSR = f(FIBERID) relation in.
	path_plate_ssr : str, optional
		path to SSR = f(plate_SN) fit result.
	path_fiberid_ssr : str, optional
		path to SSR = f(FIBERID) fit result.

	"""
	logger.info('Applying fitplatesnfitfiberid correction for redshift failures.')

	results = spectro._fit_plate_ssr_(catalogue)
	if path_plate_ssr is not None: utils.save(path_plate_ssr,results)
	spectro._add_model_plate_ssr_(catalogue,results,key_model_ssr=key_plate_ssr,normalize=False)
	
	results = spectro._fit_fiberid_ssr_(catalogue,weight=1./catalogue[key_plate_ssr])
	if path_fiberid_ssr is not None: utils.save(path_fiberid_ssr,results)
	spectro._add_model_fiberid_ssr_(catalogue,results,key_model_ssr=key_fiberid_ssr,normalize=False)
	
	catalogue[key_weight] = 1./(catalogue[key_plate_ssr]*catalogue[key_fiberid_ssr])

def add_weight_noz_fitplatesnfitxyfocal(catalogue,key_weight='WEIGHT_NOZ',key_plate_ssr='zfail_platesn',key_xyfocal_ssr='zfail_xyfocal',path_plate_ssr=None,path_xyfocal_ssr=None):
	"""Add redshift failures weights.
	It calculates a plate and XYFOCAL dependent SSR weight WEIGHT_NOZ.
	WEIGHT_NOZ is given by 1./zfail_platesn/zfail_xyfocal.
	zfail_platesn and zfail_xyfocal result from a fit zfail_platesn = f(plate_SN_MEDIAN_ALL)
	and zfail_xyfocal = f(XYFOCAL) respectively.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	key weight : str, optional
		catalogue field to save redshift failures weights in.
	key_plate_ssr : str, optional
		catalogue field to save SSR = f(plate_SN) relation in.
	key_xyfocal_ssr : str, optional
		catalogue field to save SSR = f(XYFOCAL) relation in.
	path_plate_ssr : str, optional
		path to SSR = f(plate_SN) fit result.
	path_xyfocal_ssr : str, optional
		path to SSR = f(XYFOCAL) fit result.

	"""
	logger.info('Applying fitplatesnfitxyfocal correction for redshift failures.')

	results = spectro._fit_plate_ssr_(catalogue)
	if path_plate_ssr is not None: utils.save(path_plate_ssr,results)
	spectro._add_model_plate_ssr_(catalogue,results,key_model_ssr=key_plate_ssr,normalize=False)
	
	results = spectro._fit_xyfocal_ssr_(catalogue,weight=1./catalogue[key_plate_ssr])
	if path_xyfocal_ssr is not None: utils.save(path_xyfocal_ssr,results)
	spectro._add_model_xyfocal_ssr_(catalogue,results,key_model_ssr=key_xyfocal_ssr,normalize=False)
	
	catalogue[key_weight] = 1./(catalogue[key_plate_ssr]*catalogue[key_xyfocal_ssr])

def add_weight_noz_magcorrection(catalogue,mask_spectro_subsample,mask_reliable_redshift,key_weight='WEIGHT_NOZ'):
	"""(Not used) Updating redshift failures weights to correct for the dependence of redshift efficiency
	with the magnitude in g-band.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	mask_spectro_subsample : boolean array
		mask corresponding to the spectroscopic subsample (e.g. SPECTYPE != STAR).
	mask_reliable_redshift : boolean array
		mask corresponding to reliable redshifts (e.g. Z_ok).
	key weight : str, optional
		field to redshift failures weight.

	"""
	key,bins = 'g',30
	hasfiber,bins,binnumber = stats.binned_statistic(catalogue[key][mask_spectro_subsample],values=catalogue[key][mask_spectro_subsample],statistic='count',bins=bins)
	reliable = stats.binned_statistic(catalogue[key][mask_spectro_subsample & mask_reliable_redshift],values=catalogue[key][mask_spectro_subsample & mask_reliable_redshift],statistic='count',bins=bins)[0]
	ratio = hasfiber*1./reliable
	ratio[reliable<10] = 1.
	catalogue[key_weight][mask_spectro_subsample] *= ratio[binnumber-1]
	logger.info('Updating WEIGHT_NOZ according to {}-mag; min={:.4f}, max={:.4f}.'.format(key,ratio.min(),ratio.max()))


def save_mask_sky_lines(width=1.):
	"""Save mask for sky lines.
	Credit to Johan Comparat.

	Parameters
	----------
	width : float
		width to take around the sky lines.
	mask_spectro_subsample : boolean array
		mask corresponding to the spectroscopic subsample (e.g. SPECTYPE != STAR).
	mask_reliable_redshift : boolean array
		mask corresponding to reliable redshifts (e.g. Z_ok).

	"""
	path_sky_lines = os.path.join(os.getenv('ELGCAT'),'skylines','dr12-sky-mask.txt')
	
	lambda_mask = scipy.loadtxt(path_sky_lines,unpack=True)
	z_array = scipy.arange(0.6,1.2,0.0001)
	line = {}
	line['O2a'] = 3726.032 * (1+z_array)
	line['O2b'] = 3728.814 * (1+z_array)
	line['O3b'] = 5006.841 * (1+z_array)
	line['Hb'] = 4861.331 * (1+z_array)

	def get_mask(wl):
		ratio = scipy.amin(scipy.absolute(10000.*scipy.log10(scipy.outer(wl,1./lambda_mask))),axis=1)
		return ratio <= width

	for key in line:
		line[key] = get_mask(line[key])
		path = path_mask_sky_lines(key,width)
		logger.info('Saving {} line mask to: {}.'.format(key,path))
		scipy.savetxt(path,scipy.transpose([z_array[line[key]]]),fmt=str('%1.4f'))


def geometric_area(chunk,nbar=1e4):
	"""Compute geometric area based on sectors.
	It generates uniform randoms at a given angular density,
	then subsamples them according to the geometry masks.
	Credit to Anand Raichoor.

	Parameters
	----------
	chunk : str
		the chunk to consider, in ['eboss21','eboss22','eboss23','eboss25'].
	nbar : float
		the angular density / deg^2.

	Returns
	-------
	area : float
		estimation of the area.
	error : float
		poisson error on this estimation.

	"""
	import pymangle
	# creating randoms in the footprint
	# rectangle containing the footprint#
	if chunk == 'eboss21': ramin=315. ; ramax=360. ; decmin=-2.; decmax=2.
	if chunk == 'eboss22': ramin=0.   ; ramax=45.  ; decmin=-5.; decmax=5.
	if chunk == 'eboss23': ramin=126. ; ramax=157. ; decmin=13.; decmax=29.
	if chunk == 'eboss25': ramin=131. ; ramax=166. ; decmin=23.; decmax=33.
	# first generating randoms within ramin<ra<ramax, -90<dec<90, then cutting on decmin<dec<decmax
	# first estimating the area for ramin<ra<ramax, -90<dec<90
	area = 4*constants.pi*(ramax-ramin)/360./constants.degree**2
	# number of randoms
	nrand  = int(scipy.rint(nbar*area))
	# creating randoms
	ra = scipy.random.uniform(low=ramin,high=ramax,size=nrand)
	dec = scipy.arcsin(1.-scipy.random.uniform(low=0,high=1,size=nrand)*2.)/constants.degree
	# cutting on decmin<dec<decmax
	mask = (dec>=decmin) & (dec<=decmax)
	ra = ra[mask]
	dec = dec[mask] 
	# cutting on the footprint
	mask_mng = pymangle.Mangle(path_geometry(chunk))
	tmp = mask_mng.polyid(ra,dec) != -1
	remain = tmp.sum()
	return remain/dens, scipy.sqrt(remain)/dens
