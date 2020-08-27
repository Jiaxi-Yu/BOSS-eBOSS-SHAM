import copy
import logging
import scipy
from scipy import constants,stats,optimize
import utils
from catalogueELG import ELGCatalogue
from photometric_correction import BaseFit

logger = logging.getLogger('Spectroscopy')

def set_parameters(params):
	global parameters
	parameters = params

class FitSpectroscopy(BaseFit):
	"""Class to fit spectroscopy, i.e. to regress redshift
	against photometric parameters (e.g. depth).
	Covariance matrix is taken to be data completeness weights.

	Parameters
	----------
	params : dict
		parameter dictionary.

	"""
	logger = logging.getLogger('FitSpectroscopy')
		
	def chi2(self,*args,**kwargs):
		delta = self.z-self.model(*args,**kwargs)
		return scipy.sum(delta**2*self.invcovariance)
		
	def model(self,*coeffs):
		return self._model_(self.photo,coeffs,ibin=0)

	def _model_(self,photo,coeffs,ibin=None):
		coeffs = scipy.atleast_2d(coeffs)
		epsilon = coeffs[...,0]
		photo_ = photo
		linear = self.indices_linear[ibin]
		if linear[1] != self.indices_linear[None][1]: photo_ = photo[:,linear[1]]
		toret = epsilon + (coeffs[:,linear[0]]*photo_).sum(axis=1)
		if self.nquadratic:
			quadratic = self.indices_quadratic[ibin]
			toret += (coeffs[:,quadratic[0]]*photo[:,quadratic[1]]*photo[:,quadratic[2]]).sum(axis=1)
		return toret
	
	def get_photo(self,data,mask=None):
		if mask is None: mask = data.trues()
		photo = []
		for key in self.params['params']:
			tmp = data[key][mask]
			if 'depth' in key: tmp = utils.depth_to_flux(tmp)
			photo += [tmp]
		return scipy.array(photo).T

	def set_data(self,data,key_redshift=None,mask=None):

		if mask is None: mask = data.trues()
		subsample = self.params.get('data_subsample',{})
		mask = mask & data.subsample(subsample,txt='subsample')

		if key_redshift is None: key_redshift = self.params.get('key_redshift','Z')
		self.params['key_redshift'] = key_redshift
		
		for key in self.params['params']: mask &= data.good_value(key)
		self.nobs = mask.sum()
		
		weight = data.weight_object[mask]
		if utils.isnaninf(weight).any():
			self.logger.warning('For some reason, Weight has incorrect ({}) values. Taking Weight = 1.'.format(ELGCatalogue.bad_values('WEIGHT_SYSTOT')))
			weight[:] = 1.
		
		self.z = data[key_redshift][mask]
		self.invcovariance = weight
		self.photo = self.get_photo(data,mask=mask)

	def predicted_redshift(self,data,mask=None):
		photo = self.get_photo(data,mask=mask)
		return self._model_(photo,[self.bestfit['values'][key] for key in self.parameters]) # to save memory... why?
	
	def getstate(self):
		state = {}
		for key in ['params','null','bestfit']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

def _fit_redshift_(catalogue,key_redshift='Z',mask=None):
	
	if mask is None: mask = catalogue.trues()
	mask_data = mask & catalogue.all_data_cuts(parameters,exclude=[key_redshift])
	
	survey_subsample = parameters['fit_spectro']['survey_subsample']
	results = {key:{} for key in survey_subsample}

	for survey in results:
		fit = FitSpectroscopy(params=parameters['fit_spectro'])
		mask_survey = mask_data & catalogue.subsample(survey_subsample[survey])
		fit.set_data(catalogue,key_redshift=key_redshift,mask=mask_survey)
		if not mask_survey.any():
			logger.info('Skipping survey {}.'.format(survey))
		else:
			logger.info('Fitting survey {}.'.format(survey))
			fit.minimize()
		results[survey] = fit.getstate()
	
	return results

def fit_redshift(path_data):
	"""Fit the z = f(depth) relation.
	The fit result is saved in parameters['paths']['fit_spectro'].

	Parameters
	----------
	path_data : str
		path to data.
	key_redshift : str, optional
		the catalogue field where redshifts are stored.

	"""
	logger.info(utils.log_header('Fitting redshift'))

	path_fit = parameters['paths']['fit_spectro']

	catalogue = ELGCatalogue.load(path_data)
	results = _fit_redshift_(catalogue)
		
	utils.save(path_fit,results)

def _add_model_z_(catalogue,results,key_model_z='comb_galdepth',mask=None):
	
	#catalogue.fill_default_value(key_model_z)
	catalogue[key_model_z] = catalogue.nans()
	if mask is None: mask = catalogue.trues()

	for survey in results:
		fit = FitSpectroscopy.setstate(results[survey])
		mask_survey = mask & catalogue.subsample(fit.params['survey_subsample'][survey])
		if not mask_survey.any():
			logger.info('Skipping survey {}.'.format(survey))
			continue
		catalogue[key_model_z][mask_survey] = fit.predicted_redshift(catalogue,mask=mask_survey)

	mask_bad = mask & catalogue.bad_value(key_model_z)
	sum_bad = mask_bad.sum()
	if sum_bad>0:
		logger.warning('{} has {:d} incorrect values: {}.'.format(key_model_z,sum_bad,catalogue[key_model_z][mask_bad]))
	model_z = catalogue[key_model_z][mask]
	logger.info('{} range: {:.4f} - {:.4f}.'.format(key_model_z,model_z.min(),model_z.max()))

def add_model_z(catalogue,key_model_z='comb_galdepth'):
	"""Add comb_galdepth to catalogue, as predicted by the fitted z = f(depth) relation 
	in parameters['paths']['fit_spectro'].

	Parameters
	----------
	catalogue : ELGCatalogue
		catalogue.
	key_model_z :
		field of catalogue to save model_z in.

	"""
	results = utils.load(parameters['paths']['fit_spectro'])
	_add_model_z_(catalogue,results,key_model_z=key_model_z)

def get_plate_mjd_stats(catalogue,mask=None,key_plate='plate_MJD',key_sn='SN_MEDIAN_ALL',uniques=False,return_counts=False):
	"""Add the median of SN_MEDIAN_ALL and the Spectroscopic Success Rate (SSR) in each plate_MJD/spectro_MJD.

	Parameters
	----------
	catalogue : ELGCatalogue
		data catalogue.
	mask : boolean array, optional
		veto mask to be applied before any calculation.
	key_plate : str, optional
		field to plate.
	key_sn : str, optional
		field to SN.
	uniques : bool, optional
		whether to return unique plates only (within mask & spectro_subsample).
	return_counts : bool, optional 
		if uniques, whether to return counts per plate (including mask & spectro_subsample).
	
	Returns
	-------
	plates : array
		the plates.
	sn : array
		the SN binned per plate.
	ssr : array
		the SSR binned per plate.
	counts : array
		number of objects per plate, if uniques and return_counts.

	"""

	if mask is None: mask = catalogue.trues()
	mask_spectro_subsample = mask.copy()
	mask_spectro_subsample &= catalogue.subsample(parameters['target_subsample'],txt='target_subsample')
	mask_spectro_subsample &= catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')
	mask_spectro_subsample &= catalogue.subsample(parameters['spectro_subsample'],txt='spectro_subsample')
	mask_reliable_redshift = mask_spectro_subsample.copy()
	mask_reliable_redshift &= catalogue.subsample(parameters['reliable_redshift'],txt='reliable_redshift')

	if key_plate == 'plate_MJD': uniqid = catalogue.plate_mjd
	elif key_plate == 'spectro_MJD': uniqid = catalogue.spectro_mjd
	else: uniqid = catalogue[key_plate]
	if 'SPECSN2' in key_sn:
		sn = catalogue.spectro_sn(b=key_sn.replace('SPECSN2_','').upper())
	else:
		sn = catalogue[key_sn]
	sn = utils.interp_digitized_statistics(uniqid,uniqid[mask_spectro_subsample],fill=scipy.nan,values=sn[mask_spectro_subsample],statistic='median')
	ssr = utils.digitized_statistics(uniqid,values=mask_reliable_redshift)*1./utils.digitized_statistics(uniqid,values=mask_spectro_subsample)

	toret = [uniqid,sn,ssr]
	if uniques:
		_,index,inverse,counts = scipy.unique(uniqid[mask_spectro_subsample],return_index=True,return_inverse=True,return_counts=True)
		if uniques: toret = [t[mask_spectro_subsample][index] for t in toret]
		if return_counts: toret.append(counts)

	return toret

class ModelPower(object):
	"""
	scipy.optimize.minimize or curve_fit used instead of minuit for historical reasons.
	"""

	logger = logging.getLogger('ModelPower')

	def model(self,x,*coeffs):
		#return coeffs[0] - scipy.absolute(coeffs[1]*(x-coeffs[2]))**coeffs[3]
		return coeffs[0] - coeffs[1]*scipy.absolute(x-coeffs[2])**coeffs[3]
	
	def jac_model(self,x,*coeffs):
		"""
		c0 = scipy.ones_like(x,dtype='f8')
		diff = coeffs[1]*(x-coeffs[2])
		diffabs = scipy.absolute(diff)
		tmp = diff*diffabs**(coeffs[3]-2.)
		c1 = -coeffs[3]*(x-coeffs[2])*tmp
		c2 = coeffs[3]*coeffs[1]*tmp
		c3 = -scipy.log(diffabs)*diffabs**coeffs[3]
		"""
		c0 = scipy.ones_like(x,dtype='f8')
		diff = x-coeffs[2]
		diffabs = scipy.absolute(diff)
		c1 = -diffabs**coeffs[3]
		c2 = coeffs[1]*coeffs[3]*diff*diffabs**(coeffs[3]-2.)
		c3 = -coeffs[1]*scipy.log(diffabs)*diffabs**coeffs[3]
		return scipy.transpose([c0,c1,c2,c3])

	def lnpoisson(self,coeffs,x,y):
		model = self.counts*self.model(x,*coeffs)
		if scipy.any(model<=0.): return 1e10
		ngood = self.counts*y
		return scipy.sum(-ngood*scipy.log(model)+model)
	
	def jac_lnpoisson(self,coeffs,x,y):
		model = self.model(x,*coeffs)
		jac = self.counts*self.jac_model(x,*coeffs).T
		return scipy.sum(-y/model*jac+jac,axis=-1)

	def chi2(self,coeffs,x,y):
		model = self.model(x,*coeffs)
		return scipy.sum((y-model)**2*self.counts)
	
	def jac_chi2(self,coeffs,x,y):
		model = self.model(x,*coeffs)
		jac = self.jac_model(x,*coeffs).T
		return -2.*scipy.sum((y-model)*self.counts*jac,axis=-1)

	def _minimize_(self,x,y,p0=None,bounds=None,cost=None,method=None):
		if cost is None: cost = self.params.get('cost','lnpoisson')
		if bounds is None: bounds = self.params.get('bounds',None)
		self.params['cost'] = cost
		self.params['bounds'] = bounds
		if cost == 'lmfit':
			self.logger.info('Minimizing with lmfit.')
			import lmfit
			fit_params = lmfit.Parameters()
			name_params = ['a{:d}'.format(ipar+1) for ipar in range(len(p0))]
			if bounds is not None:
				for name,value,min_,max_ in zip(name_params,p0,bounds[0],bounds[-1]): fit_params.add(name,value=value,min=min_,max=max_)
			else:
				for name,value in zip(name_params,p0): fit_params.add(name,value=value)
			sigma = scipy.sqrt(self.counts)
			def residual(coeffs,x,y):
				model = self.model(x,*[coeffs[par].value for par in name_params])
				return (y-model)*sigma
			popt = lmfit.minimize(residual,fit_params,args=(x,y)).params
			popt = [popt[par].value for par in name_params]
		elif cost == 'model':
			if method is None: method = self.params.get('method','lm' if bounds is None else 'trf')
			self.logger.info('Using curve_fit method {} with power-law model.'.format(method))
			bounds = (-scipy.inf,scipy.inf) if bounds is None else bounds
			popt, pcov = optimize.curve_fit(self.model,x,y,p0=p0,sigma=scipy.sqrt(1./self.counts),method=method,jac=self.jac_model,bounds=bounds,maxfev=100000)
		else:
			if method is None: method = self.params.get('method','BFGS')
			self.logger.info('Minimizing cost function {} with method {}.'.format(cost,method))
			bounds = (-scipy.inf,scipy.inf) if bounds is None else bounds
			popt = optimize.minimize(getattr(self,cost),x0=p0,args=(x,y),method=method,jac=getattr(self,'jac_{}'.format(cost)),bounds=bounds).x
		
		self.params['method'] = method
		self.logger.info('popt: {}.'.format(popt))

		return popt

	def normalize(self,normalize=False):
		if normalize:
			self.logger.info('Rescaling {} result by 1/{:.5g}.'.format(self.__class__.__name__,self.norm_ssr))
			return 1./self.norm_ssr
		return 1.

	def clip(self,toret,clip=None,default=None):
		if clip is None: clip = self.params.get('clip',default)
		if clip:
			mask = toret < clip[0]
			if mask.any():
				toret[mask] = clip[0]
				self.logger.info('Applying low-bound {:.4g} to {:d} objects.'.format(clip[0],mask.sum()))
			mask = toret > clip[-1]
			if mask.any():
				toret[mask] = clip[-1]
				self.logger.info('Applying high-bound {:.4g} to {:d} objects.'.format(clip[-1],mask.sum()))

class FitPlateSSR(ModelPower):
	"""Class to fit the SSR = f(plate_MJD_SN_MEDIAN_ALL) relation,
	following Anand's recipe.
	Covariance matrix is taken to be the number of objects in each plate-MJD.

	Parameters
	----------
	params : dict
		parameter dictionary.

	"""
	logger = logging.getLogger('FitPlateSSR')

	def __init__(self,params={}):
		self.params = params

	def set_data(self,data,mask=None,key_plate=None,key_sn=None):
		
		if mask is None: mask = data.trues()
		subsample = self.params.get('data_subsample',{})
		mask = mask & data.subsample(subsample,txt='subsample')

		if key_plate is None: key_plate = self.params.get('key_plate','plate_MJD')
		if key_sn is None: key_sn = self.params.get('key_sn','SN_MEDIAN_ALL')
		for key,key_str in zip([key_plate,key_sn],['key_plate','key_sn']): self.params[key_str] = key
		self.logger.info('Using keys (plate,sn) = ({},{}).'.format(key_plate,key_sn))
		_,self.sn,self.ssr,self.counts = get_plate_mjd_stats(data,mask=mask,key_plate=key_plate,key_sn=key_sn,uniques=True,return_counts=True)

		self.norm_ssr = scipy.average(self.ssr,weights=self.counts)
		self.logger.info('Norm SSR is {:.4f}.'.format(self.norm_ssr))

	def minimize(self,cost=None,method=None):
		self.bestfit = {}
		self.bestfit['values'] = self._minimize_(self.sn,self.ssr,p0=[self.norm_ssr,1.,1.,2.],cost=cost,method=method)
	
	def predicted_ssr(self,data,mask=None,key_plate=None,key_sn=None,clip=None,normalize=False):
		if isinstance(data,scipy.ndarray):
			sn = data
		else:
			if key_plate is None: key_plate = self.params.get('key_plate','plate_MJD')
			if key_sn is None: key_sn = self.params.get('key_sn','SN_MEDIAN_ALL')
			self.logger.info('Using key sn = {}.'.format(key_sn))
			plates,sn,ssr = get_plate_mjd_stats(data,mask=mask,key_plate=key_plate,key_sn=key_sn)
		if mask is not None: sn = sn[mask]
		
		toret = self.model(sn,*self.bestfit['values'])
		self.clip(toret,clip,default=False)
		toret *= self.normalize(normalize)

		return toret

	@classmethod
	def setstate(cls,state):
		self = cls(params=state.get('params',{}))
		self.__dict__.update(state)
		return self
	
	def getstate(self):
		state = {}
		for key in ['params','bestfit','norm_ssr']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

def _fit_plate_ssr_(catalogue,mask=None,key_plate='plate_MJD',key_sn='SN_MEDIAN_ALL'):
	
	survey_subsample = parameters['fit_plate_ssr']['survey_subsample']
	results = {key:{} for key in survey_subsample}

	if mask is None: mask = catalogue.trues()
	mask_hasfiber = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample') & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')
	
	for survey in results:

		fit = FitPlateSSR(params=parameters['fit_plate_ssr'])
		fit.set_data(catalogue,mask=mask_hasfiber & catalogue.subsample(survey_subsample[survey]),key_plate=key_plate,key_sn=key_sn)
		logger.info('Fitting survey {}.'.format(survey))
		fit.minimize()
		results[survey] = fit.getstate()
		
	return results

def fit_plate_ssr(path_data):
	"""Fit the SSR = f(plate_SN) relation.
	The fit result is saved in parameters['paths']['fit_plate_ssr'].

	Parameters
	----------
	path_data : str
		path to data.

	"""	
	logger.info(utils.log_header('Fitting plate_ssr'))

	path_fit = parameters['paths']['fit_plate_ssr']

	catalogue = ELGCatalogue.load(path_data)
	results = _fit_plate_ssr_(catalogue)
		
	utils.save(path_fit,results)

def _add_model_plate_ssr_(catalogue,results,key_model_ssr='model_plate_ssr',normalize=False,mask=None):

	catalogue[key_model_ssr] = catalogue.nans()
	if mask is None: mask = catalogue.trues()
	mask_hasfiber = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample') & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')

	for survey in results:

		fit = FitPlateSSR.setstate(results[survey])
		mask_survey = mask_hasfiber & catalogue.subsample(fit.params['survey_subsample'][survey])
		catalogue[key_model_ssr][mask_survey] = fit.predicted_ssr(catalogue,mask=mask_survey,normalize=normalize)
	
	mask_hasfiber &= catalogue.subsample(parameters['spectro_subsample'],txt='spectro_subsample')
	mask_bad = catalogue.bad_value(key_model_ssr) & mask_hasfiber
	sum_bad = mask_bad.sum()
	if sum_bad>0:
		logger.warning('{} has {:d} incorrect ({}) values.'.format(key_model_ssr,sum_bad,ELGCatalogue.bad_values(key_model_ssr)))
	ssr = catalogue[key_model_ssr][mask_hasfiber]
	logger.info('{} range: {:.4f} - {:.4f}.'.format(key_model_ssr,ssr.min(),ssr.max()))

def add_model_plate_ssr(catalogue,key_model_ssr='model_plate_ssr',normalize=False):
	"""Add model_plate_ssr to catalogue, as predicted by the fitted SSR = f(plate_MJD_SN_MEDIAN_ALL) relation 
	in parameters['paths']['fit_plate_ssr'].

	Parameters
	----------
	catalogue : ELGCatalogue
		catalogue.
	key_model_ssr : str, optional
		field to SSR = f(plate_SN) relation.
	normalize : bool, optional
		whether to normalize key_model_ssr to 1.

	"""
	path_fit = parameters['paths']['fit_plate_ssr']
	results = utils.load(path_fit,comments='#')

	_add_model_plate_ssr_(catalogue,results,key_model_ssr=key_model_ssr,normalize=normalize)

class FitFiberidSSR(ModelPower):
	"""Class to fit the SSR = f(FIBERID) relation,
	following Anand's recipe.
	Covariance matrix is taken to be the number of objects per fiberid.

	Parameters
	----------
	params : dict
		parameter dictionary.

	"""
	logger = logging.getLogger('FitFiberidSSR')

	def __init__(self,params=None):
		self.params = params

	def set_data(self,data,mask=None,key_fiberid=None,bad_fiberid=[],bad_fiberssr=None,weight=None):

		if mask is None: mask = data.trues()
		subsample = self.params.get('data_subsample',{})
		mask = mask & data.subsample(subsample,txt='subsample')
		subsample = self.params.get('reliable_subsample',{})
		mask_reliable = mask & data.subsample(subsample,txt='reliable')
		if weight is None: weight = data.ones()
		else: weight = weight.copy()
		weight[~mask_reliable] = 0.

		if key_fiberid is None: key_fiberid = self.params.get('key_fiberid','FIBERID')
		self.params[key_fiberid] = key_fiberid
		
		# for each fiber first
		self.fiberid = scipy.unique(data[key_fiberid][mask])
		counts = utils.interp_digitized_statistics(self.fiberid,data[key_fiberid][mask],fill=0.)
		self.fiberssr = utils.interp_digitized_statistics(self.fiberid,data[key_fiberid][mask],fill=0.,values=mask_reliable[mask]*weight[mask])/counts
		self.norm_ssr = (mask_reliable[mask]*weight[mask]).sum()*1./mask.sum()
		self.logger.info('Norm SSR is {:.4f}.'.format(self.norm_ssr))

		self.bad_fiberid = scipy.array(bad_fiberid)
		if bad_fiberssr is None:
			self.logger.info('Recalculating bad fiber SSR.')
			self.bad_fiberssr = utils.digitized_interp(bad_fiberid,self.fiberid,self.fiberssr,fill=scipy.nan)
		else:
			self.bad_fiberssr = scipy.array(bad_fiberssr)
		
		if len(self.bad_fiberid): self.logger.info('Treating {:d} bad fibers separately: {}.'.format(len(self.bad_fiberid),self.bad_fiberid))
		mask_good = ~scipy.in1d(self.fiberid,self.bad_fiberid)

		fidedges = self.params.get('fidedges',scipy.concatenate([self.fiberid,[self.fiberid[-1]+1]]))
		binssr,self.fidedges,binnumber = stats.binned_statistic(self.fiberid[mask_good],self.fiberssr[mask_good],statistic='median',bins=fidedges)
		counts = stats.binned_statistic(self.fiberid[mask_good],counts[mask_good],statistic='sum',bins=self.fidedges)[0]
		
		mask_good = utils.isnotnaninf(binssr)
		fiberid = (self.fidedges[:-1] + self.fidedges[1:] - 1.)/2.
		self.binned_fiberssr = []; self.binned_fiberid = []; self.binned_counts = []
		self.specedges = self.params.get('specedges',[1,251,501,751,1001])
		for fidmin,fidmax in zip(self.specedges[:-1],self.specedges[1:]):
			mask = mask_good & (self.fidedges[:-1] >= fidmin) & (self.fidedges[1:] <= fidmax)
			#tmp = (self.fidedges[:-1] >= fidmin) & (self.fidedges[1:] <= fidmax); print fiberid[tmp].min(),fiberid[tmp].max()
			self.binned_fiberid.append(fiberid[mask])
			self.binned_fiberssr.append(binssr[mask])
			self.binned_counts.append(counts[mask])

	def minimize(self,cost=None,method=None):
		self.bestfit = []
		for fiberid,binssr,self.counts in zip(self.binned_fiberid,self.binned_fiberssr,self.binned_counts):
			min_fiberid,max_fiberid = fiberid.min(),fiberid.max()
			if self.params.get('bounds',None) is not None:
				self.params['bounds'][0][2] = min_fiberid
				self.params['bounds'][1][2] = max_fiberid
			popt = self._minimize_(fiberid,binssr,p0=[self.norm_ssr,1.,(min_fiberid+max_fiberid)/2.,2.],cost=cost,method=method)
			self.bestfit.append({'values':popt})
		
	def predicted_ssr(self,data,mask=None,add_bad_fiberssr=True,key_fiberid=None,clip=None,normalize=False):
	
		if key_fiberid is None: key_fiberid = self.params.get('key_fiberid','FIBERID')

		if isinstance(data,scipy.ndarray):
			fiberid = data
		else:
			fiberid = data[key_fiberid]

		if mask is not None: fiberid = fiberid[mask]
		
		toret = scipy.nan*scipy.ones_like(fiberid,dtype=scipy.float64)
		for fidmin,fidmax,bestfit in zip(self.specedges[:-1],self.specedges[1:],self.bestfit):
			mask = (fiberid >= fidmin) & (fiberid < fidmax)
			#print fiberid[mask].min(),fiberid[mask].max()
			toret[mask] = self.model(fiberid[mask],*bestfit['values'])
		
		if add_bad_fiberssr:
			self.logger.info('Adding bad fiber SSR for {}.'.format(self.bad_fiberid))
			for bad_fiberid,bad_fiberssr in zip(self.bad_fiberid,self.bad_fiberssr):
				mask = fiberid == bad_fiberid
				toret[mask] = bad_fiberssr

		self.clip(toret,clip,default=False)
		toret *= self.normalize(normalize)

		return toret
		
	def get_bad_fiberid(self,nsigmas=None):
		
		if nsigmas is None: nsigmas = self.params.get('bad_fiberid_sigmas',3.)
		self.params['bad_fiberid_sigmas'] = nsigmas
		ssr = scipy.nan*scipy.ones_like(self.fiberid,dtype=scipy.float64)
		for fidmin,fidmax,bestfit in zip(self.specedges[:-1],self.specedges[1:],self.bestfit):
			mask = (self.fiberid >= fidmin) & (self.fiberid < fidmax)
			ssr[mask] = self.model(self.fiberid[mask],*bestfit['values'])

		diff = ssr - self.fiberssr
		self.std_ssr = scipy.std(diff)
		mask_bad = scipy.absolute(diff) >= nsigmas*self.std_ssr

		return self.fiberid[mask_bad],self.fiberssr[mask_bad]

	@classmethod
	def setstate(cls,state):
		self = cls(params=state.get('params',{}))
		self.__dict__.update(state)
		return self
	
	def getstate(self):
		state = {}
		for key in ['params','bestfit','specedges','fidedges','norm_ssr','bad_fiberid','bad_fiberssr']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

def _fit_fiberid_ssr_(catalogue,mask=None,key_fiberid='FIBERID',weight=None):
	
	survey_subsample = parameters['fit_fiberid_ssr']['survey_subsample']
	results = {key:{} for key in survey_subsample}
	
	if mask is None: mask = catalogue.trues()
	mask_hasfiber = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample') & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')
	
	fit = FitFiberidSSR(parameters['fit_fiberid_ssr'])
	fit.set_data(catalogue,mask=mask_hasfiber,key_fiberid=key_fiberid,weight=weight)
	fit.minimize() # preliminary fit to spot outliers
	bad_fiberid,bad_fiberssr = fit.get_bad_fiberid()

	for survey in results:

		fit = FitFiberidSSR(params=parameters['fit_fiberid_ssr'])
		fit.set_data(catalogue,mask=mask_hasfiber & catalogue.subsample(survey_subsample[survey]),key_fiberid=key_fiberid,bad_fiberid=bad_fiberid,bad_fiberssr=bad_fiberssr,weight=weight)
		logger.info('Fitting survey {}.'.format(survey))
		fit.minimize()
		results[survey] = fit.getstate()
		
	return results

def fit_fiberid_ssr(path_data):
	"""Fit the SSR = f(FIBERID) relation.
	The fit result is saved in parameters['paths']['fit_fiberid_ssr'].

	Parameters
	----------
	path_data : str
		path to data.

	"""
	logger.info(utils.log_header('Fitting fiberid_ssr'))

	path_fit = parameters['paths']['fit_fiberid_ssr']

	catalogue = ELGCatalogue.load(path_data)
	results = _fit_fiberid_ssr_(catalogue)
		
	utils.save(path_fit,results)

def _add_model_fiberid_ssr_(catalogue,results,key_fiberid='FIBERID',key_model_ssr='model_fiberid_ssr',normalize=False,mask=None):

	catalogue[key_model_ssr] = catalogue.nans()
	if mask is None: mask = catalogue.trues()
	mask_hasfiber = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample') & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')

	for survey in results:

		fit = FitFiberidSSR.setstate(results[survey])
		mask_survey = mask_hasfiber & catalogue.subsample(fit.params['survey_subsample'][survey])
		catalogue[key_model_ssr][mask_survey] = fit.predicted_ssr(catalogue,mask=mask_survey,key_fiberid=key_fiberid,normalize=normalize)
	
	mask_hasfiber &= catalogue.subsample(parameters['spectro_subsample'],txt='spectro_subsample')
	mask_bad = catalogue.bad_value(key_model_ssr) & mask_hasfiber
	sum_bad = mask_bad.sum()
	if sum_bad>0:
		logger.warning('{} has {:d} incorrect ({}) values.'.format(key_model_ssr,sum_bad,ELGCatalogue.bad_values(key_model_ssr)))
	ssr = catalogue[key_model_ssr][mask_hasfiber]
	logger.info('{} range: {:.4f} - {:.4f}.'.format(key_model_ssr,ssr.min(),ssr.max()))


def add_model_fiberid_ssr(catalogue,key_model_ssr='model_fiberid_ssr',normalize=False):
	"""Add model_fiberid_ssr to catalogue, as predicted by the fitted SSR = f(FIBERID) relation 
	in parameters['paths']['fit_fiberid_ssr'].

	Parameters
	----------
	catalogue : ELGCatalogue
		catalogue.
	key_model_ssr : str, optional
		field to SSR = f(FIBERID) relation.
	normalize : bool, optional
		whether to normalize key_model_ssr to 1.

	"""
	path_fit = parameters['paths']['fit_fiberid_ssr']
	results = utils.load(path_fit,comments='#')

	_add_model_fiberid_ssr_(catalogue,results,key_model_ssr=key_model_ssr,normalize=normalize)


class FitXYFocalSSR(ModelPower):
	"""Class to fit the SSR = f(XYFOCAL) relation,
	following Anand's recipe.
	Covariance matrix is taken to be the number of objects per XYFocal bin.

	Parameters
	----------
	params : dict
		parameter dictionary.

	"""
	logger = logging.getLogger('FitXYFocalSSR')

	def __init__(self,params=None):
		self.params = params

	def model(self,x,*coeffs):
		#return coeffs[0] - scipy.absolute(coeffs[1]*(x[0]-coeffs[2]))**coeffs[3] - scipy.absolute(coeffs[4]*(x[1]-coeffs[5]))**coeffs[6]
		return coeffs[0] - coeffs[1]*scipy.absolute(x[0]-coeffs[2])**coeffs[3] - coeffs[4]*scipy.absolute(x[1]-coeffs[5])**coeffs[6]

	def jac_half(self,x,*coeffs):
		"""
		diff = coeffs[1]*(x-coeffs[2])
		diffabs = scipy.absolute(diff)
		tmp = diff*diffabs**(coeffs[3]-2.)
		c1 = -coeffs[3]*(x-coeffs[2])*tmp
		c2 = coeffs[3]*coeffs[1]*tmp
		c3 = -scipy.log(diffabs)*diffabs**coeffs[3]
		"""	
		diff = x-coeffs[2]
		diffabs = scipy.absolute(diff)
		c1 = -diffabs**coeffs[3]
		c2 = coeffs[1]*coeffs[3]*diff*diffabs**(coeffs[3]-2.)
		c3 = -coeffs[1]*scipy.log(diffabs)*diffabs**coeffs[3]
		return c1,c2,c3
	
	def jac_model(self,x,*coeffs):
		c0 = scipy.ones_like(x[0],dtype='f8')
		c1,c2,c3 = self.jac_half(x[0],*coeffs)
		c4,c5,c6 = self.jac_half(x[1],*coeffs[3:])
		return scipy.transpose([c0,c1,c2,c3,c4,c5,c6])

	def set_data(self,data,mask=None,key_xfocal=None,key_yfocal=None,weight=None):

		if mask is None: mask = data.trues()
		subsample = self.params.get('data_subsample',{})
		mask = mask & data.subsample(subsample,txt='subsample')
		subsample = self.params.get('reliable_subsample',{})
		mask_reliable = mask & data.subsample(subsample,txt='reliable')
		if weight is None: weight = data.ones()
		else: weight = weight.copy()
		weight[~mask_reliable] = 0.

		if key_xfocal is None: key_xfocal = self.params.get('key_xfocal','XFOCAL')
		if key_yfocal is None: key_yfocal = self.params.get('key_yfocal','YFOCAL')
		for key,key_str in zip([key_xfocal,key_yfocal],['key_xfocal','key_yfocal']): self.params[key_str] = key
		
		xyedges = self.params.get('xyedges',(scipy.linspace(-326.,326.,21),)*2)
		xfocal,yfocal = data[key_xfocal][mask],data[key_yfocal][mask]
		self.counts,xedges,yedges = stats.binned_statistic_2d(xfocal,yfocal,values=mask_reliable[mask],statistic='count',bins=xyedges)[:3]
		self.xyedges = (xedges,yedges)
		reliable = stats.binned_statistic_2d(xfocal,yfocal,values=mask_reliable[mask]*weight[mask],statistic='sum',bins=self.xyedges)[0]
		self.ssr = reliable/self.counts
		self.norm_ssr = (mask_reliable[mask]*weight[mask]).sum()*1./mask.sum()
		self.logger.info('Norm SSR is {:.4f}.'.format(self.norm_ssr))

		#x = stats.binned_statistic_2d(xfocal,yfocal,values=xfocal,statistic='mean',bins=xyedges)[0]
		#y = stats.binned_statistic_2d(xfocal,yfocal,values=yfocal,statistic='mean',bins=xyedges)[0]
		self.x,self.y = map(lambda e: (e[:-1] + e[1:])/2.,self.xyedges)
		self.xy = scipy.meshgrid(self.x,self.y,indexing='ij')

		self.allcounts = self.counts.copy()
		mask = self.counts > 0
		for key in ['counts','ssr']: setattr(self,key,getattr(self,key)[mask])
		self.xy = tuple(xy[mask] for xy in self.xy)

	def minimize(self,cost=None,method=None):
		self.bestfit = {}
		self.bestfit['values'] = self._minimize_(self.xy,self.ssr,p0=[self.norm_ssr,1,0.,2.,1.,0.,2.],cost=cost,method=method)

	def handle_boundaries(self,x,y,extrapolate=True):
		
		if extrapolate == True: return

		xybin = stats.binned_statistic_2d(x,y,values=x,statistic='count',bins=self.xyedges,expand_binnumbers=True)[3]
		xbin,ybin = xybin[0]-1,xybin[1]-1
		xb,yb = map(lambda e: (e[:-1] + e[1:])/2.,self.xyedges)

		if extrapolate == 'inside':
			mask_inside = (xbin > 0) & (xbin < len(self.xyedges[0])-1) & (ybin > 0) & (ybin < len(self.xyedges[1])-1)
			xbin,ybin = (scipy.clip(bin,1,len(edges)-3) for bin,edges in zip([xbin,ybin],self.xyedges))
			mask_inside &= self.allcounts[(xbin,ybin)] > 0
			mask_inside &= (self.allcounts[(xbin-1,ybin)] > 0) & (self.allcounts[(xbin+1,ybin)] > 0)
			mask_inside &= (self.allcounts[(xbin,ybin-1)] > 0) & (self.allcounts[(xbin,ybin-1)] > 0)
			mask_inside &= (self.allcounts[(xbin-1,ybin-1)] > 0) & (self.allcounts[(xbin+1,ybin-1)] > 0)
			mask_inside &= (self.allcounts[(xbin-1,ybin+1)] > 0) & (self.allcounts[(xbin+1,ybin+1)] > 0)
			mask_outside = ~mask_inside

		elif extrapolate == 'xylim':
			mask_high = scipy.zeros_like(x,dtype=scipy.bool_)
			mask_low = scipy.zeros_like(x,dtype=scipy.bool_)
			for xc,yc in zip(*self.xy):
				mask_high |= (x<=xc) & (y<=yc)
				mask_low |= (x>=xc) & (y>=yc)
			mask_outside = ~(mask_high & mask_low)
		
		elif extrapolate == 'bin':
			mask_outside = scipy.ones_like(x,dtype=scipy.bool_)

		self.logger.info('Replacing {:d}/{:d} (XFOCAL,YFOCAL) by their binned value.'.format(mask_outside.sum(),len(mask_outside))) 
		x[mask_outside] = xb[xybin[0][mask_outside]-1]
		y[mask_outside] = yb[xybin[1][mask_outside]-1]
	
	def predicted_ssr(self,data,mask=None,key_xfocal=None,key_yfocal=None,extrapolate=True,clip=None,normalize=False):

		if key_xfocal is None: key_xfocal = self.params.get('key_xfocal','XFOCAL')
		if key_yfocal is None: key_yfocal = self.params.get('key_yfocal','YFOCAL')
		if extrapolate is None: extrapolate = self.params.get('extrapolate',True)
		x,y = data[key_xfocal][mask],data[key_yfocal][mask]

		self.handle_boundaries(x,y,extrapolate=extrapolate)
		toret = self.model((x,y),*self.bestfit['values'])
		self.clip(toret,clip,default=False)
		toret *= self.normalize(normalize)

		return toret

	@classmethod
	def setstate(cls,state):
		self = cls(params=state.get('params',{}))
		self.__dict__.update(state)
		return self
	
	def getstate(self):
		state = {}
		for key in ['params','bestfit','xyedges','xy','allcounts','norm_ssr']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

def _fit_xyfocal_ssr_(catalogue,mask=None,key_xfocal='XFOCAL',key_yfocal='YFOCAL',weight=None):
	
	survey_subsample = parameters['fit_xyfocal_ssr']['survey_subsample']
	results = {key:{} for key in survey_subsample}
	
	if mask is None: mask = catalogue.trues()
	mask_hasfiber = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample') & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')

	for survey in results:

		fit = FitXYFocalSSR(params=parameters['fit_xyfocal_ssr'])
		fit.set_data(catalogue,mask=mask_hasfiber & catalogue.subsample(survey_subsample[survey]),key_xfocal=key_xfocal,key_yfocal=key_yfocal,weight=weight)
		logger.info('Fitting survey {}.'.format(survey))
		fit.minimize()
		results[survey] = fit.getstate()
		
	return results

def fit_xyfocal_ssr(path_data):
	"""Fit the SSR = f(XYFOCAL) relation.
	The fit result is saved in parameters['paths']['fit_xyfocal_ssr'].

	Parameters
	----------
	path_data : str
		path to data.

	"""
	logger.info(utils.log_header('Fitting xyfocal_ssr'))

	path_fit = parameters['paths']['fit_xyfocal_ssr']

	catalogue = ELGCatalogue.load(path_data)
	results = _fit_xyfocal_ssr_(catalogue)
		
	utils.save(path_fit,results)

def _add_model_xyfocal_ssr_(catalogue,results,key_xfocal='XFOCAL',key_yfocal='YFOCAL',key_model_ssr='model_xyfocal_ssr',normalize=False,mask=None):

	catalogue[key_model_ssr] = catalogue.nans()
	if mask is None: mask = catalogue.trues()
	mask_hasfiber = mask & catalogue.subsample(parameters['target_subsample'],txt='target_subsample') & catalogue.subsample(parameters['fiber_subsample'],txt='fiber_subsample')

	for survey in results:

		fit = FitXYFocalSSR.setstate(results[survey])
		mask_survey = mask_hasfiber & catalogue.subsample(fit.params['survey_subsample'][survey])
		catalogue[key_model_ssr][mask_survey] = fit.predicted_ssr(catalogue,mask=mask_survey,key_xfocal=key_xfocal,key_yfocal=key_yfocal,normalize=normalize)
	
	mask_hasfiber &= catalogue.subsample(parameters['spectro_subsample'],txt='spectro_subsample')
	mask_bad = catalogue.bad_value(key_model_ssr) & mask_hasfiber
	sum_bad = mask_bad.sum()
	if sum_bad>0:
		logger.warning('{} has {:d} incorrect ({}) values.'.format(key_model_ssr,sum_bad,ELGCatalogue.bad_values(key_model_ssr)))
	ssr = catalogue[key_model_ssr][mask_hasfiber]
	logger.info('{} range: {:.4f} - {:.4f}.'.format(key_model_ssr,ssr.min(),ssr.max()))


def add_model_xyfocal_ssr(catalogue,key_model_ssr='model_xyfocal_ssr',normalize=False):
	"""Add model_fiberid_ssr to catalogue, as predicted by the fitted SSR = f(XYFOCAL) relation 
	in parameters['paths']['fit_xyfocal_ssr'].

	Parameters
	----------
	catalogue : ELGCatalogue
		catalogue.
	key_model_ssr : str, optional
		field to SSR = f(XYFOCAL) relation.
	normalize : bool, optional
		whether to normalize key_model_ssr to 1.

	"""
	path_fit = parameters['paths']['fit_xyfocal_ssr']
	results = utils.load(path_fit,comments='#')

	_add_model_xyfocal_ssr_(catalogue,results,key_model_ssr=key_model_ssr,normalize=normalize)

