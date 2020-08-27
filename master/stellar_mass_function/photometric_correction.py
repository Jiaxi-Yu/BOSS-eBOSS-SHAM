import os
import copy
import logging
import scipy
from scipy import constants,stats
from catalogueELG import ELGCatalogue
import utils

logger = logging.getLogger('Photometry')

def set_parameters(params):
	global parameters
	parameters = params

def get_array_parameters(minuit,params,repeat_params=[],nrepeat=1):
	"""Setup the minuit parameter dictionary.

	Parameters
	----------
	minuit : dict
		minuit parameter dictionary.
	params : list
		list of parameters to use.
	repeat_params : list, optional
		list of parameters to repeat (appending _num) for e.g. each redshift slice.
	nrepeat: int, optional
		number of repetitions.

	Returns
	-------
	new : dict
		Minuit parameter dictionary.
	indices_repeat : 2d array
		list of coeff index (i.e. indices in new['forced_parameters']) for each repeat 
	list_ref : list
		list of base parameters = 'epsilon' + regression variables
	indices_linear : dict
		each key is the repeat number and each value the corresponding (coeff index in the repeat,photo parameter index)
	indices_quadratic :
		each key is the repeat number and each value the corresponding (coeff index in the repeat,photo parameter index #1,photo parameter index #2)

	"""
	new = {}
	for key in ['pedantic','print_level','errordef']:
		if key in minuit: new[key] = minuit[key]
	new['forced_parameters'] = []
	indices_repeat = [[] for irep in range(nrepeat)]
	
	parameters = []
	par_to_rep = {}
	par_to_ref = {}
	indices_linear = {irep:([],[]) for irep in [None]+range(nrepeat)}
	indices_quadratic = {irep:([],[],[]) for irep in [None]+range(nrepeat)}
	
	ref = 'epsilon'
	list_ref = [ref]
	if ref in repeat_params:
		for irep in range(nrepeat):
			par = '{}_{:d}'.format(ref,irep+1)
			par_to_ref[par] = ref
			par_to_rep[par] = irep
			new[par] = minuit.get(ref,1.)
			new['error_{}'.format(par)] = minuit.get('error_{}'.format(par),minuit.get('error_{}'.format(ref),1.))
			new['fix_{}'.format(par)] = minuit.get('fix_{}'.format(par),minuit.get('fix_{}'.format(ref),False))
			new['forced_parameters'].append(par)
			indices_repeat[irep].append(len(new['forced_parameters'])-1)
	else:
		par_to_ref[ref] = ref
		par_to_rep[ref] = None
		new[ref] = minuit.get(ref,1.)
		new['error_{}'.format(ref)] = minuit.get('error_{}'.format(ref),1.)
		new['fix_{}'.format(ref)] = minuit.get('fix_{}'.format(ref),False)
		new['forced_parameters'].append(ref)
		for irep in range(nrepeat): indices_repeat[irep].append(len(new['forced_parameters'])-1)
	
	for iref,ref in enumerate(params):
		list_ref.append(ref)
		indices_linear[None][0].append(iref+1)
		indices_linear[None][1].append(iref)
		if ref in repeat_params:
			for irep in range(nrepeat):
				par = '{}_{:d}'.format(ref,irep+1)
				par_to_ref[par] = ref
				par_to_rep[par] = irep
				parameters.append(par)
				new[par] = minuit.get(par,minuit.get(ref,0.))
				new['error_{}'.format(par)] = minuit.get('error_{}'.format(par),minuit.get('error_{}'.format(ref),1.))
				new['fix_{}'.format(par)] = minuit.get('fix_{}'.format(par),minuit.get('fix_{}'.format(ref),(par not in minuit) and (ref not in minuit)))
				new['forced_parameters'].append(par)
				indices_repeat[irep].append(len(new['forced_parameters'])-1)
				if not (new['fix_{}'.format(par)] and (new[par] == 0.)): # keep coefficient if left free or non-zero
					indices_linear[irep][0].append(len(indices_repeat[irep])-1)
					indices_linear[irep][1].append(iref)
		else:
			parameters.append(ref)
			par_to_ref[ref] = ref
			par_to_rep[ref] = None
			new[ref] = minuit.get(ref,1.)
			new['error_{}'.format(ref)] = minuit.get('error_{}'.format(ref),1.)
			new['fix_{}'.format(ref)] = minuit.get('fix_{}'.format(ref),ref not in minuit)
			new['forced_parameters'].append(ref)
			for irep in range(nrepeat):
				indices_repeat[irep].append(len(new['forced_parameters'])-1)
				if not (new['fix_{}'.format(ref)] and (new[ref] == 0.)):
					indices_linear[irep][0].append(len(indices_repeat[irep])-1)
					indices_linear[irep][1].append(iref)
	
	quadratic = False
	for par1 in parameters:
		for par2 in parameters:
			if (par_to_rep[par1] is None) or (par_to_rep[par2] is None) or (par_to_rep[par1] == par_to_rep[par2]): # quadratic terms are made of coeff_ibin1_ibin2 with ibin1 == ibin2 or one of them None
				par = '{}_{}'.format(par1,par2)
				ref = '{}_{}'.format(par_to_ref[par1],par_to_ref[par2])
				zero = minuit.get(par,minuit.get(ref,0.)) == 0.
				fixed = minuit.get('fix_{}'.format(par),minuit.get('fix_{}'.format(ref),(par not in minuit) and (ref not in minuit)))
				if (not zero) or (not fixed):
					quadratic = True # add quadratic terms if at least one quadratic coefficient is left free or set to non zero
					break
	
	if quadratic:
		for ipar1,par1 in enumerate(params):
			for ipar2,par2 in enumerate(params[ipar1:]): # start at ipar1 to avoid double counting cross-terms
				indices_quadratic[None][1].append(ipar1)
				indices_quadratic[None][2].append(ipar1+ipar2)
				indices_quadratic[None][0].append(len(indices_linear[None][0])+len(indices_quadratic[None][1])) # + 1 (coeffs start by epsilon) - 1 (index of quadratic coeff) = 0
	
		for ipar1,par1 in enumerate(parameters):
			for par2 in parameters[ipar1:]:
				if (par_to_rep[par1] is None) or (par_to_rep[par2] is None) or (par_to_rep[par1] == par_to_rep[par2]):
					par = '{}_{}'.format(par1,par2)
					ref = '{}_{}'.format(par_to_ref[par1],par_to_ref[par2])
					if ref not in list_ref: list_ref.append(ref)
					new[par] = minuit.get(par,minuit.get(ref,0.))
					new['error_{}'.format(par)] = minuit.get('error_{}'.format(par),minuit.get('error_{}'.format(ref),1.))
					new['fix_{}'.format(par)] = minuit.get('fix_{}'.format(par),minuit.get('fix_{}'.format(ref),(par not in minuit) and (ref not in minuit)))
					new['forced_parameters'].append(par)
					ipar = len(new['forced_parameters'])-1
					indices = (params.index(par_to_ref[par1]),params.index(par_to_ref[par2]))
					if par_to_rep[par1] is not None:
						indices_repeat[par_to_rep[par1]].append(ipar)
						if not (new['fix_{}'.format(par)] and (new[par] == 0.)):
							indices_quadratic[par_to_rep[par1]][0].append(len(indices_repeat[par_to_rep[par1]])-1)
							for i in (1,2): indices_quadratic[par_to_rep[par1]][i].append(indices[i-1])
					elif par_to_rep[par2] is not None:
						indices_repeat[par_to_rep[par2]].append(ipar)
						if not (new['fix_{}'.format(par)] and (new[par] == 0.)):
							indices_quadratic[par_to_rep[par2]][0].append(len(indices_repeat[par_to_rep[par2]])-1)
							for i in (1,2): indices_quadratic[par_to_rep[par2]][i].append(indices[i-1])
					else:
						for irep in range(nrepeat):
							indices_repeat[irep].append(ipar)
							if not (new['fix_{}'.format(par)] and (new[par] == 0.)):
								indices_quadratic[irep][0].append(len(indices_repeat[irep])-1)
								for i in (1,2): indices_quadratic[irep][i].append(indices[i-1])

	return new,scipy.array(indices_repeat),list_ref,indices_linear,indices_quadratic

class BaseFit(object):
	"""Base class for fit, based on minuit.

	Parameters
	----------
	params : dict
		parameter dictionary.

	"""
	_MINOS_ARGS = ['lower','upper','lower_valid','upper_valid','is_valid']
	
	def __init__(self,params={}):
		self.params = copy.deepcopy(params)
		self.params['minuit'],_,_,self.indices_linear,self.indices_quadratic = get_array_parameters(self.params['minuit'],self.params['params'],repeat_params=[],nrepeat=1)
		self.nlinear = len(self.indices_linear[None][0])
		self.nquadratic = len(self.indices_quadratic[None][0])
		
	def chi2(self,*args,**kwargs):
		"""Placeholder to remind that this function needs to be defined for a new fitting class.
		Raises
		------
		NotImplementedError

		"""
		raise NotImplementedError('You must implement method chi2 in your fitting class.')
		
	def model(self,*args,**kwargs):
		"""Placeholder to remind that this function needs to be defined for a new fitting class.
		Raises
		------
		NotImplementedError

		"""
		raise NotImplementedError('You must implement method chi2 in your fitting class.')
	
	def set_minuit(self,minuit={},cost=None):
		import iminuit
		if not minuit: minuit = self.params['minuit']
		if cost is None: cost = self.params.get('cost','chi2')
		self.params['chi2'] = cost
		self.logger.info('Minimizing cost function {}.'.format(cost))
		self.minuit = iminuit.Minuit(getattr(self,cost),**minuit)
	
	def run_migrad(self,migrad={}):
		if not migrad: migrad = self.params['migrad']
		self.minuit.migrad(**migrad)
		self.set_bestfit()
		
	def run_minos(self,minos={}):
		if not minos: minos = self.params['minos']
		self.minuit.minos(**minos)
		self.set_bestfit()
	
	def set_bestfit(self):
		self.bestfit = {'values':{},'errors':{}}
		for key in self.bestfit.keys():
			if hasattr(self.minuit,key): self.bestfit[key] = dict(getattr(self.minuit,key))
		if hasattr(self.minuit,'fval'): self.bestfit['fval'] = getattr(self.minuit,'fval')
		minos = self.minuit.get_merrors()
		if minos:
			self.bestfit['minos'] = {}
			for par in minos:
				for key in self._MINOS_ARGS:
					if not key in self.bestfit['minos']: self.bestfit['minos'][key] = {}
					self.bestfit['minos'][key][par] = minos[par][key]
		self.bestfit['chi2'] = self.chi2(*[self.bestfit['values'][key] for key in self.parameters])
		self.bestfit['rchi2'] = self.bestfit['chi2']/(self.nobs-self.nvary)
	
	@property	
	def vary(self):
		return self.minuit.list_of_vary_param()
	
	@property
	def nvary(self):
		return len(self.vary)
		
	@property	
	def fixed(self):
		return self.minuit.list_of_fixed_param()
	
	@property
	def nfixed(self):
		return len(self.fixed)
		
	@property
	def parameters(self):
		return self.params['minuit']['forced_parameters']
	
	def set_null(self,cost=None):
	
		params = copy.deepcopy(self.params)
		params['minuit']['pedantic'] = False
		params['minuit']['print_level'] = 0
		self.nvarynull = 0
		for key in params['minuit']:
			if 'fix_' in key:
				params['minuit'][key] = 'epsilon' not in key
				self.nvarynull += not params['minuit'][key]
		self.set_minuit(minuit=params['minuit'],cost=cost)
		self.run_migrad(migrad=params['migrad'])
		self.null = copy.deepcopy(self.bestfit)

	def minimize(self,cost=None):

		self.set_null(cost=cost)
		self.set_minuit(cost=cost)
		self.run_migrad()

		for when in ['null','bestfit']:
			chi2 = getattr(self,when)['chi2']
			rchi2 = getattr(self,when)['rchi2']
			self.logger.info('Reduced {}-chi2: {:.5g}/({:d}-{:d}) = {:.5g}'.format(when,chi2,self.nobs,self.nvarynull if when=='null' else self.nvary,rchi2))
		
		model = self.model(*[self.bestfit['values'][key] for key in self.parameters])
		self.logger.info('Model range: {:.4g} - {:.4g}.'.format(model.min(),model.max()))

	@classmethod
	def setstate(cls,state):
		self = object.__new__(cls)
		self.__dict__.update(state)
		self.__init__(params=state.get('params',{}))
		return self


class FitDensity(BaseFit):
	"""Class to fit density, i.e. to regress ndata/nrandoms
	v.s. photometric parameters in healpix pixels, possibly binned in e.g. redshift.
	Covariance matrix is taken to be Poisson.

	Parameters
	----------
	params : dict
		parameter dictionary.

	"""
	logger = logging.getLogger('FitDensity')
	
	def __init__(self,params={}):
	
		self.params = copy.deepcopy(params)
		self.params['minuit'],self.indices_bin,self.unbinned_parameters,self.indices_linear,self.indices_quadratic = get_array_parameters(self.params['minuit'],self.params['params'],repeat_params=self.params.get('params_bin',[]),nrepeat=self.nbins)
		self.nlinear = len(self.indices_linear[None][0])
		self.nquadratic = len(self.indices_quadratic[None][0])

	def chi2(self,*args,**kwargs):
		models = [randoms*model for randoms,model in zip(self.randoms,self.model(*args,**kwargs))]
		return scipy.sum([scipy.sum((data-model)**2*invcovariance) for data,model,invcovariance in zip(self.data,models,self.invcovariance)])
		
	def lnpoisson(self,*args,**kwargs):
		models = [randoms*model for randoms,model in zip(self.randoms,self.model(*args,**kwargs))]
		for model in models:
			if scipy.any(model<=0.): return 1e10
		return scipy.sum([scipy.sum(-data*scipy.log(model)+model) for data,model in zip(self.data,models)])
		
	@property
	def edges(self):
		return self.params.get('edges',scipy.array([0.,scipy.inf]))
	
	@property
	def nbins(self):
		return len(self.edges)-1
	
	def get_ibin_coeffs(self,coeffs,ibin):
		if isinstance(coeffs,scipy.ndarray):
			return coeffs[self.indices_bin[ibin]] 
		return [coeffs[ind] for ind in self.indices_bin[ibin]]
	
	def get_ibin_bestfit(self,ibin,result='values'):
		if result in ['values','errors']: values = self.bestfit[result]
		elif result in ['upper','lower']: values = self.bestfit['minos'][result]
		else: raise ValueError('I do not have result {}.'.format(result))
		coeffs = [values[key] for key in self.parameters]
		coeffs = self.get_ibin_coeffs(scipy.array(coeffs),ibin).T
		return {par:coeff for par,coeff in zip(self.unbinned_parameters,coeffs)}

	def model(self,*coeffs):
		return self._binned_model_(self.photo,*coeffs)
	
	def _binned_model_(self,photo,*coeffs):
		return [self._unbinned_model_(photo[ibin],self.get_ibin_coeffs(coeffs,ibin),ibin=ibin) for ibin in range(self.nbins)]
	
	def _unbinned_model_(self,photo,coeffs,ibin=None):
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

	def minimize(self,cost=None):

		self.set_null(cost=cost)
		self.set_minuit(cost=cost)
		self.run_migrad()

		for when in ['null','bestfit']:
			chi2 = getattr(self,when)['chi2']
			rchi2 = getattr(self,when)['rchi2']
			self.logger.info('Reduced {}-chi2: {:.5g}/({:d}-{:d}) = {:.5g}'.format(when,chi2,self.nobs,self.nvarynull if when=='null' else self.nvary,rchi2))
		
		models = self.model(*[self.bestfit['values'][key] for key in self.parameters])
		for ibin,model in enumerate(models):
			self.logger.info('Model range in bin {:d}: {:.4g} - {:.4g}.'.format(ibin+1,model.min(),model.max()))
	
	def set_density(self,healpix,keys_data,mean_values=None,keys_randoms=None):
		self.data,self.randoms = [],[]
		assert len(keys_data) == self.nbins
		for ikey,key_data in enumerate(keys_data):
			self.data.append(healpix[key_data])
			if keys_randoms is None: key_randoms = key_data.replace('data_','randoms_')
			elif isinstance(keys_randoms,(str,unicode)): key_randoms = keys_randoms
			else: key_randoms = keys_randoms[ikey]
			self.randoms.append(healpix[key_randoms])
		if mean_values is not None:
			self.mean_values = mean_values
		else:
			self.mean_values = (self.edges[:-1] + self.edges[1:])/2.
		
	def set_photo(self,healpix,keys_photo,values=None):
		self.photo = []
		if not isinstance(keys_photo[0],list): keys_photo = [keys_photo]*self.nbins
		self.photo = []
		for key_photo in keys_photo:
			assert self.nlinear == len(key_photo)
			photo = []
			for key in key_photo:
				if 'depth' in key: photo += [utils.depth_to_flux(healpix[key])]
				#elif 'stardens' in key: photo += [scipy.log10(healpix[key])]
				else: photo += [healpix[key]]
			self.photo.append(scipy.array(photo).T)
		self.values = values
	
	def prepare(self,mask=None,normalize_randoms=True):
	
		if 'normalize_randoms' not in self.params: self.params['normalize_randoms'] = normalize_randoms
		
		self.alpha = [0.]*self.nbins
		self.invcovariance = [0.]*self.nbins
		
		for ibin in range(len(self.randoms)):
			
			if mask is None: mask_ = self.randoms[ibin] > 0.
			else: mask_ = mask & self.randoms[ibin] > 0.

			#mask_ &= self.data[ibin]>0
			mask_ &= utils.isnotnaninf(self.photo[ibin]).all(axis=-1)
			#range_photo = scipy.percentile(self.photo[ibin][mask_].T,[5.,95.],axis=1)
			#mask_ &= scipy.all((self.photo[ibin] >= range_photo[0]) & (self.photo[ibin] < range_photo[-1]),axis=-1)
			self.data[ibin] = self.data[ibin][mask_]
			self.randoms[ibin] = self.randoms[ibin][mask_]
			self.photo[ibin] = self.photo[ibin][mask_]
			self.alpha[ibin] = self.data[ibin].sum()/self.randoms[ibin].sum()
			self.invcovariance[ibin] = 1./(self.alpha[ibin]*self.randoms[ibin])
		
		self.alpha_global = scipy.sum(self.data)/scipy.sum(self.randoms)
		self.nobs = sum(map(len,self.data))

		if self.params['normalize_randoms']:
			for ibin in range(len(self.randoms)):
				self.logger.info('Normalizing randoms of bin {:d} by {:.4g}.'.format(ibin+1,self.alpha[ibin]))
				self.randoms[ibin] *= self.alpha[ibin]
	
	def interpolate_coeffs(self,result='values',interpolate='bin'):
	
		coeffs = self.get_ibin_bestfit(ibin=range(self.nbins),result=result)
		if interpolate == 'bin':
			interpolations = {par:utils.interpolate_bin(self.edges,coeffs[par]) for par in coeffs}
		elif interpolate == 'linear':
			if len(self.mean_values) < 2:
				self.logger.warning('There is just one bin, cannot use linear interpolation, switching to bin.')
				return self.interpolate_coeffs(result=result,interpolate='bin')
			interpolations = {par:utils.interpolate_linear(self.mean_values,coeffs[par]) for par in coeffs}
		elif interpolate == 'spline':
			if len(self.mean_values) < 2:
				self.logger.warning('There is just one bin, cannot use spline interpolation, switching to bin.')
				return self.interpolate_coeffs(result=result,interpolate='bin')
			interpolations = {par:utils.interpolate_spline(self.mean_values,coeffs[par]) for par in coeffs}
		else:
			raise ValueError('I do not know interpolation scheme {}.'.format(interpolate))
		return interpolations

	def clip(self,toret,clip=None,default=[0.5,1.5]):
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

	def flatten(self,toret):
		"""To renormalize density in each redshift slice."""
		mask_good = utils.isnotnaninf(toret)
		binned,_,binnumber = stats.binned_statistic(self.values[mask_good],values=toret[mask_good],statistic='mean',bins=self.edges)
		binnumber = scipy.clip(binnumber-1,0,self.nbins)
		toret /= binned[binnumber]

	def predicted_density(self,interpolate=None,extrapolate=None,clip=None,flatten=False):

		if interpolate is None: interpolate = self.params.get('interpolate','spline')
		if extrapolate is None: extrapolate = self.params.get('extrapolate','lim')
		
		if self.values is None:
			self.logger.info('Cannot use bins.')
			coeffs = scipy.array([self.bestfit['values'][key] for key in self.parameters])
			return self._unbinned_model_(self.photo[0],coeffs,ibin=None)
		
		interpolate = self.interpolate_coeffs(interpolate=interpolate)
		nslabs = max(len(self.values)//1000,1)
		slabs = [scipy.arange(islab*len(self.values)/nslabs,(islab+1)*len(self.values)/nslabs) for islab in range(nslabs)]
		toret = []
		for slab in slabs:
			coeffs = scipy.array([interpolate[par](self.values[slab],extrapolate=extrapolate) for par in self.unbinned_parameters]).T
			toret.append(self._unbinned_model_(self.photo[0][slab],coeffs,ibin=None))
		toret = scipy.concatenate(toret,axis=0)

		self.clip(toret,clip=clip,default=False)
		
		if flatten: self.flatten(toret)

		return toret

	def getstate(self):
		state = {}
		for key in ['params','alpha','alpha_global','mean_values','null','bestfit']:
			if hasattr(self,key): state[key] = getattr(self,key)
		return state

def _fit_healpix_(healpix):

	results = {survey:{} for survey in parameters['fit_photo']['survey_subsample']}
	
	edges = parameters['fit_photo'].get('edges',None)
	
	for survey in results:
	
		fit = FitDensity(params=parameters['fit_photo'])
		keys_randoms = '{}_{}'.format(parameters['fit_photo']['randoms'],survey)
		if edges is not None:
			keys_data = ['{}_{}_{:d}'.format(parameters['fit_photo']['data'],survey,ibin+1) for ibin in range(len(edges)-1)]
			#keys_photo = [['{}_{:d}'.format(field,ibin+1) if field in parameters['fit_photo']['params_bin'] else field for field in parameters['fit_photo']['params']] for ibin in range(len(edges)-1)]
			keys_photo = parameters['fit_photo']['params']
			mean_values = [healpix['meanz_{}'.format(key_data)][0] for key_data in keys_data]
		else:
			keys_data = ['{}_{}'.format(parameters['fit_photo']['data'],survey)]
			keys_photo = parameters['fit_photo']['params']
			mean_values = None
		fit.set_density(healpix,keys_data,mean_values=mean_values,keys_randoms=keys_randoms)
		fit.set_photo(healpix,keys_photo=keys_photo)
		fit.prepare()
		logger.info('Fitting survey {}.'.format(survey))
		fit.minimize()
		results[survey] = fit.getstate()
	
	return results

def fit_healpix():
	"""Fit the ndata/nrandoms = f(photometry)
	relation in healpix pixels.
	The fit result is saved in parameters['paths']['fit_photo'].

	"""
	logger.info(utils.log_header('Fitting pixels'))

	path_fit = parameters['paths']['fit_photo']
	path_healpix = parameters['paths']['healpix_density']

	healpix = ELGCatalogue.load(path_healpix)
	results = _fit_healpix_(healpix)
	
	utils.save(path_fit,results)

def _fit_binned_(binned):

	results = {survey:{} for survey in parameters['fit_photo']['survey_subsample']}
	
	edges = parameters['fit_photo'].get('edges',None)
	
	for survey in results:
	
		fit = FitDensity(params=parameters['fit_photo'])
		catalogue = binned[survey]
		
		keys_randoms = parameters['fit_photo']['randoms']
		if edges is not None:
			keys_data = ['{}_{:d}'.format(parameters['fit_photo']['data'],ibin+1) for ibin in range(len(edges)-1)]
			keys_photo = parameters['fit_photo']['params']
			mean_values = [catalogue['meanz_{}'.format(key_data)][0] for key_data in keys_data]
		else:
			keys_data = [parameters['fit_photo']['data']]
			keys_photo = parameters['fit_photo']['params']
			mean_values = None
		fit.set_density(catalogue,keys_data,mean_values=mean_values,keys_randoms=keys_randoms)
		fit.set_photo(catalogue,keys_photo=keys_photo)
		fit.prepare()
		logger.info('Fitting survey {}.'.format(survey))
		fit.minimize()
		results[survey] = fit.getstate()

	return results

def fit_binned():
	"""Fit the ndata/nrandoms = f(photometry)
	relation in bins of photometric parameters.
	The fit result is saved in parameters['paths']['fit_photo'].

	"""
	logger.info(utils.log_header('Fitting binned data'))

	path_binned = parameters['paths']['binned_density']
	path_fit = parameters['paths']['fit_photo']
	
	binned = {survey: ELGCatalogue.load(path_binned[survey]) for survey in path_binned}
	results = _fit_binned_(binned)
	
	utils.save(path_fit,results)

def get_density_alpha(results):
	alpha = {}
	for survey in results:
		fit = FitDensity.setstate(results[survey])
		alpha[survey] = fit.alpha_global
	mean_alpha = scipy.mean(alpha.values())
	for survey in alpha: alpha[survey] /= mean_alpha
	return alpha

def _add_healpix_weight_(healpix,results,clip=False):	

	alpha = get_density_alpha(results)
	update = utils.sorted_alpha_numerical_order(results.keys())

	for survey in update:

		fit = FitDensity.setstate(results[survey])
		fit.set_photo(healpix,parameters['fit_photo']['params'])
		predicted_density = fit.model(*[fit.bestfit['values'][key] for key in fit.parameters])
		#mask_survey = healpix['{}_{}'.format(parameters['fit_photo']['randoms'],survey)] > 0.
		mask_survey = healpix.trues()
		
		if fit.params.get('edges',None) is not None:
			for ibin in range(fit.nbins):
				key_weight = '{}_{}_{:d}'.format(parameters['fit_photo']['weight'],survey,ibin+1)
				healpix.add_field({'field':key_weight,'description':'Calculated weights','format':'float64'})
				healpix[key_weight][:] = healpix.default_value(key_weight)
				tmp = predicted_density[ibin][mask_survey]*alpha[survey]
				healpix[key_weight][mask_survey] = scipy.clip(tmp,*clip) if clip else tmp
				#mask_good = mask_survey & healpix.good_value(key_weight)
				mask_good = (healpix['{}_{}'.format(parameters['fit_photo']['randoms'],survey)] > 0.) & healpix.good_value(key_weight)
				logger.info('{} range in {}: {:.4f} - {:.4f}.'.format(key_weight,survey,healpix[key_weight][mask_good].min(),healpix[key_weight][mask_good].max()))
	
		key_weight = '{}_{}'.format(parameters['fit_photo']['weight'],survey)
		healpix.add_field({'field':key_weight,'description':'Calculated weights','format':'float64'})
		healpix[key_weight][:] = healpix.default_value(key_weight)
		tmp = scipy.average([predicted_density[ibin][mask_survey] for ibin in range(fit.nbins)],weights=fit.alpha,axis=0)*alpha[survey] # predicted density is the weighted average of predicted density in each redshift bin
		healpix[key_weight][mask_survey] = scipy.clip(tmp,*clip) if clip else tmp
		#mask_good = mask_survey & healpix.good_value(key_weight)
		mask_good = (healpix['{}_{}'.format(parameters['fit_photo']['randoms'],survey)] > 0.) & healpix.good_value(key_weight)
		logger.info('{} range in {}: {:.4f} - {:.4f}.'.format(key_weight,survey,healpix[key_weight][mask_good].min(),healpix[key_weight][mask_good].max()))
	
	return healpix

def add_healpix_weight(clip=False):
	"""Add healpix weights to healpix map parameters['paths']['healpix_density'],
	using the fit result in parameters['paths']['fit_photo'].

	Parameters
	----------
	clip : list, optional
		bounds to clip weights.

	"""
	logger.info(utils.log_header('Adding photometric weights to healpix'))

	path_fit = parameters['paths']['fit_photo']
	path_healpix = parameters['paths']['healpix_density']

	results = utils.load(path_fit)
	healpix = ELGCatalogue.load(path_healpix)

	healpix = _add_healpix_weight_(healpix,results,clip=clip)
	
	healpix.save(path_healpix)

	logger.info('Adding photometric weights to healpix completed.')
	

def _add_weight_healpix_(catalogue,healpix,key_weight_pixel,key_weight='WEIGHT_SYSTOT',key_redshift='Z',to_weight='data',mask=None):

	params_healpix = parameters['healpix']['params']
	edges = parameters['fit_photo'].get('edges',None)
	
	if mask is None: mask = catalogue.trues()
	catalogue.fill_default_value(key_weight,mask=mask)

	pix = catalogue.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)
	survey_subsample = parameters['fit_photo']['survey_subsample']

	for survey in survey_subsample:
		
		mask_survey = mask & catalogue.subsample(survey_subsample[survey])
		if edges is not None:
			for ibin,(low,up) in enumerate(zip(edges[:-1],edges[1:])):
				mask_survey_bin = mask_survey & (catalogue[key_redshift] >= low) & (catalogue[key_redshift] < up)
				catalogue[key_weight][mask_survey_bin] = utils.digitized_interp(pix[mask_survey_bin],healpix['hpind'],healpix['{}_{}_{:d}'.format(key_weight_pixel,survey,ibin+1)],fill=catalogue.default_value(key_weight))
		else:
			catalogue[key_weight][mask_survey] = utils.digitized_interp(pix[mask_survey],healpix['hpind'],healpix['{}_{}'.format(key_weight_pixel,survey)],fill=catalogue.default_value(key_weight))
		
		if to_weight == 'data': # weight on data is the inverse of the data predicted density
			catalogue[key_weight][mask_survey] = 1./catalogue[key_weight][mask_survey]
		
		mask_bad = mask_survey & catalogue.bad_value(key_weight)
		sum_bad = mask_bad.sum()
		mean_weight = scipy.mean(catalogue[key_weight][mask_survey & ~mask_bad])
		if sum_bad>0:
			logger.warning('{} has {:d} incorrect values in survey {}: {} in hp pixels {}. They are replaced by {:.4f}.'.format(key_weight,sum_bad,survey,catalogue[key_weight][mask_bad],scipy.unique(pix[mask_bad]),mean_weight))
		catalogue[key_weight][mask_bad] = mean_weight
	
	mask_bad = mask & catalogue.bad_value(key_weight)
	sum_bad = mask_bad.sum()
	mean_weight = scipy.mean(catalogue[key_weight][mask & ~mask_bad])
	if sum_bad>0:
		logger.warning('{} has {:d} incorrect values: {} in hp pixels {}. They are replaced by {:.4f}.'.format(key_weight,sum_bad,catalogue[key_weight][mask_bad],scipy.unique(pix[mask_bad]),mean_weight))
	catalogue[key_weight][mask_bad] = mean_weight
	catalogue[key_weight][mask] /= mean_weight
	

def add_weight_healpix(catalogue,key_weight_pixel,key_weight='WEIGHT_SYSTOT',key_redshift='Z',to_weight='data',mask=None):
	"""Add healpix weights to catalogue.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	key_weight_pixel : str
		the name of the weights to take from parameters['paths']['healpix_density'].
	key_weight : str, optional
		the catalogue field to save weights in.
	key_redshift : str, optional
		the catalogue field where redshifts are stored;
		used in the case weights are calculated in redshift slices.
	to_weight : str, optional
		whether to catalogue is data or randoms.
	mask : boolean array, optional
		veto mask to be applied before any calculation.

	"""
	
	logger.info('Importing healpix weight.')
	
	path_healpix = parameters['paths']['healpix_density']
	
	healpix = ELGCatalogue.load(path_healpix)
	_add_weight_healpix_(catalogue,healpix,key_weight_pixel,key_weight=key_weight,key_redshift=key_redshift,to_weight=to_weight,mask=mask)
	

def photo_from_catalogue_healpix(catalogue,healpix=None,atleast=[],replace=[],path_healpix=None,params_healpix={}):
	"""Create an ELGCatalogue instance by taking columns from catalogue (if column is available) or healpix.

	Parameters
	----------
	catalogue : ELGCatalogue
		catalogue.
	healpix : ELGCatalogue, optional
		healpix map.
	atleast : list, optional
		columns required in output.
	replace : list, optional
		catalogue columns replaced by healpix ones.
	path_healpix : str, optional
		path to healpix catalogue.
	params_healpix : dict, optional
		healpix parameter dict with keys (nside, nest).

	Returns
	-------
	photo : ELGCatalogue
		the ELGCatalogue containing set(atleast + replace) columns.

	"""
	indata = {field:ELGCatalogue.pixel_to_object(field) in catalogue.data for field in atleast}
	photo = ELGCatalogue(data={field:catalogue[ELGCatalogue.pixel_to_object(field)] for field in atleast if indata[field]})
	for field in ['RA','DEC']: photo[field] = catalogue[field]
	fields_healpix = replace + [field for field in indata if (not indata[field]) and (field not in replace)]
	if fields_healpix:
		if healpix is None: healpix = ELGCatalogue.load(path_healpix)
		if not params_healpix: params_healpix = parameters['healpix']['params']
		photo.from_healpix(healpix,fields=fields_healpix,fields_healpix=fields_healpix,**params_healpix)
	return photo


def _add_weight_object_(catalogue,results,key_weight='WEIGHT_SYSTOT',key_redshift='Z',to_weight='data',import_healpix=[],interpolate=None,extrapolate=None,clip=None,flatten=False,mask=None):
	
	if mask is None: mask = catalogue.trues()
	catalogue.fill_default_value(key_weight,mask=mask)
	
	photo = photo_from_catalogue_healpix(catalogue,healpix=None,atleast=parameters['fit_photo']['params'],replace=[],params_healpix=parameters['healpix']['params'],path_healpix=parameters['paths']['healpix_density'])
	alpha = get_density_alpha(results)
	survey_subsample = parameters['fit_photo']['survey_subsample']

	for survey in survey_subsample:

		fit = FitDensity.setstate(results[survey])
		mask_survey = mask & catalogue.subsample(fit.params['survey_subsample'][survey])
		fit.set_photo(photo[mask_survey],parameters['fit_photo']['params'],values=catalogue[key_redshift][mask_survey] if key_redshift in catalogue else None)
		catalogue[key_weight][mask_survey] = fit.predicted_density(interpolate=interpolate,extrapolate=extrapolate,clip=clip,flatten=flatten)*alpha[survey]
		
		if to_weight == 'data':
			catalogue[key_weight][mask_survey] = 1./catalogue[key_weight][mask_survey]
		
		mask_bad = mask_survey & catalogue.bad_value(key_weight)
		sum_bad = mask_bad.sum()
		mean_weight = scipy.mean(catalogue[key_weight][mask_survey & ~mask_bad])
		if sum_bad>0:
			logger.warning('{} has {:d} incorrect values in survey {}: {}. They are replaced by {:.4f}.'.format(key_weight,sum_bad,survey,catalogue[key_weight][mask_bad],mean_weight))
		catalogue[key_weight][mask_bad] = mean_weight
	
	mask_bad = mask & catalogue.bad_value(key_weight)
	sum_bad = mask_bad.sum()
	mean_weight = scipy.mean(catalogue[key_weight][mask & ~mask_bad])
	if sum_bad>0:
		logger.warning('{} has {:d} incorrect values: {}. They are replaced by {:.4f}.'.format(key_weight,sum_bad,catalogue[key_weight][mask_bad],mean_weight))
	catalogue[key_weight][mask_bad] = mean_weight
	catalogue[key_weight][mask] /= mean_weight

def add_weight_object(catalogue,key_weight='WEIGHT_SYSTOT',key_redshift='Z',to_weight='data',import_healpix=[],clip=None,interpolate='spline',extrapolate='lim',flatten=False,mask=None):
	"""Add fitted weights to catalogue, without using the healpix map weights.
	It uses per-object (instead of per-pixel) photometric attributes given in data catalogue.

	Parameters
	----------
	catalogue : ELGCatalogue
		data.
	key_weight : str
		the catalogue field to save weights in.
	key_redshift : str, optional
		the catalogue field where redshifts are stored;
		used in the case weights are calculated in redshift slices.
	to_weight : str, optional
		whether to catalogue is data or randoms.
	import_healpix : list, optional
		list of photometric parameters to import from healpix map.
	clip : list, optional
		bounds to clip weights.
	interpolate : str, optional
		the interpolation scheme between redshift slices.
	extrapolate : str, optional
		the extrapolation scheme outside of the range of redshift slices.
	flatten : bool, optional
		whether to renormalize weights in each redshift slice.
	mask : boolean array, optional
		veto mask to be applied before any calculation.

	"""
	path_fit = parameters['paths']['fit_photo']

	logger.info('Importing object weights.')

	results = utils.load(path_fit)
	_add_weight_object_(catalogue,results,key_weight=key_weight,key_redshift=key_redshift,to_weight=to_weight,import_healpix=import_healpix,clip=clip,interpolate=interpolate,extrapolate=extrapolate,flatten=flatten,mask=mask)

###############################################################################
# Others
###############################################################################

def add_maps_values(catalogue,healpix=False):
	"""Add EBV, NHI to catalogue.

	Parameters
	----------
	catalogue : ELGCatalogue
		catalogue.
	healpix : bool, optional
		whether catalogue is an healpix map.

	"""
	logger.info('Importing maps values.')

	pathEBV_sfd = parameters['paths'].get('EBV_sfd',None)
	pathTemp_sfd = parameters['paths'].get('temp_sfd',None)
	pathEBV_Lenz = parameters['paths'].get('EBV_Lenz',None)
	pathNHI = parameters['paths'].get('NHI',None)
	
	fields = {field:field for field in ['ra','dec','ebv_sfd','logtemp_sfd','ebv_lenz','lognh1','debv']}
	if healpix:
		for field in fields: fields[field] = ELGCatalogue.object_to_pixel(field)
	
	### E(B-V) ###
	if any([pathEBV_sfd,pathTemp_sfd,pathEBV_Lenz,pathNHI]):
		from astropy.coordinates import SkyCoord
		from dustmaps.healpix_map import HEALPixFITSQuery
		radec = SkyCoord(ra=catalogue[fields['ra']],dec=catalogue[fields['dec']],unit='deg',frame='icrs')
		if pathEBV_sfd is not None:
			from dustmaps.sfd import SFDQuery
			sfd = SFDQuery(map_dir=pathEBV_sfd)
			catalogue[fields['ebv_sfd']] = sfd(radec)
			logger.info('Importing {}.'.format(fields['ebv_sfd']))
		if pathTemp_sfd is not None:
			sfd = HEALPixFITSQuery(pathTemp_sfd,'galactic',hdu=1,field='TEMPERATURE')
			catalogue[fields['logtemp_sfd']] = scipy.log10(sfd(radec))
			logger.info('Importing {}.'.format(fields['logtemp_sfd']))
		if pathEBV_Lenz is not None:
			lenz = HEALPixFITSQuery(pathEBV_Lenz,'galactic',hdu=1,field='EBV')
			catalogue[fields['ebv_lenz']] = lenz(radec)
			logger.info('Importing {}.'.format(fields['ebv_lenz']))
		if pathNHI is not None:
			NHI = HEALPixFITSQuery(pathNHI,'galactic',hdu=1,field='NHI')
			catalogue[fields['lognh1']] = scipy.log10(scipy.absolute(NHI(radec))) # negative values of NHI are outside the ELG footprint
			logger.info('Importing {}.'.format(fields['lognh1']))
	
	if all(fields[field] in catalogue.data for field in ['ebv_sfd','ebv_lenz']):
		catalogue[fields['debv']] = catalogue[fields['ebv_lenz']] - catalogue[fields['ebv_sfd']]
		mask_bad = catalogue.bad_value(fields['debv'])
		catalogue[fields['debv']][mask_bad] = 0.
	
	
def add_healpix_photometry(healpix):
	"""Add PS1stars, EBV, NHI, airmass or skyflux to healpix.

	Parameters
	----------
	healpix : ELGCatalogue
		healpix map.

	"""
	import healpy

	logger.info('Adding photometric parameters to healpix.')

	params_healpix = parameters['healpix']['params']
	
	path_ccd = parameters['paths'].get('CCD',None)
	pathPS1_star = parameters['paths'].get('PS1_star',None)
	pathGaia_star = parameters['paths'].get('Gaia_star',None)

	if path_ccd is not None:
		### Loading data ###
		data = ELGCatalogue.load(path_ccd)
		pixel_data = data.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)

		### Adds photometric values ###
		replace_pixel = ['hpairmass','hpskyflux']
		replace_object = ['airmass','ccdskymag']
		for key_pixel,key_object in zip(replace_pixel,replace_object):
			tmp = data[key_object]
			if 'mag' in key_object: tmp = utils.mag_to_flux(data[key_object])
			healpix[key_pixel] = utils.interp_digitized_statistics(healpix['hpind'],pixel_data,fill=healpix.default_value(key_pixel),values=tmp,statistic='mean')
	
	### Stars ###
	for path_star,key_pixel in zip([pathPS1_star,pathGaia_star],['hpps1stardens','hpgaiastardens']):
		if path_star is not None:
			data = ELGCatalogue.load(path_star)
			pixel_data = data.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)
			healpix[key_pixel] = utils.interp_digitized_statistics(healpix['hpind'],pixel_data,fill=healpix.default_value(key_pixel))/healpy.nside2pixarea(params_healpix['nside'],degrees=True)

	### E(B-V) ###
	add_maps_values(healpix,healpix=True)

def make_healpix(path_data,path_randoms,path_data_model=None):
	"""Make the healpix map from scratch.
	Can import photometric parameters from data and randoms.
	The file data model can be saved to path_data_model if provided.

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the random catalogue.
	path_data_model : str, optional
		if provided, save data model in path_data_model.

	"""
	import healpy

	logger.info(utils.log_header('Making healpix'))

	params_healpix = parameters['healpix']['params']
	path_healpix = parameters['paths']['healpix_density']
	
	### New ELGCatalogue ###
	size = healpy.nside2npix(params_healpix['nside'])
	healpix = ELGCatalogue(header=parameters['healpix']['header'],datamodel=parameters['healpix']['data_model'],size=size)
	
	healpix['hpind'] = scipy.arange(size)
	theta_rad,phi_rad = healpy.pix2ang(params_healpix['nside'],healpix['hpind'],nest=params_healpix['nest'])
	healpix['hpra'],healpix['hpdec'] = utils.healpix_to_radec(theta_rad,phi_rad,degree=True)
	
	### Importing photometric maps ###
	add_healpix_photometry(healpix)
	
	### Loading data ###
	data = ELGCatalogue.load(path_data)
	mask_data = data.subsample(parameters['target_subsample'])
	pixel_data = data.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)

	### Loading randoms ###
	randoms = ELGCatalogue.load(path_randoms)
	mask_randoms = randoms.subsample(parameters['randoms_target_subsample'])
	pixel_randoms = randoms.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)

	### From data ###
	replace_pixel = ['hpnobs_{}'.format(b) for b in ELGCatalogue.LIST_BAND]
	replace_object = map(ELGCatalogue.pixel_to_object,replace_pixel)
	for key_pixel,key_object in zip(replace_pixel,replace_object):
		mask_good = mask_data & data.good_value(key_object)
		healpix[key_pixel] = utils.interp_digitized_statistics(healpix['hpind'],pixel_data[mask_good],fill=healpix.default_value(key_pixel),values=data[key_object][mask_good],statistic='mean')

	### From randoms ###
	replace_pixel = ['hppsfsize_{}'.format(b) for b in ELGCatalogue.LIST_BAND] + ['hpgaldepth_{}'.format(b) for b in ELGCatalogue.LIST_BAND] + ['hppsfdepth_{}'.format(b) for b in ELGCatalogue.LIST_BAND]
	replace_object = map(ELGCatalogue.pixel_to_object,replace_pixel)
	for key_pixel,key_object in zip(replace_pixel,replace_object):
		mask_good = mask_randoms & randoms.good_value(key_object)
		healpix[key_pixel] = utils.interp_digitized_statistics(healpix['hpind'],pixel_randoms[mask_good],fill=healpix.default_value(key_pixel),values=randoms[key_object][mask_good],statistic='mean')
	healpix['hpdecalsdens'] = healpix['hpndet_all']*1./utils.interp_digitized_statistics(healpix['hpind'],pixel_randoms,fill=healpix.default_value('hpdecalsdens'))*parameters['density']['density_randoms']

	for key in parameters['healpix']['fields_photo']:
		mask_bad = healpix.bad_value(key)
		healpix[key][mask_bad] = healpix.default_value(key)

	for survey in ELGCatalogue.LIST_CHUNK + ELGCatalogue.LIST_CAP:
		if survey in healpix:
			mask_survey = mask_randoms & randoms.survey(survey)
			healpix[survey][:] = False
			healpix[survey][scipy.in1d(healpix['hpind'],pixel_randoms[mask_survey])] = True

	### Writing fits ###
	healpix.save(path_healpix,path_data_model)

	logger.info('Making healpix completed.')


def to_standard_healpix_format(path_data,path_randoms,path_data_model=None):
	"""Make the healpix map from Anand's parameters['paths']['healpix'].
	Can import photometric parameters from data and randoms.
	The file data model can be saved to path_data_model if provided.

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the random catalogue.
	path_data_model : str, optional
		if provided, save data model in path_data_model.

	"""

	logger.info(utils.log_header('Exporting to standard healpix format'))

	params_healpix = parameters['healpix']['params']
	path_healpix = parameters['paths']['healpix_density']

	### Open input fits ###
	healpix = ELGCatalogue.load(parameters['paths']['healpix'])

	### Making columns with type, unit ###
	healpix.set_header(parameters['healpix']['header'])
	healpix.set_datamodel(parameters['healpix']['data_model'])
	
	### Importing photometric maps ###
	add_healpix_photometry(healpix)
	
	### Loading data ###
	data = ELGCatalogue.load(path_data)
	mask_data = data.subsample(parameters['target_subsample'])
	pixel_data = data.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)

	### Loading randoms ###
	randoms = ELGCatalogue.load(path_randoms)
	mask_randoms = randoms.subsample(parameters['randoms_target_subsample'])
	pixel_randoms = randoms.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)

	### Some additionnal conversion or precomputation stuff ###
	replace_pixel = ['hpnobs_{}'.format(b) for b in ELGCatalogue.LIST_BAND]
	replace_object = map(ELGCatalogue.pixel_to_object,replace_pixel)
	for key_pixel,key_object in zip(replace_pixel,replace_object):
		mask_good = mask_data & data.good_value(key_object)
		healpix[key_pixel] = utils.interp_digitized_statistics(healpix['hpind'],pixel_data[mask_good],fill=healpix.default_value(key_pixel),values=data[key_object][mask_good],statistic='median')
	
	replace_pixel = []
	replace_object = map(ELGCatalogue.pixel_to_object,replace_pixel)
	for key_pixel,key_object in zip(replace_pixel,replace_object):
		mask_good = mask_randoms & randoms.good_value(key_object)
		healpix[key_pixel] = utils.interp_digitized_statistics(healpix['hpind'],pixel_randoms[mask_good],fill=healpix.default_value(key_pixel),values=randoms[key_object][mask_good],statistic='mean')

	replace_pixel = ['hppsfsize_{}'.format(b) for b in ELGCatalogue.LIST_BAND] + ['hpgaldepth_{}'.format(b) for b in ELGCatalogue.LIST_BAND] + ['hppsfdepth_{}'.format(b) for b in ELGCatalogue.LIST_BAND]
	replace_object = map(ELGCatalogue.pixel_to_object,replace_pixel)
	for key_pixel,key_object in zip(replace_pixel,replace_object):
		mask_bad = healpix.bad_value(key_pixel) # replace only bad pixel values
		#tmp = healpix[key_pixel].copy()
		mask_good = mask_randoms & randoms.good_value(key_object) & scipy.in1d(pixel_randoms,healpix['hpind'][mask_bad])
		if mask_good.sum() > 0:
			logger.info('Filling {:d} pixels with {} from randoms.'.format(scipy.unique(pixel_randoms[mask_good]).size,key_pixel))
			healpix[key_pixel][mask_bad] = utils.interp_digitized_statistics(healpix['hpind'][mask_bad],pixel_randoms[mask_good],fill=healpix.default_value(key_pixel),values=randoms[key_object][mask_good],statistic='median')
			#print healpix[key_pixel][(tmp != healpix[key_pixel]) & ~(scipy.isnan(tmp) | scipy.isnan(healpix[key_pixel]))]
	healpix['hpdecalsdens'] = healpix['hpndet_all']*1./utils.interp_digitized_statistics(healpix['hpind'],pixel_randoms,fill=healpix.default_value('hpdecalsdens'))*parameters['density']['density_randoms']

	for key in parameters['healpix']['fields_photo']:
		mask_bad = healpix.bad_value(key)
		healpix[key][mask_bad] = healpix.default_value(key)

	for survey in ELGCatalogue.LIST_CHUNK + ELGCatalogue.LIST_CAP:
		if survey in healpix:
			mask_survey = mask_randoms & randoms.survey(survey)
			healpix[survey][:] = False
			healpix[survey][scipy.in1d(healpix['hpind'],pixel_randoms[mask_survey])] = True

	### Writing fits ###
	healpix.save(path_healpix,path_data_model)

	logger.info('Exporting to standard healpix format completed.')


def _add_healpix_density_(healpix,data,randoms,suffix='photo',key_redshift='Z',mask_data=None,mask_randoms=None,update=None):
	
	params_healpix = parameters['healpix']['params']
	
	if mask_data is None: mask_data = data.trues()
	if mask_randoms is None: mask_randoms = randoms.trues()
	
	if suffix == 'photo': mask_data = mask_data & data.subsample(parameters['target_subsample'])
	if suffix == 'spectro': mask_data = mask_data & data.all_data_cuts(parameters,exclude=[key_redshift])	
	pixel_data = data.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)

	if suffix == 'photo': mask_randoms = mask_randoms & randoms.subsample(parameters['randoms_target_subsample'])
	if suffix == 'spectro': mask_randoms = mask_randoms & randoms.all_randoms_cuts(parameters,exclude=[key_redshift])
	pixel_randoms = randoms.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)
	
	survey_subsample = parameters['fit_photo']['survey_subsample']
	if update is None: update = utils.sorted_alpha_numerical_order(survey_subsample.keys())
	edges = parameters['fit_photo'].get('edges',None)
	
	for survey in update:
	
		logger.info('Processing survey {}.'.format(survey))
		
		key_randoms,key_data = 'randoms_{}_{}'.format(suffix,survey),'data_{}_{}'.format(suffix,survey)
		mask_data_survey = mask_data & data.subsample(survey_subsample[survey]) & data.subsample(parameters['fit_photo']['data_subsample'])
		mask_randoms_survey = mask_randoms & randoms.subsample(survey_subsample[survey]) & randoms.subsample(parameters['fit_photo']['randoms_subsample'])
		if not mask_randoms_survey.any():
			logger.warning(' No object in {}.'.format(survey))
		
		logger.info('Filling data and randoms fields {} and {}.'.format(key_data,key_randoms))
		if suffix == 'spectro':
			weight = data.weight_object/data['WEIGHT_SYSTOT']
			if edges is not None:
				for ibin,(low,up) in enumerate(zip(edges[:-1],edges[1:])):
					key_data_bin = '{}_{:d}'.format(key_data,ibin+1)
					key_meanz = 'meanz_{}'.format(key_data_bin)
					mask_data_bin = mask_data_survey & (data[key_redshift] >= low) & (data[key_redshift] < up)
					healpix.add_field({'field':key_data_bin,'description':'Spectroscopic density','format':'float64'})
					healpix[key_data_bin] = utils.interp_digitized_statistics(healpix['hpind'],pixel_data[mask_data_bin],fill=0.,values=weight[mask_data_bin])
					healpix.add_field({'field':key_meanz,'description':'Mean redshift','format':'float64'})
					healpix[key_meanz][:] = scipy.mean(data[key_redshift][mask_data_bin])
					
			healpix.add_field({'field':key_data,'description':'Spectroscopic density','format':'float64'})
			healpix[key_data] = utils.interp_digitized_statistics(healpix['hpind'],pixel_data[mask_data_survey],fill=0.,values=weight[mask_data_survey])
			healpix.add_field({'field':key_randoms,'description':'Spectroscopic randoms','format':'float64'})
			healpix[key_randoms] = utils.interp_digitized_statistics(healpix['hpind'],pixel_randoms[mask_randoms_survey],fill=0.,values=randoms['COMP_BOSS'][mask_randoms_survey])

		else: #photometric targets
			healpix.add_field({'field':key_data,'description':'Photometric density','format':'float64'})
			healpix.add_field({'field':key_randoms,'description':'Photometric randoms','format':'float64'})
			healpix[key_data] = utils.interp_digitized_statistics(healpix['hpind'],pixel_data[mask_data_survey],fill=0.)
			healpix[key_randoms] = utils.interp_digitized_statistics(healpix['hpind'],pixel_randoms[mask_randoms_survey],fill=0.)
	
	return healpix


def add_healpix_density(path_data,path_randoms,suffix='photo',key_redshift='Z',update=None):
	"""Add ndata/nrandoms in each healpix pixel of parameters['paths']['healpix_density'].

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the random catalogue.
	suffix : str
		in ['photo','spectro']; whether to consider photometric targets or spectroscopic objects.
	key_redshift : str, optional
		the catalogue field where redshifts are stored;
		used in the case weights are calculated in redshift slices.
	update : list, optional
		list of surveys to update.

	"""

	logger.info(utils.log_header('Adding {} density to healpix'.format(suffix)))

	path_healpix = parameters['paths']['healpix_density']

	### Loading data ###
	data = ELGCatalogue.load(path_data)

	### Loading randoms ###
	randoms = ELGCatalogue.load(path_randoms)
	
	### Loading healpix ###
	healpix = ELGCatalogue.load(path_healpix)
	
	healpix = _add_healpix_density_(healpix,data,randoms,suffix=suffix,key_redshift=key_redshift,update=update)

	### Writing fits ###
	healpix.save(path_healpix)

	logger.info('Adding healpix density completed.')


def add_heapix_external_weights(path_weights,key_pixel_index='hpind',key_weight='hpweight',key_weight_new='hpweight',list_survey=None,key_weight_fill=None,invert=False,nside_in=None,nest_in=None,description='External weights'):
	"""Import healpix weights from external source path_weights.

	Parameters
	----------
	path_weights : str
		path to the external healpix map; the healpix parameters (nside, nest)
		should be the same as parameters['paths']['healpix_density'].
	key_index_pixel : str, optional
		field to the healpix index.
	key_weight : str, dict, optional
		field to the weights to import; if dict, each key is a survey of parameters['fit_photo']['list_survey'].
	key_weight_new : str, optional
		new base name of the weights; full name is '{}_{}'.format(key_weight_new,survey) for survey in survey_subsample.
	list_survey : list, optional
		the surveys for which '{}_{}'.format(key_weight_new,survey) is saved.
	key_weight_fill : str, optional
		fill other surveys with '{}_{}'.format(key_weight_fill,survey) (must be already saved).
	invert : bool, optional
		invert weights to obtain weights to be applied on randoms (i.e. prop to data density).
	nside_in : int, optional
		input nside used.
	nest_in : bool, optional
		if convert, whether nested scheme is used in input.
	description : str, optional
		description for data model.

	"""
	import healpy

	healpix = ELGCatalogue.load(parameters['paths']['healpix_density'])
	params_healpix = parameters['healpix']['params']
	weights = ELGCatalogue.load(path_weights)
	list_survey_all = utils.sorted_alpha_numerical_order(parameters['fit_photo']['survey_subsample'].keys())
	if list_survey is None: list_survey = list_survey_all

	nside_out = params_healpix['nside']
	nest_out = params_healpix['nest']
	if nside_in is None:
		try: nside_in = healpy.npix2nside(len(hpind))
		except: nside_in = nside_out
	if nest_in is None: nest_in = nest_out
	logger.info('Input parameters found: (nside,nest) = ({},{}).'.format(nside_in,nest_in))

	def convert_map(hpind,values,fill=0.):
		newhpind = scipy.arange(healpy.nside2npix(nside_in))
		values = utils.digitized_interp(newhpind,hpind,values,fill=fill)
		order_in = 'NESTED' if nest_in else 'RING'
		order_out = 'NESTED' if nest_out else 'RING'
		newvalues = healpy.ud_grade(values,nside_out=nside_out,pess=False,order_in=order_in,order_out=order_out,power=None,dtype=None)
		newhpind = scipy.arange(healpy.nside2npix(nside_out))
		return newhpind,newvalues
	
	for survey in list_survey_all:
		if survey in list_survey:
			if isinstance(key_weight,dict): key_weight_survey = key_weight[survey]
			else: key_weight_survey = key_weight
		elif  key_weight_fill:
			key_weight_survey = '{}_{}'.format(key_weight_fill,survey)
			logger.info('Using default weight {} in {}.'.format(key_weight_survey,survey))
		else:
			continue
		key_weight_new_survey = '{}_{}'.format(key_weight_new,survey)
		healpix.add_field({'field':key_weight_new_survey,'description':description,'format':'float64'})
		logger.info('Adding {} (provided: {}) to healpix map.'.format(key_weight_new_survey,key_weight_survey))
		if survey in list_survey:
			hpind,values = convert_map(weights[key_pixel_index],weights[key_weight_survey],fill=healpix.default_value(key_weight_new_survey))
			if invert: values = 1./values
			healpix[key_weight_new_survey] = utils.digitized_interp(healpix['hpind'],hpind,values,fill=healpix.default_value(key_weight_new_survey))
			mask_bad = healpix.bad_value(key_weight_new_survey)
			healpix[key_weight_new_survey][mask_bad] = healpix.default_value(key_weight_new_survey)
		else:
			healpix[key_weight_new_survey] = healpix[key_weight_survey]
			
	### Writing fits ###
	healpix.save(parameters['paths']['healpix_density'])

	logger.info('Adding healpix external weights completed.')	

def add_healpix_values(catalogue):
	"""Import healpix (pixel-averaged) photometric parameters into catalogue.

	Parameters
	----------
	catalogue : ELGCatalogue
		data/randoms.

	"""
	
	logger.info('Importing healpix values.')

	params_healpix = parameters['healpix']['params']
	path_healpix = parameters['paths']['healpix_density']
	keys_healpix = parameters['healpix']['fields_photo']

	pix = catalogue.pixelize(key_ra='RA',key_dec='DEC',degree=True,**params_healpix)

	healpix = ELGCatalogue.load(path_healpix)

	### Adding hp values to data catalogue ###
	for key_pixel in keys_healpix:
		if key_pixel in catalogue:
			logger.info('Updating field {}.'.format(key_pixel))
			catalogue[key_pixel] = utils.digitized_interp(pix,healpix['hpind'],healpix[key_pixel],fill=healpix.default_value(key_pixel))
	#catalogue['hpind'] = pix

	### Replacing data bad values by their hp value ###
	for key_pixel in keys_healpix:
		key_object = ELGCatalogue.pixel_to_object(key_pixel)
		if key_object in catalogue:
			logger.debug('Checking field {}.'.format(key_object))
			mask_bad = catalogue.bad_value(key_object)
			sum_bad = mask_bad.sum()
			if sum_bad>0: logger.warning('There are {:d} incorrect ({}) values for parameter {} in the data; they are replaced by healpix values.'.format(sum_bad,ELGCatalogue.bad_values(key_object),key_object))
			fill = catalogue.default_value(key_object)
			mask_good = healpix.good_value(key_pixel)
			if isinstance(catalogue[key_object][0],scipy.integer):
				catalogue[key_object][mask_bad] = utils.digitized_interp(pix[mask_bad],healpix['hpind'][mask_good],scipy.rint(healpix[key_pixel][mask_good]).astype(catalogue[key_object].dtype),fill=fill)
			else:
				catalogue[key_object][mask_bad] = utils.digitized_interp(pix[mask_bad],healpix['hpind'][mask_good],healpix[key_pixel][mask_good].astype(catalogue[key_object].dtype),fill=fill)


def _make_binned_density_(data,randoms,import_healpix=[],suffix='photo',key_redshift='Z',update=None):
	
	from binned_statistic import BinnedStatistic
	
	params_binned = parameters['binned']['params']

	if suffix == 'spectro': data = data[data.all_data_cuts(parameters,exclude=[key_redshift])]	
	if suffix == 'spectro': randoms = randoms[randoms.all_randoms_cuts(parameters,exclude=[key_redshift])]
	
	photo_data = photo_from_catalogue_healpix(data,healpix=None,atleast=parameters['fit_photo']['params'],replace=import_healpix,params_healpix=parameters['healpix']['params'],path_healpix=parameters['paths']['healpix_density'])
	photo_randoms = photo_from_catalogue_healpix(randoms,healpix=None,atleast=parameters['fit_photo']['params'],replace=import_healpix,params_healpix=parameters['healpix']['params'],path_healpix=parameters['paths']['healpix_density'])
	
	survey_subsample = parameters['fit_photo']['survey_subsample']
	if update is None: update = utils.sorted_alpha_numerical_order(survey_subsample.keys())
	edges = parameters['fit_photo'].get('edges',None)
	
	mask_data = data.subsample(parameters['fit_photo']['data_subsample'])
	mask_randoms = randoms.subsample(parameters['fit_photo']['randoms_subsample'])
	for field in parameters['fit_photo']['params']:
		mask_data &= utils.isnotnaninf(photo_data[field])
		mask_randoms &= utils.isnotnaninf(photo_randoms[field])
	
	catalogues_binned = {}
	
	for survey in update:
	
		logger.info('Processing survey {}.'.format(survey))
		
		mask_data_survey = mask_data & data.subsample(survey_subsample[survey])
		mask_randoms_survey = mask_randoms & randoms.subsample(survey_subsample[survey])
		if not mask_randoms_survey.any():
			logger.warning('No object in {}.'.format(survey))
		
		samples = scipy.array([photo_randoms[field][mask_randoms_survey] for field in parameters['fit_photo']['params']])
		values = randoms['COMP_BOSS'][mask_randoms_survey] if suffix == 'spectro' else None
		
		binned = BinnedStatistic(samples,values=values,**params_binned)
		catalogue = ELGCatalogue(header=parameters['binned']['header'],datamodel=parameters['binned']['data_model'],data={field:photo for field,photo in zip(parameters['fit_photo']['params'],binned.samples)},size=binned.size)
		
		key_randoms,key_data = 'randoms_{}'.format(suffix),'data_{}'.format(suffix)
		catalogue[key_randoms] = binned.values
		
		logger.info('Filling data and randoms fields {} and {}.'.format(key_data,key_randoms))
		if suffix == 'spectro':
			weight = data.weight_object/data['WEIGHT_SYSTOT']
			if edges is not None:
				for ibin,(low,up) in enumerate(zip(edges[:-1],edges[1:])):
					key_data_bin = '{}_{:d}'.format(key_data,ibin+1)
					key_meanz = 'meanz_{}'.format(key_data_bin)
					mask_data_bin = mask_data_survey & (data[key_redshift] >= low) & (data[key_redshift] < up)
					catalogue.add_field({'field':key_data_bin,'description':'Spectroscopic data','format':'float64'})
					samples = scipy.array([photo_data[field][mask_data_bin] for field in parameters['fit_photo']['params']])
					values = weight[mask_data_bin]
					catalogue[key_data_bin] = binned(samples,values)
					catalogue.add_field({'field':key_meanz,'description':'Mean redshift','format':'float64'})
					catalogue[key_meanz][:] = scipy.mean(data[key_redshift][mask_data_bin])
					
			catalogue.add_field({'field':key_data,'description':'Spectroscopic density','format':'float64'})
			samples = scipy.array([photo_data[field][mask_data_survey] for field in parameters['fit_photo']['params']])
			values = weight[mask_data_survey]
			catalogue[key_data] = binned(samples,values)
			catalogue.add_field({'field':key_randoms,'description':'Spectroscopic randoms','format':'float64'})

		else: #photometric targets
			catalogue.add_field({'field':key_data,'description':'Photometric density','format':'float64'})
			samples = scipy.array([photo_data[field][mask_data_survey] for field in parameters['fit_photo']['params']])
			catalogue[key_data] = binned(samples,None)
			catalogue.add_field({'field':key_randoms,'description':'Photometric randoms','format':'float64'})

		catalogues_binned[survey] = catalogue
	
	return catalogues_binned

def make_binned_density(path_data,path_randoms,import_healpix=[],suffix='photo',key_redshift='Z',update=None):
	
	"""Save binned density parameters['paths']['binned_density'][survey].

	Parameters
	----------
	path_data : str
		path to the data catalogue.
	path_randoms : str
		path to the random catalogue.
	import_healpix : list
		list of photometric parameters to import from healpix map.
	suffix : str
		in ['photo','spectro']; whether to consider photometric targets or spectroscopic objects.
	key_redshift : str, optional
		the catalogue field where redshifts are stored;
		used in the case weights are calculated in redshift slices.
	update : list, optional
		list of surveys to update.

	"""
	
	logger.info(utils.log_header('Adding {} density to binned statistic'.format(suffix)))

	path_binned = parameters['paths']['binned_density']

	### Loading data ###
	data = ELGCatalogue.load(path_data)

	### Loading randoms ###
	randoms = ELGCatalogue.load(path_randoms)
	
	binned = _make_binned_density_(data,randoms,suffix=suffix,key_redshift=key_redshift,update=update)

	### Writing fits ###
	for survey in binned: binned[survey].save(path_binned[survey])

	logger.info('Making binned density completed.')
