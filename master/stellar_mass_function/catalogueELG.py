import scipy
import numpy
import logging
from catalogue import Catalogue,Header,DataModel
import utils

class ELGCatalogue(Catalogue):
	"""Class for ELG catalogues.
	Implements useful ELG-specific methods, such as survey, weight_object, etc.

	"""
	logger = logging.getLogger('ELGCatalogue')

	LIST_CHUNK = ['eboss21','eboss22','eboss23','eboss25']
	LIST_CAP = ['NGC','SGC']
	LIST_BAND = ['g','r','z']
	LIST_NON_ZERO = ['ebv_sfd','ebv_lenz','depth','size','decalsdens','stardens','nobs','weight','WEIGHT','SN_MEDIAN_ALL']
	OFFSET_SECTOR = 1000
	OFFSET_INGROUP = 100000

#	def __init__(self,*args,**kwargs):
#		super(ELGCatalogue,self).__init__(*args,**kwargs)
	
	@classmethod
	def survey_to_chunk(cls,survey):
		if survey == 'NGC':
			return ['eboss23','eboss25']
		if survey == 'SGC':
			return ['eboss21','eboss22']
		if survey == 'ALL':
			return cls.LIST_CHUNK
		if survey in cls.LIST_CHUNK:
			return [survey]

	@classmethod
	def chunk_to_cap(cls,chunk):
		if chunk in ['eboss21','eboss22','SGC']:
			return 'SGC'
		if chunk in ['eboss23','eboss25','NGC']:
			return 'NGC'
	
	@classmethod
	def survey_to_cap(cls,survey):
		if survey in ['eboss21','eboss22']:
			return ['SGC']
		if survey in ['eboss23','eboss25']:
			return ['NGC']
		if survey == 'ALL':
			return ['NGC','SGC']
		return [survey]
	
	def survey(self,survey):
		"""Return survey mask."""
		self.logger.info('Survey: {}.'.format(survey))
		if survey in ['ALL'] + self.__class__.LIST_CAP + self.__class__.LIST_CHUNK:
			return scipy.any([self['chunk'] == chunk for chunk in self.__class__.survey_to_chunk(survey)],axis=0)
		self.logger.warning('Survey {} not available. All objects are considered.'.format(survey))
		return self.trues()
		
	def selection(self,cut,key,txt='selection'):
		"""Compute mask related to the selection cut.

		Parameters
		----------
		cut : list, tuple str, bool, float, int
			the selection;
			- if list, key == cut[i]
			- if tuple, select cut[0] <= self[key] < cut[1];
			- if str; if starts with !, then self[key] != cut[1:].
			- else: select self[key] == cut.
		key : str
			field in self
		txt : str
			keyword to print in logger

		Returns
		-------
		mask : boolean array
			mask

		"""	
		if key not in self.data:
			self.logger.warning('Column {} does not exist. Cut not applied.'.format(key))
			return self.trues()
		if isinstance(cut,list) and isinstance(cut[0],(str,unicode)):
			if cut[0].startswith('bit'):
				if cut[0].endswith('!'):
					mask = self.trues()
					self.logger.info('{}: {} == ~ ({})'.format(txt,key,' | '.join([str(c) for c in cut[1:]])))
					for bit in cut[1:]:
						if bit == -scipy.inf:
							mask &= self[key] != 0
						else:
							mask &= (self[key] & 2**bit) == 0
				else:
					mask = self.falses()
					self.logger.info('{}: {} == {}'.format(txt,key,' | '.join([str(c) for c in cut[1:]])))
					for bit in cut[1:]: mask |= (self[key] & 2**bit) > 0
			else:
				if cut[0].endswith('!'):
					mask = self.trues()
					self.logger.info('{}: {} == ~ ({})'.format(txt,key,' | '.join([str(c) for c in cut[1:]])))
					for bit in cut[1:]: mask &= self[key] != bit
				elif cut[0] == '&':
					mask = self.trues()
					self.logger.info('{}: {} == {}'.format(txt,key,' & '.join([str(c) for c in cut[1:]])))
					for bit in cut[1:]: mask &= self[key] == bit
				else:
					mask = self.falses()
					self.logger.info('{}: {} == {}'.format(txt,key,' | '.join([str(c) for c in cut[1:]])))
					for bit in cut[1:]: mask |= self[key] == bit
		elif isinstance(cut,(list,tuple)):
			self.logger.info('{}: {:.4g} <= {} < {:.4g}'.format(txt,cut[0],key,cut[1]))
			if self.bad_value(key).all():
				self.logger.warning('Column {} is filled with incorrect ({}) values. Cut not applied.'.format(key,self.bad_values(key)))
				return self.trues()
			mask = (self[key] >= cut[0]) & (self[key] < cut[1])
		elif isinstance(cut,str):
			self.logger.info('{}: {} == {}'.format(txt,key,str(cut)))
			if cut.startswith('!'): mask = (self[key] != cut[1:])
			else: mask = (self[key] == cut)
		else:
			self.logger.info('{}: {} == {}'.format(txt,key,str(cut)))
			mask = (self[key] == cut)
		#self.logger.info('Selecting {:d}/{:d} objects.'.format(mask.sum(),len(mask)))
		return mask
		
	def subsample(self,subsample,txt='subsample'):
		"""Loop over the different cuts in the dictionary subsample."""
		mask = self.trues()
		for key in subsample:
			if key == 'survey': mask &= self.survey(subsample[key])
			else: mask &= self.selection(subsample[key],key=key,txt=txt)
		self.logger.info('Subsample with {:d}/{:d} objects.'.format(mask.sum(),len(mask)))
		return mask

	def all_data_cuts(self,parameters,exclude=[],exclude_subsample=[]):
		"""Apply all data cuts, but exclude."""
		mask = self.trues()
		for subsample in ['target_subsample','fiber_subsample','spectro_subsample','reliable_redshift','final_subsample']:
			if subsample in exclude_subsample: continue
			for key in parameters[subsample]:
				if key not in exclude: mask &= self.selection(parameters[subsample][key],key=key,txt=subsample)
		self.logger.info('Subsample with {:d}/{:d} objects.'.format(mask.sum(),len(mask)))
		return mask
	
	def all_randoms_cuts(self,parameters,exclude=[],exclude_subsample=[]):
		"""Apply all randoms cuts, but exclude."""
		mask = self.trues()
		for subsample in ['randoms_target_subsample','randoms_fiber_subsample','final_subsample']:
			if subsample in exclude_subsample: continue
			for key in parameters[subsample]:
				if key not in exclude: mask &= self.selection(parameters[subsample][key],key=key,txt=subsample)
		self.logger.info('Subsample with {:d}/{:d} objects.'.format(mask.sum(),len(mask)))
		return mask
	
	def pixelize(self,key_ra='RA',key_dec='DEC',nside=256,nest=False,degree=True,mask=None):
		"""Pixelize self."""
		import healpy
		if mask is None:
			theta_rad,phi_rad = utils.radec_to_healpix(self[key_ra],self[key_dec],degree=degree)
		else:
			theta_rad,phi_rad = utils.radec_to_healpix(self[key_ra][mask],self[key_dec][mask],degree=degree)
		return healpy.ang2pix(nside,theta_rad,phi_rad,nest=nest)
	
	def from_healpix(self,healpix,fields=[],fields_healpix=None,key_hpind='hpind',**kwargs):
		if isinstance(fields,(str,unicode)): fields = [fields]
		if isinstance(fields_healpix,(str,unicode)): fields_healpix = [fields_healpix]
		if fields_healpix is None: fields_healpix = map(self.__class__.object_to_pixel,fields)
		if fields is None: fields = map(self.__class__.pixel_to_object,fields_healpix)
		pix = self.pixelize(**kwargs)
		for field,field_healpix in zip(fields,fields_healpix):
			self.logger.info('Updating field {} with healpix values.'.format(field))
			self[field] = utils.digitized_interp(pix,healpix[key_hpind],healpix[field_healpix],fill=healpix.default_value(field_healpix))
			
	@property
	def weight_object(self):
		comp_weights = ['WEIGHT_SYSTOT','WEIGHT_CP','WEIGHT_NOZ']
		for field in comp_weights[1:]:
			if field not in self.data:
				if comp_weights[0] in self.data:
					self.logger.warning('Only WEIGHT_SYSTOT, right?')
					return self[comp_weights[0]]
				else:
					self.logger.warning('No weight at all, right?')
					return self.ones()
		return scipy.prod([self[field] for field in comp_weights],axis=0)
		
	def weight_fkp(self,key_density='NZ',P0=4000.):
		return 1./(1.+self[key_density]*P0)
	
	@property
	def num_chunk(self):
		"""Return chunk number: eboss22 -> 22."""
		#return scipy.array(map(lambda s: int(s.replace('eboss','')),self['chunk']),dtype=int)
		toret = self.zeros(dtype=int)
		for chunk in scipy.unique(self['chunk']):
			toret[self['chunk']==chunk] = int(chunk.replace('eboss',''))
		return toret
	
	@property	
	def ELG_sector(self):
		"""Return ELG_sector, unique to all ELG chunks."""
		return self.num_chunk*self.__class__.OFFSET_SECTOR + self['sector']
		
	@property	
	def ELG_INGROUP(self):
		"""Return ELG_INGROUP, unique to all ELG chunks."""
		return self.num_chunk*self.__class__.OFFSET_INGROUP + self['INGROUP']
	
	@property
	def decals_uniqid(self):
		"""Return decals_uniqid, unique DECaLS identifier."""
		return scipy.array(['{}/{}/{}'.format(dr,br,obj) for (dr,br,obj) in zip(self['decals_dr'],self['brickname'],self['decals_objid'])])
		
	@property
	def plate_mjd(self):
		return scipy.array(['{:d}_{:d}'.format(p,m) for p,m in zip(self['plate'],self['MJD'])])

	@property
	def spectro_mjd(self):
		spectro = (self['FIBERID'] >= 501) + 1
		return scipy.array(['{:d}_{:d}_{:d}'.format(p,m,s) for p,m,s in zip(self['plate'],self['MJD'],spectro)])

	def spectro_sn(self,b='G',dtype='f8'):
		spectro = (self['FIBERID'] >= 501)
		toret = self.ones(dtype=dtype)
		b = b.upper()
		toret[spectro] = self['SPEC2_{}'.format(b)][spectro]
		toret[~spectro] = self['SPEC1_{}'.format(b)][~spectro]
		return toret

	@staticmethod
	def object_to_pixel(txt):
		if txt.startswith('hp'):
			return txt
		return 'hp{}'.format(txt)
	
	@staticmethod
	def pixel_to_object(txt):
		keep_hp = ['decalsdens','stardens']
		for k in keep_hp:
			if k in txt: return txt
		return txt.replace('hp','')
		
	def redrock_to_standard(self):
		self['Z_ok'] = self['rr_ZOK']
		for field in list(self.data.fields):
			if field.startswith('rr_') and ('BRICKNAME' not in field):
				self.logger.debug('Changing field name {} to {}.'.format(field,field[3:]))
				self[field[3:]] = self[field]
		#self['Z_reliable'] = self['rr_ZOK'] & ((self['rr_Z_zQ'] != 0) | (self['rr_Z_zCont'] != 0))
