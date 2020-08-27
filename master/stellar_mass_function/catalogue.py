import os
import numpy
import scipy
from astropy.io import fits
import logging
import textwrap

_NATURAL_TO_FITS = {'bool':'L','signed byte':'B','unsigned byte':'X','int8':'I','int16':'I','int32':'J','int64':'K','char':'A','float32':'E','float64':'D','complex64':'C','complex128':'M','object':'A'}
_FITS_TO_NATURAL = {_NATURAL_TO_FITS[key]:key for key in _NATURAL_TO_FITS}
_FITS_TO_NATURAL['I'] = 'int16'
_FITS_TO_NATURAL['A'] = 'char'
_NATURAL_TO_NUMPY = {'bool':'bool','signed byte':'b','unsigned byte':'uint8','int8':'int8','int16':'int16','int32':'int32','int64':'int64','char':'U','float32':'float32','float64':'float64','complex64':'complex64','complex128':'complex128','object':'object'}
_NUMPY_TO_NATURAL = {_NATURAL_TO_NUMPY[key]:key for key in _NATURAL_TO_NUMPY}
_TAB = 20
_WIDTH = 100

class Format(object):
	
	def __init__(self,fmt=None):
	
		if isinstance(fmt,(str,unicode)):
			try:
				self.__dict__.update(Format.from_fits(fmt).__dict__)
			except KeyError:
				self.__dict__.update(Format.from_str(fmt).__dict__)
		elif isinstance(fmt,scipy.dtype):
			self.__dict__.update(Format.from_numpy(fmt).__dict__)
		elif isinstance(fmt,Format):
			self.base = fmt.base
			self.shape = fmt.shape
		else:
			raise ValueError('Please provide valid format.')
		
	@classmethod
	def from_str(cls,fmt):
		self = object.__new__(cls)
		self.base = fmt
		self.shape = []
		if '[' in fmt:
			beg = fmt.index('[') + 1
			end = fmt.index(']',beg)
			self.shape = fmt[beg:end].split(',')
			self.shape = [int(s) for s in self.shape if s]
			self.base = fmt[:beg-1]
		return self
	
	def __str__(self):
		return self.natural
		
	@property
	def fits(self):
		base = _NATURAL_TO_FITS[self.base]
		if self.shape:
			return str(self.shape[0]) + base
		return base

	@property
	def numpy(self):
		base = _NATURAL_TO_NUMPY[self.base]
		if self.shape:
			if 'U' in base: return scipy.dtype(('U',self.shape[0]))
			return scipy.dtype((base,self.shape))
		return scipy.dtype(base)
	
	@property
	def natural(self):
		if self.shape:
			return '{}[{}]'.format(self.base,','.join([str(s) for s in self.shape]))
		return self.base
	
	@classmethod
	def from_numpy(cls,dt):
		self = object.__new__(cls)
		self.base = str(dt.base)
		if 'U' in self.base:
			self.shape = [int(self.base[2:])]
			if self.shape[0] == 0: self.shape = []
			self.base = 'char'
		else:
			self.base = _NUMPY_TO_NATURAL[self.base]
			self.shape = list(dt.shape)
		return self
		
	@classmethod
	def from_fits(cls,ft):
		self = object.__new__(cls)
		self.base = _FITS_TO_NATURAL[ft[-1]]
		self.shape = [int(ft[:-1])] if len(ft)>1 else []
		return self


class ColumnDescriptor(dict):
	
	DESCRIPTORS = ['field','format','origin','unit','description']
	
	def __init__(self,**kwargs):
		
		self.update({des:'' for des in self.__class__.DESCRIPTORS})
		for key in kwargs:
			if key.lower() in self.__class__.DESCRIPTORS:
				self[key.lower()] = kwargs[key]
		for key in ['origin','unit','description']:
			if not self[key]: self[key] = ''
		self.format = Format(self.format)
		if not self.field: raise ValueError('Please provide valid field name.')
	
	def __str__(self):
		return self.tostr()
	
	def tostr(self,tab=_TAB):
		txt = ''
		for des in self.getstate():
			txt += '{:<{tab}}'.format(des,tab=tab)
		return txt
	
	def getstate(self):
		return [self[des] for des in self]
	
	def setstate(self,state):
		for des,st in zip(self,state):
			self[des] = st

	def __eq__(self,name):
		if isinstance(name,(str,unicode)):
			return self.field == name
		return self.field == name.field

	@classmethod
	def fromstr(cls,txt,header):

		indices = []
		descriptors = []
		for des in cls.DESCRIPTORS:
			if des in header:
				indices.append(header.index(des))
				descriptors.append(des)
				
		argsort = scipy.argsort(indices)
		indices = scipy.asarray(indices)[argsort]
		descriptors = scipy.asarray(descriptors)[argsort]
		new = {}
		for des,beg,end in zip(descriptors[:-1],indices[:-1],indices[1:]):
			while end>beg:
				if txt[end-1]==' ': end = end-1
				else: break
			new[des] = txt[beg:end]
		new[descriptors[-1]] = txt[indices[-1]:]
		
		return cls(**new)
		
	def __iter__(self):
		return self.__class__.DESCRIPTORS.__iter__()
	
	def __getattr__(self,name):
		return self[name]
		
	def __setattr__(self,name,value):
		self[name] = value
		
	def copy(self):
		return self.__class__(**self)
		
	def deepcopy(self):
		return self.copy()		


class DataModel(list):

	logger = logging.getLogger('DataModel')
	DESCRIPTORS = ['{}s'.format(des) for des in ColumnDescriptor.DESCRIPTORS]

	def __init__(self,listcols=[]):
		for col in listcols: self.add(col)

	def __str__(self):
		return self.tostr()

	def add(self,kwargs):
		if not isinstance(kwargs,ColumnDescriptor):
			kwargs = ColumnDescriptor(**kwargs)
		try: 
			self.remove(kwargs)
		except ValueError:
			pass
		self.append(kwargs)
	
	@property
	def tabstr(self):
		return max(map(len,self.fields))+1
	
	@property
	def widthstr(self):
		return len(self.__class__.DESCRIPTORS)*self.tabstr

	def tostr(self,tab=None,width=None):
		
		txt = ''
		if not tab: tab = self.tabstr
		if not width: width = self.widthstr
		for des in self.__class__.DESCRIPTORS:
			txt += '{:<{tab}}'.format(des,tab=tab)
		txt += '\n' + '-'*width + '\n'
		for col in self: txt += col.tostr(tab=tab) + '\n'	
		return txt
	
	@classmethod
	def fromstr(cls,txt):
		self = cls()
		txt = txt.split('\n')
		for col in txt:
			if col and (col[:len(self.__class__.DESCRIPTORS[0])] != self.__class__.DESCRIPTORS[0]) and (col[0] != '-'):
				self += [ColumnDescriptor.fromstr(col,txt[0])]
		return self
	
	def save(self,path,tab=_TAB,width=_WIDTH):
		with open(path,'w') as file:
			file.write(self.tostr(tab,width))
	
	@classmethod
	def load(cls,path):
		cls.logger.info('Loading data model {}.'.format(path))
		with open(path,'r') as file:
			return cls.fromstr(file.read())
		
	def __getitem__(self,name):
		if isinstance(name,(str,unicode,ColumnDescriptor)):
			for col in self:
				if col == name:
					return col
			raise AttributeError
		return list.__getitem__(self,name)

	def index(self,name):
		for icol,col in enumerate(self):
			if col == name:
				return icol
		raise IndexError
	
	def remove(self,name):
		try:
			ind = self.index(name)
		except IndexError:
			raise ValueError
		del self[ind]
	
	def __getattr__(self,name):
		if name in self.__class__.DESCRIPTORS: 
			return [col[name[:-1]] for col in self]
			
	def __and__(self,other):
		fields = [field for field in self.fields if field in other.fields]
		return self.__class__([self[field] for field in fields])

	def copy(self):
		return self.__class__(self)
		
	def deepcopy(self):
		return self.__class__([col.deepcopy() for col in self])

class Header(dict):

	logger = logging.getLogger('header')
	DESCRIPTORS = ['code','date','author','version','tracer','pipeline','vetomask','wiki']

	def __init__(self,**kwargs):
		self.update(kwargs)
		self['LONGSTRN'] = ''

	def __str__(self):	
		return self.tostr()
	
	def tostr(self,tab=_TAB,width=_WIDTH):
		txt = 'header\n'
		txt += '-'*width + '\n'
		for key in self:
			txt += '{:<{tab}}{}\n'.format(key,self[key],tab=tab)
		return txt
	
	def __iter__(self):
		self_descriptors = {key.lower():key for key in self.keys()}
		listed = []
		for key in self.__class__.DESCRIPTORS:
			if key in self_descriptors:
				yield self_descriptors[key]
				listed.append(key)
		for key in self_descriptors:
			if key not in listed:
				yield self_descriptors[key]

	def copy(self):
		return self.__class__(**self)
	
	def deepcopy(self):
		return self.copy()


class Data(object):

	logger = logging.getLogger('Data')

	def __init__(self,columns={},fields=None):
		self.columns = {}
		if fields is None:
			self.fields = columns.keys()
			self.columns.update(columns)
		else:
			self.fields = list(fields)
			for key in self.fields:
				self.columns[key] = columns[key]

	def __getitem__(self,name):
		if isinstance(name,list) and isinstance(name[0],(str,unicode)):
			return [self[name_] for name_ in name]
		if isinstance(name,(str,unicode)):
			try:
				ind = self.fields.index(name)
			except ValueError:
				try:
					ind = self.lowerfields.index(name.lower())
				except ValueError:
					raise KeyError('There is no field {} in the data.'.format(name))
				self.logger.info('{} is not in the data; {} (lower case) found.'.format(name,name.lower()))
			return self.columns[self.fields[ind]]
		else:
			return self.__class__({key:self.columns[key][name] for key in self.fields})	
	
	def __setitem__(self,name,item):
		if isinstance(name,(str,unicode)):
			if name not in self.fields:
				self.fields.append(name)
			self.columns[name] = item
		else:
			for key in self.fields:
				tmp = getattr(self,key)
				tmp[name] = item
				self.columns[key] = tmp

	def __delitem__(self,name):
		del self.columns[name]
	
	@property
	def lowerfields(self):
		return [field.lower() for field in self.fields]
			
	def	__contains__(self,name):
		return name.lower() in self.lowerfields
	
	def __iter__(self):
		return self.fields.__iter__()
	
	def __str__(self):
		return str(self.columns)
		
	def __len__(self):
		return len(self.columns[self.fields[0]])	
		
	def remove(self,name):
		if name in self.fields:
			del self.columns[name]
			self.fields.remove(name)
		
	def __radd__(self,other):
	
		if other == 0: return self
		else: return self.__add__(other)
		
	def __add__(self,other):
		new = {}
		fields = [field for field in self.fields if field in other.fields]
		for field in fields:
			new[field] = scipy.concatenate([self[field],other[field]])
		return self.__class__(new,fields=fields)

	def as_dict(self,fields=None):
		if fields is None: fields = self.fields
		return {field:self[field] for field in fields}
	
	def copy(self):
		return self.__class__(self)
		
	def deepcopy(self):
		new = {}
		fields = list(self.fields)
		for field in fields:
			new[field] = self[field].copy()
		return self.__class__(new,fields=fields)
	

class Catalogue(object):

	logger = logging.getLogger('Catalogue')
	
	LIST_NON_ZERO = []

	def __init__(self,header={},datamodel=[],data={},size=None):
		self.size = size
		self.set_data(data)
		if self.data.fields: self.size = len(self.data)
		self.set_header(header)
		self.set_datamodel(datamodel)
		
	@classmethod
	def load(cls,path,*args,**kwargs):
		if isinstance(path,list):
			return sum(cls.load(path_,*args,**kwargs) for path_ in path)
		if path.endswith('.fits') or path.endswith('.fits.gz'):
			return cls.load_fits(path,*args,**kwargs)
		else:
			return cls.load_csv(path,*args,**kwargs)
	
	@classmethod	
	def load_fits(cls,path,path_data_model=None,ext=1):
		self = cls()
		self.logger.info('Loading catalogue {}.'.format(path))
		HDUList = fits.open(path,mode='readonly',memmap=True)
		columns = HDUList[ext].columns
 		self.data = Data(HDUList[ext].data,fields=columns.names)
 		self.header = Header(**HDUList[0].header)
 		self.size = len(HDUList[ext].data)
		if path_data_model is not None:
			self.set_datamodel(path_data_model)
		else:
			self.datamodel = DataModel()
			for ifield,field in enumerate(columns.names):
				key = 'TTYPE{0:d}'.format(ifield+1)
				description = HDUList[ext].header.comments[key]
				self.add_field(dict(field=field,format=columns[field].format,unit=columns[field].unit,description=description))
		return self
		
	@classmethod	
	def load_csv(cls,path,path_data_model=None,sep='\s+',**kwargs):
		import pandas
		self = cls()
		self.logger.info('Loading catalogue {}.'.format(path))
		df = pandas.read_csv(path,sep=sep,**kwargs)
 		self.data = Data({field: df[field].values for field in df.columns},fields=df.columns)
 		self.header = Header()
 		self.size = len(df)
		if path_data_model is not None:
			self.set_datamodel(path_data_model)
		else:
			self.datamodel = DataModel()
			for field in df.columns:
				self.add_field(dict(field=field,format=df[field].dtype,unit='',description=''))
		return self
	
	def add_field(self,*fields):
		for field in fields:
			self.datamodel.add(field)

	def remove_field(self,*fields):
		for field in fields:
			self.datamodel.remove(field)
			self.data.remove(field)
			
	def keep_field(self,*fields):
		torm = []
		for field in fields: assert field in self.datamodel, '{} not in datamodel'.format(field)
		for col in self.datamodel:
			if True not in map(col.__eq__,fields):
				torm.append(col.field)
		self.remove_field(*torm)
		self.logger.info('Keeping field(s) {}.'.format(self.fields))
	
	def set_data(self,data):
		if isinstance(data,Data):
			self.data = data
		else: self.data = Data(data)

	def set_header(self,header):
		if isinstance(header,Header):
			self.header = header
		else: self.header = Header(**header)

	def set_datamodel(self,datamodel):
		if isinstance(datamodel,(str,unicode)):
			self.datamodel = DataModel.load(datamodel)
		elif isinstance(datamodel,DataModel):
			self.datamodel = datamodel
		else: self.datamodel = DataModel(datamodel)
		
	def __getattribute__(self,name):
		if name in DataModel.DESCRIPTORS:
			return getattr(object.__getattribute__(self,'datamodel'),name)
		else:
			return object.__getattribute__(self,name)

	def __setitem__(self,name,item):
		if isinstance(name,list) and isinstance(name[0],(str,unicode)):
			for name_,item_ in zip(name,item):
				self.data[name_] = item_
		if isinstance(name,(str,unicode)):
			self.data[name] = item
		else: self.data = self.data[name]
		
	def __getitem__(self,name):
		if isinstance(name,list) and isinstance(name[0],(str,unicode)):
			return [self[name_] for name_ in name]
		if isinstance(name,(str,unicode)):
			if name in self.datamodel:
				dtype = self.datamodel[name].format.numpy
				if name not in self.data:
					self.data[name] = scipy.full(len(self),self.default_value(name),dtype=dtype)
				else:
					self.data[name] = self.data[name].astype(dtype.base,copy=False)
			return self.data[name]
		return self.__class__(data=self.data[name],datamodel=self.datamodel.deepcopy(),header=self.header.deepcopy())

	def __delitem__(self,name):
		del self.data[name]
		
	def copy(self):
		return self.__class__(data=self.data.copy(),datamodel=self.datamodel.copy(),header=self.header.copy())
	
	def deepcopy(self):
		return self.__class__(data=self.data.deepcopy(),datamodel=self.datamodel.deepcopy(),header=self.header.deepcopy())
	
	def __len__(self):
		return self.size
	
	@property
	def shape(self):
		return (self.size,len(self.fields))
	
	def save(self,path,*args,**kwargs):
	
		if path.endswith('.fits'):
			self.save_fits(path,*args,**kwargs)
		else:
			self.save_csv(path,*args,**kwargs)
		
	def save_fits(self,path,path_data_model=None,keep=None):
	
		if keep is not None: self.keep_field(*keep)
	
		columns = []
		for field in self.fields:
			format = self.datamodel[field].format
			if format.natural == 'object':
				format.base = 'char'
				format.shape = [len(self[field][0])]
			unit = self.datamodel[field].unit
			if unit: self.logger.debug('Saving {} of shape {} with fits format {} and unit {}.'.format(field,self[field].shape,format.fits,unit))
			else: self.logger.debug('Saving {} of shape {} with fits format {}.'.format(field,self[field].shape,format.fits))
			columns.append(fits.Column(name=field,format=format.fits,unit=unit,array=self[field]))
		hdu = fits.BinTableHDU.from_columns(columns)
		for icol,col in enumerate(self.datamodel):
			key = 'TTYPE{0:d}'.format(icol+1)
			hdu.header[key] = (field,col.description)
		primary_hdu = fits.PrimaryHDU(header=fits.Header(self.header))
	
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,path))
		fits.HDUList([primary_hdu,hdu]).writeto(path,overwrite=True)
		
		if path_data_model is not None:
			self.logger.info('Saving {} to {}.'.format(self.datamodel.__class__.__name__,path_data_model))
			self.datamodel.save(path_data_model)
			
	def save_csv(self,path,path_data_model=None,keep=None,sep=' ',na_rep='',index=False,**kwargs):
	
		import pandas

		if keep is not None: self.keep_field(*keep)
		for field in self.fields: self[field]
		df = pandas.DataFrame(data=self.data.as_dict(self.fields),columns=self.fields)
		#df = pandas.DataFrame.from_dict(data=self.data.as_dict(),orient='columns')
		#save = scipy.asarray([self[key] for key in self.fields],dtype={'names':self.fields,'formats':[fmt.numpy for fmt in self.formats]}).T
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,path))
		df.to_csv(path,sep=sep,na_rep=na_rep,index=index,**kwargs)
		#scipy.savetxt(path,save)
		
		if path_data_model is not None:
			self.logger.info('Saving {} to {}.'.format(self.datamodel.__class__.__name__,path_data_model))
			self.datamodel.save(path_data_model)
	
	def as_dict(self,fields=None):
		return self.data.as_dict(fields=fields)

	def zeros(self,dtype=scipy.float64):
		return scipy.zeros(len(self),dtype=dtype)
	
	def ones(self,dtype=scipy.float64):
		return scipy.ones(len(self),dtype=dtype)
	
	def falses(self):
		return self.zeros(dtype=scipy.bool_)
	
	def trues(self):
		return self.ones(dtype=scipy.bool_)
	
	def nans(self):
		return self.ones()*scipy.nan
	
	def __contains__(self,name):
		return name in self.datamodel
	
	def __iter__(self):
		return self.fields.__iter__()

	def __radd__(self,other):
	
		if other == 0: return self
		return self.__add__(other)
		
	def __add__(self,other):
		
		for field in self: self[field]
		for field in other: other[field]
		datamodel = self.datamodel & other.datamodel
		data = self.data + other.data
		size = self.size + other.size
		return self.__class__(header=self.header,datamodel=datamodel,data=data,size=size)

	def __str__(self):
		return self.data.__str__()

	def get(self,name,default=None):
		if name in self.data: return self[name]
		return default
		
	@classmethod
	def bad_values(cls,field):
		for nonzero in cls.LIST_NON_ZERO:
			if nonzero in field:
				return 'NaN, inf, negative'
		return 'NaN, inf'
	
	def bad_value(self,field):
		for nonzero in self.__class__.LIST_NON_ZERO:
			with numpy.warnings.catch_warnings():
				numpy.warnings.filterwarnings('ignore','invalid value encountered in less_equal')
				if nonzero in field:
					return scipy.isnan(self[field]) | scipy.isinf(self[field]) | (self[field]<=0.)
		return scipy.isnan(self[field]) | scipy.isinf(self[field])
	
	def good_value(self,*args,**kwargs):
		return ~self.bad_value(*args,**kwargs)

	def default_value(self,name):
		fmt = self.datamodel[name].format.numpy
		el = scipy.zeros(1,dtype=fmt)[0]
		if isinstance(el,scipy.integer): return 0
		if isinstance(el,scipy.bool_): return False
		if isinstance(el,(str,unicode)): return ''
		return scipy.nan

	def fill_default_value(self,name,mask=None):
		if mask is None: mask = self.trues()
		self[name][mask] = self.default_value(name)

	def to_nbodykit(self,fields=None):
	
		from nbodykit.base.catalog import CatalogSource
		from nbodykit import CurrentMPIComm
	
		comm = CurrentMPIComm.get()
		if comm.rank == 0:
			source = self.as_dict(fields=fields)
		else:
			source = None
		source = comm.bcast(source)

		# compute the size
		start = comm.rank * self.size // comm.size
		end = (comm.rank  + 1) * self.size // comm.size

		new = object.__new__(CatalogSource)
		new._size = end - start
		CatalogSource.__init__(new,comm=comm)
		for key in source:
			new[key] = new.make_column(source[key])[start:end]

		return new 	
