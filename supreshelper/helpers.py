import skimage

import numpy as np
import scipy
import skimage

from glob import glob
import os


def lowres_image_iterator(path, img_as_float=True):
	"""
	Iterator over all of a scene's low-resolution images (LR*.png) and their
	corresponding status maps (QM*.png).
	
	Returns at each iteration a `(l, c)` tuple, where:
	* `l`: matrix with the loaded low-resolution image (values as np.uint16 or
	       np.float64 depending on `img_as_float`),
	* `c`: the image's corresponding "clear pixel?" boolean mask.
	
	Scenes' image files are described at:
	https://kelvins.esa.int/proba-v-super-resolution/data/
	"""
	path = path if path[-1] in {'/', '\\'} else (path + '/')
	for f in glob(path + 'LR*.png'):
		q = f.replace('LR', 'QM')
		l = skimage.io.imread(f, dtype=np.uint16)
		c = skimage.io.imread(q, dtype=np.bool)
		if img_as_float:
			l = skimage.img_as_float64(l)
		yield (l, c)


def bicubic_upscaling(img):
    """
    Compute a bicubic upscaling by a factor of 3.
    """
    r = skimage.transform.rescale(img, scale=3, order=3, mode='edge',
                                  anti_aliasing=False, multichannel=False)
    # NOTE: Don't change these options. They're required by `baseline_upscale`.
    # http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rescale
    # http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
    return r


def baseline_upscale(path):
	"""
	Reimplementation of the image enhancement operation performed by the
	baseline code (`generate_sample_submission.py`) provided in:
	https://kelvins.esa.int/proba-v-super-resolution/submission-rules/
	
		"takes all low resolution images that have the maximum amount of clear
		pixels, computes a bicubic upscaling by a factor of 3 and averages their
		pixel intensities."
	
	This function takes as argument the `path` to a single scene, and returns
	the matrix with the scene's enhanced image.
	"""
	clearance = {}
	for (l, c) in lowres_image_iterator(path, img_as_float=True):
		clearance.setdefault(c.sum(), []).append(l)
	
	# take all the images that have the same maximum clearance
	imgs = max(clearance.items(), key=lambda i: i[0])[1]
	
	sr = np.mean([
		bicubic_upscaling(i)
		for i in imgs
		], axis=0)
	
	return sr




def central_tendency(images, agg_with='median',
	                 only_clear=False, fill_obscured=False,
	                 img_as_float=True):
	"""
	Aggregate the given `images` through a statistical central tendency measure,
	chosen by setting `agg_with` to either 'mean', 'median' or 'mode'.
	
	Expects `images` to be a list of `(image, status map)` tuples.
	Should `images` be a string, it's interpreted as the path to a scene's
	files. The code will then aggregate that scene's low resolution images
	(LR*.png), while taking also into account their status maps (QM*.png).
	
	Will optionally aggregate only images' clear pixels (if `only_clear=True`)
	by using the information in images' corresponding status maps.
	
	In some scenes, some pixels are obscured in all of the low-resolution
	images. Aggregation with mean/median will return np.nan for those pixels,
	and aggregation with mode will return 0.0.
	If called with `fill_obscured=True` those pixels will be filled with the
	`agg_with` aggregate of the values at all those obscured pixels. Setting
	`fill_obscured` to one of 'mean', 'median' or 'mode' will indicate that is
	the measure that should be used to aggregate obscured pixels.
	"""
	agg_opts = {
		'mean'   : lambda i: np.nanmean(i, axis=0),
		'median' : lambda i: np.nanmedian(i, axis=0),
		'mode'   : lambda i: scipy.stats.mode(i, axis=0, nan_policy='omit').mode[0],
		}
	agg = agg_opts[agg_with]
	
	imgs = []
	obsc = []
	
	if isinstance(images, str):
		images = lowres_image_iterator(images, img_as_float or only_clear)
	elif only_clear:
		# Images were given by the caller, rather than loaded here.
		# Because `only_clear=True`, we generate copies of all lr images, so the
		# function will have no unintended side-effects on the caller's side.
		images = [(l.copy(), c) for (l,c) in images]
	
	for (l, c) in images:
		
		if only_clear:
			
			# keep track of the values at obscured pixels
			if fill_obscured != False:
				o = l.copy()
				o[c] = np.nan
				obsc.append(o)
			
			# replace values at obscured pixels with NaNs
			l[~c] = np.nan
		
		imgs.append(l)
	
	# aggregate the images
	with np.warnings.catch_warnings():   ## https://stackoverflow.com/a/29348184
		# suppress the warnings that originate when `only_clear=True`
		# but some pixels are never clear in any of the images
		np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
		np.warnings.filterwarnings('ignore', r'Mean of empty slice')
		
		agg_img = agg(imgs)
		
		if only_clear and fill_obscured != False:
			if isinstance(fill_obscured, str):
				agg = agg_opts[fill_obscured]
			some_clear = np.isnan(obsc).any(axis=0)
			obsc = agg(obsc)
			obsc[some_clear] = 0.0
			np.nan_to_num(agg_img, copy=False)
			agg_img += obsc
	
	return agg_img


def highres_image(path, img_as_float=True):
	"""
	Load a scene's high resolution image and its corresponding status map.
	
	Returns a `(hr, sm)` tuple, where:
	* `hr`: matrix with the loaded high-resolution image (values as np.uint16 or
	        np.float64 depending on `img_as_float`),
	* `sm`: the image's corresponding "clear pixel?" boolean mask.
	
	Scenes' image files are described at:
	https://kelvins.esa.int/proba-v-super-resolution/data/
	"""
	path = path if path[-1] in {'/', '\\'} else (path + '/')
	hr = skimage.io.imread(path + 'HR.png')
	sm = skimage.io.imread(path + 'SM.png')
	if img_as_float:
		hr = skimage.img_as_float64(hr)
	return (hr, sm)
	


def lowres_image_iterator(path, img_as_float=True):
	"""
	Iterator over all of a scene's low-resolution images (LR*.png) and their
	corresponding status maps (QM*.png).
	
	Returns at each iteration a `(l, c)` tuple, where:
	* `l`: matrix with the loaded low-resolution image (values as np.uint16 or
	       np.float64 depending on `img_as_float`),
	* `c`: the image's corresponding "clear pixel?" boolean mask.
	
	Scenes' image files are described at:
	https://kelvins.esa.int/proba-v-super-resolution/data/
	"""
	path = path if path[-1] in {'/', '\\'} else (path + '/')
	for f in glob(path + 'LR*.png'):
		q = f.replace('LR', 'QM')
		l = skimage.io.imread(f)
		c = skimage.io.imread(q)
		if img_as_float:
			l = skimage.img_as_float64(l)
		yield (l, c)
	


# [============================================================================]


def check_img_as_float(img, validate=True):
	"""
	Ensure `img` is a matrix of values in floating point format in [0.0, 1.0].
	Returns `img` if it already obeys those requirements, otherwise converts it.
	"""
	if not issubclass(img.dtype.type, np.floating):
		img = skimage.img_as_float64(img)
	# https://scikit-image.org/docs/dev/api/skimage.html#img-as-float64
	
	if validate:
		# safeguard against unwanted conversions to values outside the
		# [0.0, 1.0] range (would happen if `img` had signed values).
		assert img.min() >= 0.0 and img.max() <= 1.0
	
	return img
	


# [============================================================================]


def all_scenes_paths(base_path):
	"""
	Generate a list of the paths to all scenes available under `base_path`.
	"""
	base_path = base_path if base_path[-1] in {'/', '\\'} else (base_path + '/')
	return [
		base_path + c + s
		for c in ['RED/', 'NIR/']
		for s in sorted(os.listdir(base_path + c))
		]
	


def scene_id(scene_path, incl_channel=False):
	"""
	Extract from a scene's path its unique identifier.
	
	Examples
	--------
	>>> scene_id('probav/train/RED/imgset0559/')
	'imgset0559'
	>>> scene_id('probav/train/RED/imgset0559', incl_channel=True)
	'RED/imgset0559'
	"""
	sep = os.path.normpath(scene_path).split(os.sep)
	if incl_channel:
		return '/'.join(sep[-2:])
	else:
		return sep[-1]
	

# [============================================================================]

def prepare_submission(images, scenes, subm_fname='submission.zip'):
	"""
	Prepare a set of images for submission.
	
	Given a list of `images` (as matrices of shape (384, 384)), and the paths
	to the `scenes` to which they correspond, write a zip file containing all
	images as .png files, named after their scene's identification
	(example: imgset1160.png).
	"""
	assert len(images) == 290, '%d images provided, 290 expected.' % len(images)
	assert len(images) == len(scenes), "Mismatch in number of images and scenes."
	assert subm_fname[-4:] == '.zip'
	
	# specific warnings we wish to ignore
	warns = [
		'tmp.png is a low contrast image',
		'Possible precision loss when converting from float64 to uint16']
	
	with np.warnings.catch_warnings():
		for w in warns:
			np.warnings.filterwarnings('ignore', w)
		
		print('Preparing submission. Writing to "%s".' % subm_fname)
		
		with ZipFile(subm_fname, mode='w') as zf:
			
			for img, scene in zip(tqdm(images), scenes):
				assert img.shape == (384, 384), \
					'Wrong dimensions in image for scene %s.' % scene
				
				skimage.io.imsave('tmp.png', img)
				zf.write('tmp.png', arcname=scene_id(scene) + '.png')
		
		os.remove('tmp.png')