import skimage

import numpy as np
import scipy
import skimage

import numba
import pandas as pd

from glob import glob
import os

from ._tqdm import tqdm

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
        
        
# [============================================================================]


# Baseline cPSNR values for the dataset's images. Used for normalizing scores.
# (provided by the competition's organizers)
baseline_cPSNR = pd.read_csv(
    os.path.dirname(os.path.abspath(__file__)) + '/norm.csv',
    names = ['scene', 'cPSNR'],
    index_col = 'scene',
    sep = ' ')


def score_images(imgs, scenes_paths, *args):
	"""
	Measure the overall (mean) score across multiple super-resolved images.
	
	Takes as input a sequence of images (`imgs`), a sequence with the paths to
	the corresponding scenes (`scenes_paths`), and optionally a sequence of
	(hr, sm) tuples with the pre-loaded high-resolution images of those scenes.
	"""
	return np.mean([
#		score_image(*i)
		score_image_fast(*i)
		for i in zip(tqdm(imgs), scenes_paths, *args)
		])


def score_image(sr, scene_path, hr_sm=None):
	"""
	Calculate the individual score (cPSNR, clear Peak Signal to Noise Ratio) for
	`sr`, a super-resolved image from the scene at `scene_path`.
	
	Parameters
	----------
	sr : matrix of shape 384x384
		super-resolved image.
	scene_path : str
		path where the scene's corresponding high-resolution image can be found.
	hr_sm : tuple, optional
		the scene's high resolution image and its status map. Loaded if `None`.
	"""
	hr, sm = highres_image(scene_path) if hr_sm is None else hr_sm
	
	# "We assume that the pixel-intensities are represented
	# as real numbers ∈ [0,1] for any given image."
	sr = check_img_as_float(sr)
	hr = check_img_as_float(hr, validate=False)
	
	# "Let N(HR) be the baseline cPSNR of image HR as found in the file norm.csv."
	N = baseline_cPSNR.loc[scene_id(scene_path)][0]
	
	# "To compensate for pixel-shifts, the submitted images are
	# cropped by a 3 pixel border, resulting in a 378x378 format."
	sr_crop = sr[3 : -3, 3 : -3]
	
	crop_scores = []
	
	for (hr_crop, sm_crop) in hr_crops(hr, sm):
		# values at the cropped versions of each image that
		# fall in clear pixels of the cropped `hr` image
		_hr = hr_crop[sm_crop]
		_sr = sr_crop[sm_crop]
		
		# "we first compute the bias in brightness b"
		pixel_diff = _hr - _sr
		b = np.mean(pixel_diff)
		
		# "Next, we compute the corrected clear mean-square
		# error cMSE of SR w.r.t. HR_{u,v}"
		pixel_diff -= b
		cMSE = np.mean(pixel_diff * pixel_diff)
		
		# "which results in a clear Peak Signal to Noise Ratio of"
		cPSNR = -10. * np.log10(cMSE)
		
		# normalized cPSNR
		crop_scores.append(N / cPSNR)
#		crop_scores.append(cMSE)
	
	# "The individual score for image SR is"
	sr_score = min(crop_scores)
#	sr_score = N / (-10. * np.log10(min(crop_scores)))
	
	return sr_score


# [===================================]


def hr_crops(hr, sm):
	"""
	"We denote the cropped 378x378 images as follows: for all u,v ∈ {0,…,6},
	HR_{u,v} is the subimage of HR with its upper left corner at coordinates
	(u,v) and its lower right corner at (378+u, 378+v)."
	-- https://kelvins.esa.int/proba-v-super-resolution/scoring/
	"""
	num_cropped = 6
	max_u, max_v = np.array(hr.shape) - num_cropped
	
	for u in range(num_cropped + 1):
		for v in range(num_cropped + 1):
			yield hr[u : max_u + u, v : max_v + v], \
				  sm[u : max_u + u, v : max_v + v]
	

    
# [============================================================================]


def score_image_fast(sr, scene_path, hr_sm=None):
    """
    Calculate the individual score (cPSNR, clear Peak Signal to Noise Ratio) for
    `sr`, a super-resolved image from the scene at `scene_path`.

    Parameters
    ----------
    sr : matrix of shape 384x384
    super-resolved image.
    scene_path : str
    path where the scene's corresponding high-resolution image can be found.
    hr_sm : tuple, optional
    the scene's high resolution image and its status map. Loaded if `None`.
    """

    hr, sm = highres_image(scene_path) if hr_sm is None else hr_sm

    # "We assume that the pixel-intensities are represented
    # as real numbers ∈ [0,1] for any given image."
    sr = check_img_as_float(sr)
    hr = check_img_as_float(hr, validate=False)

    # "Let N(HR) be the baseline cPSNR of image HR as found in the file norm.csv."
    N = baseline_cPSNR.loc[scene_id(scene_path)][0]

    return score_against_hr(sr, hr, sm, N)



#@numba.jit('f8(f8[:,:], f8[:,:], b1[:,:], f8)', nopython=True, parallel=True)
@numba.jit(nopython=True, parallel=True)
def score_against_hr(sr, hr, sm, N):
    """
    Numba-compiled version of the scoring function.
    """
    num_cropped = 6
    max_u, max_v = np.array(hr.shape) - num_cropped

    # "To compensate for pixel-shifts, the submitted images are
    # cropped by a 3 pixel border, resulting in a 378x378 format."
    c = num_cropped // 2
    sr_crop = sr[c : -c, c : -c].ravel()

    # create a copy of `hr` with NaNs at obscured pixels
    # (`flatten` used to bypass numba's indexing limitations)
    hr_ = hr.flatten()
    hr_[(~sm).ravel()] = np.nan
    hr = hr_.reshape(hr.shape)

    #crop_scores = []
    cMSEs = np.zeros((num_cropped + 1, num_cropped + 1), np.float64)

    for u in numba.prange(num_cropped + 1):
        for v in numba.prange(num_cropped + 1):

            # "We denote the cropped 378x378 images as follows: for all u,v ∈
            # {0,…,6}, HR_{u,v} is the subimage of HR with its upper left corner
            # at coordinates (u,v) and its lower right corner at (378+u, 378+v)"
            hr_crop = hr[u : max_u + u, v : max_v + v].ravel()

            # "we first compute the bias in brightness b"
            pixel_diff = hr_crop - sr_crop
            b = np.nanmean(pixel_diff)

            # "Next, we compute the corrected clear mean-square
            # error cMSE of SR w.r.t. HR_{u,v}"
            pixel_diff -= b
            pixel_diff *= pixel_diff
            cMSE = np.nanmean(pixel_diff)

            # "which results in a clear Peak Signal to Noise Ratio of"
            #cPSNR = -10. * np.log10(cMSE)

            # normalized cPSNR
            #crop_scores.append(N / cPSNR)

            cMSEs[u, v] = cMSE

    # "The individual score for image SR is"
    #sr_score = min(crop_scores)
    sr_score = N / (-10. * np.log10(cMSEs.min()))

    return sr_score


# [============================================================================]


class scorer(object):
	
	def __init__(self, scene_paths, preload_hr=True):
		"""
		Wrapper to `score_image()` that simplifies the scoring of multiple
		super-resolved images.
		
		The scenes over which the scorer will operate should be given in
		`scene_paths`. This is either a sequence of paths to a subset of scenes
		or a string with a single path. In this case, it is interpreted as the
		base path to the full dataset, and `all_scenes_paths()` will be used to
		locate all the scenes it contains.
		
		Scene paths are stored in the object's `.paths` variable.
		When scoring, only the super-resolved images need to be provided.
		They are assumed to be in the same order as the scenes in `.paths`.
		
		If the object is instantiated with `preload_hr=True` (the default),
		all scene's high-resolution images and their status maps will be
		preloaded. When scoring they will be sent to `score_image()`, thus
		saving computation time in repeated scoring, at the expense of memory.
		"""
		if isinstance(scene_paths, str):
			self.paths = all_scenes_paths(scene_paths)
		else:
			self.paths = scene_paths
		
		self.hr_sm = [] if not preload_hr else [
			highres_image(scn_path, img_as_float=True)
			for scn_path in tqdm(self.paths, desc='Preloading hi-res images')]
		
		self.scores = []
		
	
	def __call__(self, sr_imgs, per_image=False, progbar=True, desc=''):
		"""
		Score all the given super-resolved images (`sr_imgs`), which correspond
		to the scenes at the matching positions of the object's `.paths`.
		
		Returns the overall score (mean normalized cPSNR).
		
		An additional value is returned if `per_image=True`: a list with each
		image's individual cPSNR score. In either case, this list remains
		available in the object's `.scores` variable until the next call.
		"""
		scenes_paths = tqdm(self.paths, desc=desc) if progbar else self.paths
		hr_sm = [] if self.hr_sm == [] else [self.hr_sm]
		
		self.scores = [
#			score_image(*i)
			score_image_fast(*i)
			for i in zip(sr_imgs, scenes_paths, *hr_sm)]
		
		assert len(self.scores) == len(self.paths)
		
		score = np.mean(self.scores)
		
		if per_image:
			return score, self.scores
		else:
			return score
    
