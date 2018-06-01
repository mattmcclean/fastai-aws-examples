import json, pdb, os, numpy as np, cv2, threading, math, io

import torch
from torch.autograd import Variable

def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            print(f'Image shape is {im.shape}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


# getting val_tfms to work without fastai import

from enum import IntEnum

class TfmType(IntEnum):
    """ Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            PIXEL: x and y are images and should be transformed in the same way.
                   Example: image segmentation.
            COORD: y are coordinates (i.e bounding boxes)
            CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
    """
    NO = 1
    PIXEL = 2
    COORD = 3
    CLASS = 4

class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3
    GOOGLENET = 4
    
class ChannelOrder():
    '''
    changes image array shape from (h, w, 3) to (3, h, w). 
    tfm_y decides the transformation done to the y element. 
    '''
    def __init__(self, tfm_y=TfmType.NO): self.tfm_y=tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 2)
        #if isinstance(y,np.ndarray) and (len(y.shape)==3):
        if self.tfm_y==TfmType.PIXEL: y = np.rollaxis(y, 2)
        elif self.tfm_y==TfmType.CLASS: y = y[...,0]
        return x,y
    
class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms + [crop_tfm, normalizer, ChannelOrder(tfm_y)]
    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)    
    
def A(*a): return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

class Denormalize():
    """ De-normalizes an image, returning it to original format.
    """
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """
    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        if self.tfm_y==TfmType.PIXEL and y is not None: y = (y-self.m)/self.s
        return x,y

class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """
    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.store = threading.local()

    def set_state(self): pass
    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x,False)
        return (x, self.do_transform(y,True)) if y is not None else x

#     @abstractmethod
#     def do_transform(self, x, is_y): raise NotImplementedError
    
class CoordTransform(Transform):
    """ A coordinate transform.  """

    @staticmethod
    def make_square(y, x):
        r,c,*_ = x.shape
        y1 = np.zeros((r, c))
        y = y.astype(np.int)
        y1[y[0]:y[2], y[1]:y[3]] = 1.
        return y1

    def map_y(self, y0, x):
        y = CoordTransform.make_square(y0, x)
        y_tr = self.do_transform(y, True)
        return to_bb(y_tr, y)

    def transform_coord(self, x, ys):
        yp = partition(ys, 4)
        y2 = [self.map_y(y,x) for y in yp]
        x = self.do_transform(x, False)
        return x, np.concatenate(y2)

class Scale(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz: int
            target size to scale minimum size.
        tfm_y: TfmType
            type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        if is_y: return scale_min(x, self.sz_y, cv2.INTER_NEAREST)
        else   : return scale_min(x, self.sz,   cv2.INTER_AREA   )
    
class NoCrop(CoordTransform):
    """  A transformation that resize to a square image without cropping.

    This transforms (optionally) resizes x,y at with the same parameters.
    Arguments:
        targ: int
            target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        if is_y: return no_crop(x, self.sz_y, cv2.INTER_NEAREST)
        else   : return no_crop(x, self.sz,   cv2.INTER_AREA   )

        
imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
stats = imagenet_stats

tfm_norm = Normalize(*stats, TfmType.NO)
tfm_denorm = Denormalize(*stats)

def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
              tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT):
    """
    Generate a standard set of transformations

    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.

    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.

    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz
    scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None
             else Scale(sz, tfm_y, sz_y=sz_y)]
    if pad: scale.append(AddPadding(pad, mode=pad_mode))
    if crop_type!=CropType.GOOGLENET: tfms=scale+tfms
    return Transforms(sz, tfms, normalizer, denorm, crop_type,
                      tfm_y=tfm_y, sz_y=sz_y)

crop_fn_lu = {CropType.NO: NoCrop}

def compose(im, y, fns):
    """ apply a collection of transformation functions fns to images
    """
    for fn in fns:
        #pdb.set_trace()
        im, y =fn(im, y)
    return im if y is None else (im, y)

def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r,c,*_ = im.shape
    ratio = targ/min(r,c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz, interpolation=interpolation)

def scale_to(x, ratio, targ): 
    '''
    no clue, does not work.
    '''
    return max(math.floor(x*ratio), targ)

def crop(im, r, c, sz): 
    '''
    crop image into a square of size sz, 
    '''
    return im[r:r+sz, c:c+sz]

def no_crop(im, min_sz=None, interpolation=cv2.INTER_AREA):
    """ Returns a squared resized image """
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    return cv2.resize(im, (min_sz, min_sz), interpolation=interpolation)


# -------- end val_tfms stuff

def write_test_image(img_bytes, path, file):
    if os.path.exists(path):
        print(f'Cleaning test dir: {path}')
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
    else:
        print(f'Creating test dir: {path}')
        os.makedirs(path, exist_ok=True)
    f = open(file, 'wb')
    f.write(img_bytes)

def preproc_img(img, sz):
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=0, crop_type=CropType.NO, tfm_y=None, sz_y=None)
    trans_img = val_tfm(img)
    print(f'Image shape: {trans_img.shape}')
    return Variable(torch.FloatTensor(trans_img)).unsqueeze_(0)
        
def get_file_with_ext(path, ext):
    if type(ext) == list:
        ext = tuple(ext)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(ext):
                return os.path.join(path, file)
    return None