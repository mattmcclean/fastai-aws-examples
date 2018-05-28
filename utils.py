import json, pdb, os, numpy as np, cv2, threading, math 
from urllib.request import urlopen

import torch
from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

classes = (
 '001.ak47',
 '002.american-flag',
 '003.backpack',
 '004.baseball-bat',
 '005.baseball-glove',
 '006.basketball-hoop',
 '007.bat',
 '008.bathtub',
 '009.bear',
 '010.beer-mug',
 '011.billiards',
 '012.binoculars',
 '013.birdbath',
 '014.blimp',
 '015.bonsai-101',
 '016.boom-box',
 '017.bowling-ball',
 '018.bowling-pin',
 '019.boxing-glove',
 '020.brain-101',
 '021.breadmaker',
 '022.buddha-101',
 '023.bulldozer',
 '024.butterfly',
 '025.cactus',
 '026.cake',
 '027.calculator',
 '028.camel',
 '029.cannon',
 '030.canoe',
 '031.car-tire',
 '032.cartman',
 '033.cd',
 '034.centipede',
 '035.cereal-box',
 '036.chandelier-101',
 '037.chess-board',
 '038.chimp',
 '039.chopsticks',
 '040.cockroach',
 '041.coffee-mug',
 '042.coffin',
 '043.coin',
 '044.comet',
 '045.computer-keyboard',
 '046.computer-monitor',
 '047.computer-mouse',
 '048.conch',
 '049.cormorant',
 '050.covered-wagon',
 '051.cowboy-hat',
 '052.crab-101',
 '053.desk-globe',
 '054.diamond-ring',
 '055.dice',
 '056.dog',
 '057.dolphin-101',
 '058.doorknob',
 '059.drinking-straw',
 '060.duck',
 '061.dumb-bell',
 '062.eiffel-tower',
 '063.electric-guitar-101',
 '064.elephant-101',
 '065.elk',
 '066.ewer-101',
 '067.eyeglasses',
 '068.fern',
 '069.fighter-jet',
 '070.fire-extinguisher',
 '071.fire-hydrant',
 '072.fire-truck',
 '073.fireworks',
 '074.flashlight',
 '075.floppy-disk',
 '076.football-helmet',
 '077.french-horn',
 '078.fried-egg',
 '079.frisbee',
 '080.frog',
 '081.frying-pan',
 '082.galaxy',
 '083.gas-pump',
 '084.giraffe',
 '085.goat',
 '086.golden-gate-bridge',
 '087.goldfish',
 '088.golf-ball',
 '089.goose',
 '090.gorilla',
 '091.grand-piano-101',
 '092.grapes',
 '093.grasshopper',
 '094.guitar-pick',
 '095.hamburger',
 '096.hammock',
 '097.harmonica',
 '098.harp',
 '099.harpsichord',
 '100.hawksbill-101',
 '101.head-phones',
 '102.helicopter-101',
 '103.hibiscus',
 '104.homer-simpson',
 '105.horse',
 '106.horseshoe-crab',
 '107.hot-air-balloon',
 '108.hot-dog',
 '109.hot-tub',
 '110.hourglass',
 '111.house-fly',
 '112.human-skeleton',
 '113.hummingbird',
 '114.ibis-101',
 '115.ice-cream-cone',
 '116.iguana',
 '117.ipod',
 '118.iris',
 '119.jesus-christ',
 '120.joy-stick',
 '121.kangaroo-101',
 '122.kayak',
 '123.ketch-101',
 '124.killer-whale',
 '125.knife',
 '126.ladder',
 '127.laptop-101',
 '128.lathe',
 '129.leopards-101',
 '130.license-plate',
 '131.lightbulb',
 '132.light-house',
 '133.lightning',
 '134.llama-101',
 '135.mailbox',
 '136.mandolin',
 '137.mars',
 '138.mattress',
 '139.megaphone',
 '140.menorah-101',
 '141.microscope',
 '142.microwave',
 '143.minaret',
 '144.minotaur',
 '145.motorbikes-101',
 '146.mountain-bike',
 '147.mushroom',
 '148.mussels',
 '149.necktie',
 '150.octopus',
 '151.ostrich',
 '152.owl',
 '153.palm-pilot',
 '154.palm-tree',
 '155.paperclip',
 '156.paper-shredder',
 '157.pci-card',
 '158.penguin',
 '159.people',
 '160.pez-dispenser',
 '161.photocopier',
 '162.picnic-table',
 '163.playing-card',
 '164.porcupine',
 '165.pram',
 '166.praying-mantis',
 '167.pyramid',
 '168.raccoon',
 '169.radio-telescope',
 '170.rainbow',
 '171.refrigerator',
 '172.revolver-101',
 '173.rifle',
 '174.rotary-phone',
 '175.roulette-wheel',
 '176.saddle',
 '177.saturn',
 '178.school-bus',
 '179.scorpion-101',
 '180.screwdriver',
 '181.segway',
 '182.self-propelled-lawn-mower',
 '183.sextant',
 '184.sheet-music',
 '185.skateboard',
 '186.skunk',
 '187.skyscraper',
 '188.smokestack',
 '189.snail',
 '190.snake',
 '191.sneaker',
 '192.snowmobile',
 '193.soccer-ball',
 '194.socks',
 '195.soda-can',
 '196.spaghetti',
 '197.speed-boat',
 '198.spider',
 '199.spoon',
 '200.stained-glass',
 '201.starfish-101',
 '202.steering-wheel',
 '203.stirrups',
 '204.sunflower-101',
 '205.superman',
 '206.sushi',
 '207.swan',
 '208.swiss-army-knife',
 '209.sword',
 '210.syringe',
 '211.tambourine',
 '212.teapot',
 '213.teddy-bear',
 '214.teepee',
 '215.telephone-box',
 '216.tennis-ball',
 '217.tennis-court',
 '218.tennis-racket',
 '219.theodolite',
 '220.toaster',
 '221.tomato',
 '222.tombstone',
 '223.top-hat',
 '224.touring-bike',
 '225.tower-pisa',
 '226.traffic-light',
 '227.treadmill',
 '228.triceratops',
 '229.tricycle',
 '230.trilobite-101',
 '231.tripod',
 '232.t-shirt',
 '233.tuning-fork',
 '234.tweezer',
 '235.umbrella-101',
 '236.unicorn',
 '237.vcr',
 '238.video-projector',
 '239.washing-machine',
 '240.watch-101',
 '241.waterfall',
 '242.watermelon',
 '243.welding-mask',
 '244.wheelbarrow',
 '245.windmill',
 '246.wine-bottle',
 '247.xylophone',
 '248.yarmulke',
 '249.yo-yo',
 '250.zebra',
 '251.airplanes-101',
 '252.car-side-101',
 '253.faces-easy-101',
 '254.greyhound',
 '255.tennis-shoes',
 '256.toad',
 '257.clutter'
)

id2cat = list(classes)
sz = 224

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
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
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

def preproc_img(img):
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=0, crop_type=CropType.NO, tfm_y=None, sz_y=None)
    trans_img = val_tfm(img)
    return Variable(torch.FloatTensor(trans_img)).unsqueeze_(0)