#
# written for Amazon Linux AMI
# creates an AWS Lambda deployment package for pytorch deep learning models (Python 3.6.1)
# assumes lambda function defined in ~/main.py
# deployment package created at ~/waya-ai-lambda.zip
#

#
# install python 3.6.1
#

sudo yum update
sudo yum install -y gcc zlib zlib-devel openssl openssl-devel

PYTHON_VERSION="3.6.5"

wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar -xzvf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION} && ./configure && make
sudo make install

#
# setup a minimal virtual environment for our lambda function's dependencies
#

sudo /usr/local/bin/pip3 install virtualenv
/usr/local/bin/virtualenv ~/shrink_venv
source ~/shrink_venv/bin/activate

ls $VIRTUAL_ENV/lib/python3.6/site-packages
du -sh $VIRTUAL_ENV/lib/python3.6/site-packages

#
# it's fine to install smaller python modules with pip (note `boto3` comes pre-installed in AWS Lambda environment)
#

pip install Pillow
pip install cython  # numpy dependency
pip install pyyaml  # pytorch dependency

#
# install numpy and pytorch from source to reduce package size
#

sudo yum install git

cd
git clone --recursive https://github.com/numpy/numpy.git
cd numpy
git checkout 31465473c491829d636c9104c390062cba005681  # latest release
python setup.py install


#
# and pytorch...
#

cd
sudo yum install cmake make automake gcc gcc-c++ kernel-devel  # pytorch build dependencies
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout af3964a8725236c78ce969b827fdeee1c5c54110
export NO_CUDA=1  # reduce package size (pointless b/c AWS Lambda does not have these capabilities anyways)
export NO_CUDNN=1
python setup.py install

pip install torchvision

#
# ensure the total size of our dependencies is under 250 MB (should be ~210 MB)
#

du -sh $VIRTUAL_ENV/lib/python3.6/site-packages

#
# create the deployment package
#

cd $VIRTUAL_ENV/lib/python3.6/site-packages
zip -r9 ~/waya-ai-lambda.zip *

cd
zip -g waya-ai-lambda.zip main.py