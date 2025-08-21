#!/bin/bash

SCRIPTPATH=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
cd "$SCRIPTPATH"

# Install tong_system
git submodule update --init --recursive

pip install wheel # for venv (needed for pip install .)

pip install -r requirements.txt

pip install git+https://github.com/facebookresearch/pytorch3d.git@e3d3a67a89907476bd5b63289f9669bd427ae550

echo "----------------------------"
echo "Install ikfastpy?: (y/n)"
read input

if [ $input = 'yes' ] || [ $input = 'YES' ] || [ $input = 'y' ] ; then
    echo "  Installing ikfastpy."
    # sudo apt-get install liblapack-dev liblapack3 libopenblas-base libopenblas-dev
    cd tong_system/tongsystem/ikfastpy
    python setup.py build_ext --inplace
    cd "$SCRIPTPATH"
fi

cd tong_system
pip install -e .
cd "$SCRIPTPATH"

cd third_party/pointops
python setup.py install
cd "$SCRIPTPATH"

# Install gazebot
pip install -e .