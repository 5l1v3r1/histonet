"""
Neural network example built using pyBrain to classify images by histogram.

Copyright (C) 2012 - Guillermo Colmenero

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
from PIL import Image
from pybrain import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


NET = buildNetwork(768, 20, 1, bias=True, hiddenclass=TanhLayer) 
DATASET = SupervisedDataSet(768, 1)


def load_dataset(path, output):
    """
    Loads images contaied in an spesific folder and adds its histogram to the
    dataset.
    """
    print "Adding folder %s to dataset" % path
    try:
        listing = os.listdir(path)
        for infile in listing:
            img_path = os.path.join(path, infile) 
            print 'Processing file: %s' % img_path
            img = Image.open(img_path)
            histogram = img.histogram()
            if len(histogram) != 768:
                print "Error: Image is not RGB mode."
                
            else:
                print "Adding histogram to dataset."
                DATASET.addSample( histogram , (output,))
    except (IOError, OSError), ex:
        print "Error: %s" % ex


def use_network(path):
    """
    Activates the network.
    """
    print "Using network for files in folder %s." % path
    try:
        listing = os.listdir(path)
        for infile in listing:
            img_path = os.path.join(path, infile) 
            print 'Processing file: %s' % img_path
            img = Image.open(img_path)
            histogram = img.histogram()
            if len(histogram) != 768:
                print "Error: Image is not RGB mode."
                
            else:
                result = NET.activate(histogram)
                print "Result: %s" % result
    except (IOError, OSError), ex:
        print "Error: %s" % ex


def train_network():
    """
    Trains the network.
    """
    print 'Training network ...'
    trainer = BackpropTrainer(NET)
    trainer.trainUntilConvergence(dataset=DATASET, maxEpochs=None, verbose=None,
                                  continueEpochs=1, validationProportion=0.025)
        
if __name__ == "__main__":
    load_dataset('white/', 0)
    load_dataset('black/', 1)
    train_network()
    use_network('use/')
    

