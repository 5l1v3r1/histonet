"""
Neural network example built using pyBrain to classify images by histogram.
2012 - Guillermo Colmenero
"""
import os
from PIL import Image
from pybrain import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

#10 Neurons in the hidden layer works fast but it's inaccurate.
NET = buildNetwork(768, 10, 1, bias=True, hiddenclass=TanhLayer) 
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
                                  continueEpochs=10, validationProportion=0.25)
    
        
if __name__ == "__main__":
    load_dataset('white/', 0)
    load_dataset('black/', 1)
    train_network()
    use_network('use/')
    

