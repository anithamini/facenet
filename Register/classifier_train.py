from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

from Source.packages.classifier import training
def classy():
    t2="_"
    datadir = './pre_img'
    modeldir = '../Model'
    t = time.localtime()
    t1 = time.asctime(t)
    print("current time: %s " % t1)
    t1 = t1.split(" ")
    t1=t1[-2].split(":")
    t2=t2.join(t1)
    print(t2)
    classi="classifier_"+str(t2)+".pkl"
    classifier_filename = '../Model/class/'+str(classi)
    print("clssname:",classifier_filename)
    print ("Training Start")
    obj=training(datadir,modeldir,classifier_filename)
    get_file=obj.main_train()
    print('Saved classifier model to file "%s"' % get_file)
    print("*************************************")
    #sys.exit("All Done")
#classy()