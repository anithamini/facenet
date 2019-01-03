# Introduction 
TODO: Face detection and recognition under multi camera environment. 

# Getting Started
TODO: 

1)Need to install python.
2)Download model(.pb) file and place it in model directory.
3)Download det1.py,det2.py,det3.py and place it in packages directory.
4)Create Train_data folder and save the images in a folder with the person names.

# Build and Test
TODO: 
1)Run initializer.py and check for the pre_image folder creation.
2)Run classifier.py and check the creation of (class -> classifier.pkl) in model directory.
3)Run dynamic_register.py in PC1 and register the person dynamically by tapping "R" key -> assign the name and wait for the new model to generate.
4)Share the model directory to PC2.
5)Run the inference.py code in PC2 parallelly with PC1 and check for the updations of new model continuously.
6)Now the assigned name during registration in PC1 should be shown in PC2,when a person detects otherwise Unknown label should be displayed if a person is not registered. 

