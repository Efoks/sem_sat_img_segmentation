Class coding: 
- 0: No information 
- 1: Urban fabric 
- 2: Industrial, commercial, public, military, private and transport units 
- 3: Mine, dump and contruction sites 
- 4: Artificial non-agricultural vegetated areas 
- 5: Arable land (annual crops) 
- 6: Permanent crops 
- 7: Pastures 
- 8: Complex and mixed cultivation patterns 
- 9: Orchards at the fringe of urban classes 
- 10: Forests 
- 11: Herbaceous vegetation associations 
- 12: Open spaces with little or no vegetation 
- 13: Wetlands 
- 14: Water 
- 15: Clouds and shadows 

Region codes: 
- D029 - Brest
- D014 - Caen
- D059 - Celais Dunkerque
- D072 - LeMans
- D056 - Lorient
- D022 - St. Brieuc
- D044 - Nazaire
- D006 - Nice

---

The data is not pushed into GitHub because of the size limit, it can be found here: 

---

## Python files:
- __data_handling.py__ - Defines custom PyTorch dataset, MiniFranceDataset, for image segmentation using 
MiniFrance dataset. It loads images and corresponding masks, supports data transformations, 
and handles supervised and unsupervised data loading. 
The create_data_loaders function generates supervised and unsupervised data loaders, with an option to 
specify the ratio of unsupervised samples.
- __data_processing.py__ - This code is part of a data preparation process for satellite imagery semantic segmentation. 
It prepares the data by cutting large satellite images into smaller sub-images, ensuring they meet specific criteria, 
and saving the sub-images. The process is applied to images from various regions, and both images with associated masks 
and images without masks are processed. The secondary run ensures that the total number of processed images meets 
a specified maximum limit.
- __model_test.py__ - Defines a mock dataset and evaluation process for a DeepLabV3. 
It is designed for testing purposes in a controlled environment. The mock dataset generates random input data and 
corresponding target data for a specified number of samples and classes. The code then trains and tests a neural network
model with the mock dataset. The model is optimized using the Adam optimizer and evaluates its performance based 
on the mean Intersection over Union (IoU) score.
- __config.py__ - This code defines various configuration parameters and directories. The print_config function is used 
to print out the project's configuration settings for reference.
- __utils.py__ - The code includes a set of utility functions for visualizing and evaluating 
image semantic segmentation models in PyTorch.
- __Other python files__ - Work in progress or play-testing playgrounds.