# BehavioralCloning
Tutorial for building a model that generates steering angle based on image input. A brief summary of the workflow 
## Summary of Steps
### Exploratory Visualization of Dataset
* Pandas + SeaBorn + MatPlotLib to create , load and append dataset from dataframe
* Visualization to understand the distribution and quality of data. 
* Distribution Plot to see the spread and quantity of data
* Time Series Plot to understand the quality of data. ( To see noise to determine if filters are required)
<p align="center">
<img src= "Track1_SteeringAngle_Dist.png" width="750"/>
</p>

<p align="center">
<img src= "Track1vsTrack2.png" width="750"/>
</p>

### Data Collection based on Shortcomings
* Udacity Simulator and Udacity provided data.  
* Based on the histogram distribution plots collecting data by using certain driving styles (*Lesser data with large steering angles , then drive more on curves*, clockwise and anticlockwise )
* After initial model save and testing driving and training in problem areas to improve model on subset of data.

<p align="center">
<img src= "Udacity_StockData.png" width="750"/>
</p>

<p align="center">
<img src= "Cw_vsACW.png" width="750"/>
</p>

### Data Augmentation
* Augmentation using **Flipping**, **Translation** from left and right camera images
* Reduce the time spent on data gathering through data augmentation techniques 

### Data Perturbation to Increase Model Robustness

- **Brightness Perturbation** : Random perturbation of brightness of the image.
- **Gaussian Noise** : Blur filter with a random normal distribution across the image.
- **Adaptive Histogram Equalization** : Can greatly help in the model learning the features quickly
- **Colospace inversion** :  RBG to BGR colorspace change 

### Sampling and Image Generators 
These steps increase the challenge and generalization capability by creating harder images for the model to train on. Below is an example of augmented and perturbed image batch that is linked with the image generator that generates images during model training on the go. Seen below are the distribution of data and images of one random sample generated by the generator.

<p align="center">
<img src= "Sample_Distribution.png" width="750"/>
</p>

<p align="center">
<img src= "Sample_PreProcessed_Image_Batch.png" width="1500"/>
</p>

### Define model architecture
#### Data Pre-processing steps
 * Normalization through feature scaling 
 * Cropping region of interest
 * Resize image to increase model performance
  
#### Salient Features of Model
 * Batch Normalization before every activation
 * Overfitting prevention Dropouts and batch norm
 * NVIDIA End to End Model architecture and train from scratch
<p align="center">
<img src= "EndToEnd_NVIDIA.png" width="1500"/>
</p>

### Setup Model Training Pipeline
- **Hyperparameters**: **Epochs** , **Steps per Epoch** and **Learning Rate** decided based on search epochs on subset of data
- **Greedy best save** and **checkpoint** implementation.
- **Metrics** is a purely **loss** based. Since the label(Steering angle) here is numeric and non-categorical , RMS Loss is used as the loss type. 
### Save and Deploy Model
* Save using **json**, **hdf5** model.
