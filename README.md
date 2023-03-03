# CovFind

The COVID -19 detection using CT Scan images provides an excellent opportunity to
extend traditional healthcare methods to address COVID-19, thereby decreasing the time
taken to diagnose COVID-19 which helps in the process of Isolation and Treatment given
to the patient. Using Convolution neural network, the suspected patient’s CT scan can be
distinguished between a Non COVID patient and a COVID-19 patient. As COVID-19 is
an ongoing and new pandemic, the available datasets are insufficient and imbalanced to
train the model effectively, To deal with the data scarcity problem two strategies are used.

1) TransferLearning

where pre-trained neural network are re-trained to fit other dataset, This approach 
achieves greater performance with datasets of a limited size. The Literature Survey 
done above also shows that through transfer learning effective performance can be achieved 
when compared toany other method. Transfer Learning is an approach
The main application of transfer learning is the classification of
medical images for emerging diseases due to the limited availability of samples,
Transfer learning has the benefit that the training time of the model decreases and
is computationally less expensive as only a few layers are retrained. Since the
models are already trained, it does not require a vast amount of data.
The pre-trained models which are selected for creating diagnosis systems for
COVID-19 based on the literature survey done earlier are VGG19, Xception,
ResNet101V2. Here the knowledge gained by these pre-trained models which are
trained with a large amount of data is transferred to the new model to perform a
related task when the amount of data available to train is limited.

2) ImageAugmentation/DataAugmentation

Image augmentation is expanding the available dataset for training the model that
overcomes the data scarcity problem. It increases the number of samples in the
dataset by making slight variations in the already existing samples. For instance
we augmented our training CT images by flipping the image horizontally,
rescaling from (0,255) to (0,1), shearing the image by 0.2 degree and zooming in
the image by 0.2. Image Augmentation helps to reduce overfitting and serves as a
regularizer. The literature survey done above states that augmenting the dataset
can increase the accuracy of the model by comparing the accuracy achieved before
and after image augmentation.

## METHODOLOGY

 When we trained many models with our dataset for 11 epochs and batch size as 16,
 then we found that VGG19 and ResNet101V2 gave almost the same validation accuracy,
 but the validation loss for VGG19 was lower when compared to ResNet101V2.
 
 ![image](https://user-images.githubusercontent.com/76189053/222824035-9a46b793-4022-4f65-8373-b946c832e3b2.png)
![image](https://user-images.githubusercontent.com/76189053/222824105-ca135f90-1ad6-416e-b3d3-bfbd66785f2d.png)

When we plot the graph for training loss and validation loss, we can see that in ResNet101V2 the validation 
loss is increasing and the training loss is decreasing with the number of epochs, this indicates that the 
model is overfitting, i.e.The model starts to memorize the images instead of learning from the images, 
this could be due to very high number of layers (101 layers) ,i.e.The model is too complex.
Hence we go for the VGG19 as our pre-trained model. Now we need to increase the accuracy of the model, 
so we train the model for higher number of epochs till the validation loss is not increasing.

![image](https://user-images.githubusercontent.com/76189053/222825162-8cbd07bc-a662-4bee-aa94-695178eb3705.png)


Then we plot the validation loss and training loss with the batch size as 32, in-order to decrease the jitter

 ![image](https://user-images.githubusercontent.com/76189053/222824239-d2076118-9af6-4098-8396-6b54d97266df.png)

 From this graph we can see that the both the validation loss and training loss are in downward trend till 100 epochs
 and the validation loss starts to follow an upward trend after 100 epochs which indicates the overfitting of the model,
 so we need to train the model till 100 epochs only. 
 When we run the model for 100 epochs with the above mentioned parameters we get the validation accuracy around 97%
 
 ![image](https://user-images.githubusercontent.com/76189053/222824512-6d914aa2-89dd-46c3-9fbf-a7e6bad3c5e6.png)
 
 ## RESULT
 This shows that our model is still applicable with limited data, which is characteristic of the real situation,
 where large and diverse datasets may not be readily available. VGG 19, ResNet101V2 and Xception are the pre-trained 
 models that are tested in COVID-19 detection and found out that VGG19 performs well when compared to other models. 
 We had built an end-to-end system using deep learning mechanisms which can be utilized in the primary hotspots 
 to identify COVID-19 cases.

