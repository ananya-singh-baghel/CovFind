# CovFind

The COVID -19 detection using CT Scan images provides an excellent opportunity to
extend traditional healthcare methods to address COVID-19, thereby decreasing the time
taken to diagnose COVID-19 which helps in the process of Isolation and Treatment given
to the patient. Using Convolution neural network, the suspected patient’s CT scan can be
distinguished between a Non COVID patient and a COVID-19 patient. As COVID-19 is
an ongoing and new pandemic, the available datasets are insufficient and imbalanced to
train the model effectively, To deal with the data scarcity problem two strategies are used

1) TransferLearning

where pre-trained neural network are re-trained to fit other dataset, This approach achieves
greater performance with datasets of a limited size. The Literature Survey done above 
also shows that through transfer learning effective performance can be achieved when compared to
any other method. Transfer Learning is an approach
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

##RESULT

 When we trained many models with our dataset for 11 epochs and batch size as 16,
 then we found that VGG19 and ResNet101V2 gave almost the same validation accuracy,
 but the validation loss for VGG19 was lower when compared to ResNet101V2
