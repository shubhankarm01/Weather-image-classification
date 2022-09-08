# Weather image classification

The project was part of the hackathon conducted by the IITB Analytics club on DPhi. The objective was to classify the weather images into 5 different classes. In this project, different approaches for image classifications were explored.

In the first approach, a CNN was constructed from scratch and was used for the modelling. Image augmentation with the ImageDataGenerator module from Keras was performed to increase the number of data for the training. After adding the augmented image to the training data, a TensorFlow sequential CNN was constructed. Different activation functions in the hidden layer such as sigmoid, relu and tanh were employed, and their impact on the performance was observed. Adam optimizer and categorical_crossentropy loss function were used for training.

In the second approach, HuggingFace pretrained-model (google/vit-base-patch16-224) was used for the classification. Both Tensorflow and Pytorch frameworks were separately used to understand the implementation of pre-trained models. AutoFeatureExtractor was used for the transformation of the data into desired input formats. DataCollator was employed to create batches of the image data for training and prediction. The input data was fed to the pre-trained model, and the last hidden layer was connected dense classification layer with 5 outputs and softmax activation. All the layers from the pre-trained model were frozen and only the last layer was trained in the model. The same approach was used in Pytorch implementation, but Class object was used to perform the same.

In the last approach, vgg16 pre-trained models from TenforFlow-Keras application module was used. In this, ImageDataGenerator was used for creating batches of the input images in the required format. A sequential model with vgg16 as the first layer, two dense hidden layers and a classification layer with 5 outputs was constructed. The layers inside vgg16 were frozen and the rest were trained. 

The results from all the approaches were compared. The project was to understand the implementation of different frameworks and techniques. At last, test data was used for the prediction.
