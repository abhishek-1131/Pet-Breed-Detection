# Pet-Breed-Detection

Building an image classifier for an academic dataset (Oxford-IIIT Pet Dataset)

The dataset contains 37 distinct categories of pets, 25 dog and 12 cat breeds. Data loading,pre-processing and modelling performed using Pytorch and the fastai library.

Loading and viewing of data done by creating data objects (called databunches in fastai lib). The created model uses the resnet34 architecture. By pre-training the weights on the imagenet datset (transfer learning), only the fully connected layers on top are unfrozen and trained upon. This provided better results. Improving the learning part involved using the 1 cycle policy (which changes the learning rate over time) and by applying different learning rates to different parts of the network.Fine tuning was done by unfreezing the initial layers and training the whole network. 

Interpretation of results was done by plotting the confusion matrix and the top losses (using from_learner method in fastai and passing it to the model )
