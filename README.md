# traffic_sign_recognition

Design a target detection model based on the GTSRB dataset to recognize speed limit, no parking, and other signs on the road, providing environmental perception support for autonomous driving.

The model uses the deep deep neural network and gradient descent algorithm, 
          uses the pytorch package to train the model, 
          uses the intersection and union ratio of the box selected area and the actual area and the category accuracy as the model evaluation standard. 

The intersection and union ratio of the best model trained at present reaches 98% and the accuracy rate reaches 91%. 

Due to the use of the Data set is 
            Archive Meta Data (https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html), 
The type recognition accuracy of domestic traffic signs is low, but the box selection intersection ratio is high.
