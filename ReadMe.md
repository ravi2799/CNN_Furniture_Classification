# Furniture Classification

  Furniture classification is classification application that is buit using convolution neural network. I have used Mobilenet v2 to training. Furniture dataset has 3 classes which is chair, sofa and bed. Each folder contains 100 images. 

## How to train?
  
  Go to Script folder and Run:
  
   ```python train.py```
    
## For test the model:
  
  Load the checkpoint which is inside the checkpoints folder into infer script.
  Give path of image to be tested in __main__().
    
    ```python infer.py```
    
## For run API:
    
    ```python api.py```
    - it will run on 8383 port
    - You can test it using POST MAN. Link: --> http://0.0.0.0:8383/getPrediction
      Use body and upload image.
    
    
