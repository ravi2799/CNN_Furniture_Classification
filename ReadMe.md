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
      
  
  ![image](https://user-images.githubusercontent.com/68138958/221064793-08b2d89c-a4d2-4f98-81df-3a8246159937.png)

      
## Dockerization
    
    You can create docker image using Dockerfile.txt.
    
      ** To build Image: **
        There is Dockerfile inside Scripts. Go to that folder and run the below command.
        docker build -f Dockerfile.txt -t name .
      
      ** To Run: **
        docker run -d -p 8383 --name testapp -t name
        
        It will run the api on 8383 port..
     
     ** To go inside Docker **
        
        docker exec -it testapp bash
        
        You can also run api here using python api.py
        
      
