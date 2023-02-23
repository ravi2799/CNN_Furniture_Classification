import time
import torch

from sanic import Sanic
from sanic.response import json as j
from sanic_cors import CORS, cross_origin
from infer import DEPTH
from PIL import Image
import cv2
import time
import numpy as np
import os

import imghdr


path1 =r'C:\Users\ra_saval\Desktop\work\fulhas\checkpoints\checkpoint_37.ckpt'

app = Sanic("test")
CORS(app)


@app.route("/getPrediction", methods=['POST', 'GET'])
def predict(request):
        byt = request.files.get('data')
        check = request.form.get('name')
        print('WOW', check)
        if check != None:
            print('WOW', check)
        #print(byt)

        if byt == None:
            return j({"error_code" : "NO FILE SELECTED"})
        img_org = cv2.imdecode(np.frombuffer(byt.body, np.uint8), 3)
        
        print(img_org.shape)
        
        #INVALID IMAGE Check
        # print(imghdr.what('', h=byt.body))

        extension = imghdr.what('', h=byt.body)
        if extension == None:
            output = {"error_code":"INVALID IMAGE FILE"}
            return j(output)

        cv2.imwrite(r'C:\Users\ra_saval\Desktop\work\fulhas\Images\1.jpg',img_org)
        #INVALID IMAGE CHECK
        id_ = time.time()
        imageName = r'C:\Users\ra_saval\Desktop\work\fulhas\Images\{}.{}'.format(id_, extension)
        fi = open(imageName, 'wb')
        fi.write(byt.body)
        fi.close()

 
        #INFER
        dp = DEPTH(path1)
        img = cv2.imread(imageName)
        print(type(img))
        p_answer = dp.infer(img,id_)

        answer = '{}'.format(p_answer)

        output = {'Class' : str(answer)}
        print()
		
        return j(output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8383, debug=True)
