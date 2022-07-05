import json
import os
import bdcn
import torch
from torch.autograd import Variable
from datasets.dataset import Data
import numpy as np
import time
import cv2
from rock_detection.granulometry_stats import filters, midpoint, getContourStat
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import imutils
import math
import random


def test(data_root: str,
        config_file: str,
        trained_model: str,
        result_dir: str
    ):
    
    with open(config_file) as f:
        config = json.load(f)
    
    train_params = config["TRAINING_CONFIG"]["TRAIN_PARAMS"]
    eval_params = config["EVAL_CONFIG"]
    inference_params = config["INFERENCE_CONFIG"]

    for k, v in inference_params.items():
        print(k,v)
        
        
    # Training params
    yita = train_params["yita"]
    cuda = train_params["cuda"]
    gpu = train_params["gpu"]
    mean_bgr = [104.00699, 116.66877, 122.67892]
    # Inference params
    padding_value = inference_params['PADDING_VALUE']
    crop_dim = inference_params['CROP_DIM']
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    # Test Dataset
    test_lst = f'{inference_params["TEST_DATA_LST"]}'
    test_name_lst = f'{data_root}/test_dataset.lst'
    

    # Import model and use gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device', device)
    model = bdcn.BDCN()
    print('----------------------------',model)

    if cuda:
        model.load_state_dict(torch.load('%s' % (trained_model)))
        model.cuda()
    else:
        model.load_state_dict(torch.load('%s' % (trained_model), map_location=torch.device('cpu')))
    model.eval()
    
    # Load data
    test_img = Data(data_root, test_lst, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=4)
    nm = np.loadtxt(test_name_lst, dtype=str)
    print('The data is loaded!')
    print(len(testloader), len(nm))
    assert len(testloader) == len(nm)

    
    # Model Inference
    start_time = time.time()
    pred_time = dict(
        model_inference=[],
        filtering =[],
        postprocessing=[],
    )
    all_t = 0
    for i, (images, _) in enumerate(testloader):
        ti = time.time()
        if cuda:
            images = images.cuda()
        images = Variable(images)
        out = model(images)
        fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
        pred_time['model_inference'].append(time.time() - ti)

        if not os.path.exists(os.path.join(result_dir, 'fuse')):
            os.mkdir(os.path.join(result_dir, 'fuse'))
        if not os.path.exists(os.path.join(result_dir, 'fuse', nm[i])):
            os.mkdir(os.path.join(result_dir, 'fuse', nm[i]))
        out_path = os.path.join(result_dir, 'fuse', nm[i])
        pred_path = os.path.join(out_path, '%s.png' % nm[i])
        cv2.imwrite(pred_path, fuse*255)
        
        
        # Post-processing
        tp = time.time()
        filtered = filters(pred_path)
        pred_time['filtering'].append(time.time() - tp)

        tp = time.time()

        row, column = filtered.shape
        filtered_path = os.path.join(result_dir, 'fuse', nm[i], '%s.png' % nm[i])
        cv2.imwrite(f'{out_path}/{nm[i]}_filtered.png', filtered)

        # Detecting countours in the binary image
        _,edges=cv2.threshold(filtered,50,255,cv2.THRESH_BINARY)
        cnts= cv2.findContours(np.uint8(edges.copy()), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        (cnts, _) = imutils.contours.sort_contours(cnts) # sort the contours from left-to-right

        SizeList=[] #creating the list for every size of every object
        count=0

        result_image = 255 * np.ones((row,column,3), np.uint8)

        for c in cnts:
            meanValue , stand  = getContourStat(cnts , edges , count) # Erasing a bit of sound in the image

            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = imutils.perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            dim=dA
            area=(cv2.contourArea(c)) #measuring area

            if area>0: # If area exists

                # Calculating the y value for the centroid of the contour
                relation=(dA*dB*math.pi/4)/area #measuring the relation between actual area and rectangular area of the contour
                filter1=0.002 #2
                filter2=950 #95
                if dA<row/filter1 and dB<row/filter1 and dA<column/filter1 and dB<column/filter1\
                and dA>row/filter2 and dA>column/filter2 and dB>row/filter2 and dB>column/filter2: # Filtering particles too small or too big
                    if relation<100: # Filtering irregular sized rocks #1.34
                        if any(i>100 for i in stand.flatten()) or any(j>100 for j in meanValue.flatten()): #100
                            myname="hola"
                        else:
                            # compute the size of the object
                            # area = pi*(d^2)/4
                            SizeList.append((4*area/math.pi)**(1/2))  #adding size to a list of sizes
                            segm_mask = cv2.drawContours(result_image,cnts,count,(random.randint(0,256),random.randint(0,256),random.randint(0,256)),-1)

            count+=1
        
        cv2.imwrite(f'{out_path}/{nm[i]}_result.png', result_image)
        print(f'There are {count} contours detected.')
        SizeList.sort()
        numberofparticles=len(SizeList)
        print(f"there are {numberofparticles} rock particles ")
        print(f"{count-numberofparticles} rocks were eliminated")

        pred_time['postprocessing'].append(time.time() - tp)
        
    
    print(sum(pred_time['model_inference']) / len(pred_time['model_inference']))
    print(sum(pred_time['postprocessing']) / len(pred_time['postprocessing']))
    print(sum(pred_time['filtering']) / len(pred_time['filtering']))

if __name__ == "__main__":
    test(
        data_root='data',
        config_file='pipeline-config.json',
        trained_model='models_old/bdcn_pretrained_on_bsds500.pth',
        result_dir='test_results'
    )
