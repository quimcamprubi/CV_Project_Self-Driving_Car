import cv2
import coppelia_sim.sim as sim
import numpy as np
import os 
from os import listdir
from os.path import isfile, join                                            
import matplotlib.pyplot as plt                  
import cv2             
import tensorflow as tf                          
from PIL import Image                                                                  
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout                                    
import warnings
from sklearn.metrics import accuracy_score
import skimage.measure

def im2double(im):
    out = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return out

def pipeline(img, s_thresh=(125, 255), sx_thresh=(50, 255)):
    img = np.copy(img)
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    
    # Saturation threshold
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Sobel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Horizontal derivative
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Horizontal gradient threshold
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(original, img, dst_size):
    height,width = img.shape

    src = np.float32([(120, 200), (38, 255), (460, 255), (380, 200)])
    dst = np.float32([(128,0), (128, height), (384,height), (384,0)])
    """src1 = (120, 200)
    src2 = (38, 255)
    src3 = (460, 255)
    src4 = (380, 200)
    src_points = np.array([[src1, src2, src3, src4]]).astype('float32')"""
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, dst_size)
    """# Draw source points on original image
    src_image = np.copy(original)
    src_image = cv2.polylines(src_image, src_points.astype('int32'), 1, (255,0,0), thickness=6)
    
    plt.figure(8)
    plt.imshow(src_image)
    plt.show()"""
    return warped, M, Minv

def inv_perspective_warp(img, dst_size):
    height,width,_ = img.shape

    dst = np.float32([(120, 200), (38, 255), (460, 255), (380, 200)])
    src = np.float32([(128,0), (128, height), (384,height), (384,0)])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []


def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)
            # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a polynomial function to each line
    if len(lefty) == 0 or len(leftx) == 0 or len(righty) == 0 or len(rightx) == 0:
        success = False
        return -1, (-1, -1), (-1, -1), -1, success
    success = True
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    left_a.clear()
    left_b.clear()
    left_c.clear()
    right_a.clear()
    right_b.clear()
    right_c.clear()

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty, success

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))
    color_size = (color_img.shape[1], color_img.shape[0])
    inv_perspective = inv_perspective_warp(color_img, color_size)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

def get_dataset(crk3, clientID, counter_i):
    retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    #retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_buffer)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    while not resolution:
        retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    
    frame=np.array(frame,dtype=np.uint8)
    frame.resize([resolution[1],resolution[0],3])
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 1)
    plt.figure(1)
    plt.imshow(frame)
    plt.show()
    plt.imsave('claw_rover_k3/movement/signs_dataset/4/image4_'+str(counter_i)+'.png', frame)
    counter_i += 1
    return counter_i

def cut_sign(im):
    minimi = 256
    maximi = 0
    minimj = 256
    maximj = 0
    imf = im[:,:,0]
    '''
    for i in range(256):
        for j in range(256):
            if (im[i,j,0]>0.4 and im[i,j,1] < 0.15 and im[i,j,2] < 0.15):
                imf[i,j] = 1
            else:
                imf[i,j] = 0
    '''
    dim0 = im[:,:,0] > 0.4
    dim1 = im[:,:,1] < 0.20
    dim2 = im[:,:,2] < 0.20
    dim_t = dim0 * dim1 * dim2
    imf = dim_t.astype('uint8')
    
    labeled_image = skimage.measure.label(imf, connectivity=2, return_num=True)
    
    counter = np.histogram(labeled_image[0], labeled_image[1]+1)[0]
    
    maxim = 0
    idmaxim = -1  
    for i in range(1, labeled_image[1]+1):
        if maxim < counter[i]:
            maxim = counter[i]
            idmaxim = i
        
    trobat1 = False
    trobat2 = False
    trobat3 = False
    trobat4 = False
    for i in range(256):
        for j in range(256):
            if (labeled_image[0][i][j] == idmaxim):
                if i < minimi: minimi = i; trobat1 = True
                if i > maximi: maximi = i; trobat2 = True
                if j < minimj: minimj = j; trobat3 = True
                if j > maximj: maximj = j; trobat4 = True  
    
    if trobat1 and trobat2 and trobat3 and trobat4:
        sign = im[minimi:maximi,minimj:maximj,:]
    else:
        sign = 0     
    return sign

def model_detect_signs():
    data = []
    labels = []
    classes = 4
    
    for i in range(0,classes):
        '''llegir imatges'''
        path = os.path.join(os.getcwd(),'claw_rover_k3\\movement\\signs_dataset\\'+str(i+1),'retalls')
        images = os.listdir(path)
        
        for j in images:
            try:
                image = Image.open(path + '\\'+ j).convert('RGB')
                image = np.array(image)
                image = cv2.resize(image, dsize=(30, 30), interpolation=cv2.INTER_CUBIC)
                image = image/255
                data.append(image)
                temp_vect = [0,0,0,0]
                temp_vect[i] = 1
                labels.append(temp_vect)
            except:
                print("Error loading image")
                
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape, labels.shape)
    
    
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=68)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=68)
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.05))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.05))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.05))
    model.add(Dense(4, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=8, epochs=25, validation_data=(X_val, y_val))
    model.save("claw_rover_k3/movement/Trafic_signs_model.h5")
    #plotting graphs for accuracy 
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    #plotting graphs for loss 
    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    
    
    score = model.evaluate(X_test,y_test,verbose=0)
    print('Test Score = ',score[0])
    print('Test Accuracy =', score[1])




def detect_signs(image, model):
    
    image = cut_sign(image)
    retorn = -1
    if type(image) is not int:
        if image.shape[0] > 40 and image.shape[1] > 40:
            image = cv2.resize(image, dsize=(30, 30), interpolation=cv2.INTER_CUBIC) 
            image = np.expand_dims(image, axis=0)
            pred = model.predict([image])[0]
            idpred = np.argmax(pred)
            if pred[idpred] > 0.7:
                if idpred == 0:
                    retorn = 0
                elif idpred == 1:
                    retorn = 5
                elif idpred == 2:
                    retorn = 8
                elif idpred == 3:
                    retorn = 15

    return retorn


def IsTrash(im):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
   
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 250
    params.maxArea = 500
    
    #En cas de problemes detectant senyals com a brossa
    #params.maxArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(im)
    
    # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    # plt.figure()
    # plt.imshow(im_with_keypoints)
    
    
    if keypoints == []:
        return False
    return True


def Detect(frame):
    trash = -1

    Hth = int(np.size(frame,0) / 2)
    Wth = int(np.size(frame,1) / 2)
    
    left = frame[:Hth,:Wth]
    #plt.imshow(left)
    
    right = frame[:Hth,Wth:]
    #plt.imshow(right)
    
    if IsTrash(left):
        trash+=1

    if IsTrash(right):
        trash+=2

    return trash


def AnalyzeFrameAdaptative(crk3, clientID, prev_offset):
    # Read frame
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_oneshot_wait)
    #retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_buffer)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    while not resolution:
        retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_oneshot_wait)
    
    frame=np.array(frame,dtype=np.uint8)
    frame.resize([resolution[1],resolution[0],3])
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 1)


    height,width,channels = frame.shape
    dst = pipeline(frame)

    kernel = np.ones((1,1), np.uint8)
    dst = cv2.erode(dst,kernel,iterations = 1)

    dst_size = (dst.shape[1], dst.shape[0])
    dst, M, Minv = perspective_warp(frame, dst, dst_size)

    out_img, curves, lanes, ploty, success = sliding_window(dst)
    if success == False:
        return prev_offset, -1
    left = curves[0][len(curves[0])-1]
    right = curves[1][len(curves[1])-1]
    midpoint = (right+left)/2
    car_position = width/2
    offset = car_position - midpoint

    # switch trash =
    # -1: No detected trash
    #  0: Trash detected to the left
    #  1: Trash detected to the right
    #  2: Trash detected to the right and left

    detected_trash = Detect(frame)
    return offset, detected_trash

def AnalyzeFrameReal(crk3, clientID, prev_offset):
    # Read frame
    speed = 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_oneshot_wait)
    #retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_buffer)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    model = load_model('claw_rover_k3/movement/Trafic_signs_model.h5')
    while not resolution:
        retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_oneshot_wait)
    
    frame=np.array(frame,dtype=np.uint8)
    frame.resize([resolution[1],resolution[0],3])
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 1)


    height,width,channels = frame.shape
    dst = pipeline(frame)

    kernel = np.ones((1,1), np.uint8)
    dst = cv2.erode(dst,kernel,iterations = 1)

    dst_size = (dst.shape[1], dst.shape[0])
    #dst = change_perspective(dst, dst_size)
    dst, M, Minv = perspective_warp(frame, dst, dst_size)

    out_img, curves, lanes, ploty, success = sliding_window(dst)
    if success == False:
        return prev_offset, -1, -1
    left = curves[0][len(curves[0])-1]
    right = curves[1][len(curves[1])-1]
    midpoint = (right+left)/2
    car_position = width/2
    offset = car_position - midpoint

    # Here we will detect traffic signs and trash
    
    retCode, resolution, frameb = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    while not resolution:
        retCode, resolution, frameb = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    
    frameb=np.array(frameb,dtype=np.uint8)
    frameb.resize([resolution[1],resolution[0],3])
    frameb = cv2.rotate(frameb, cv2.ROTATE_90_CLOCKWISE)
    frameb = cv2.flip(frameb, 1)
    frameb = im2double(frameb)
    
    speed = detect_signs(frameb, model)
            
    # switch trash =
    # -1: No detected trash
    #  0: Trash detected to the left
    #  1: Trash detected to the right
    #  2: Trash detected to the right and left

    detected_trash = Detect(frame)
    #detected_trash = -1
    return offset, detected_trash, speed


#Same code with comments and console messages
'''

def AnalyzeFrame(crk3, clientID, prev_offset):
    # Read frame
    speed = 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_oneshot_wait)
    #retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_buffer)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    model = load_model('claw_rover_k3/movement/Trafic_signs_model.h5')
    while not resolution:
        retCode, resolution, frame = sim.simxGetVisionSensorImage(clientID, crk3.picam, 0, sim.simx_opmode_oneshot_wait)
    
    frame=np.array(frame,dtype=np.uint8)
    frame.resize([resolution[1],resolution[0],3])
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 1)
    #frame = frame[:, :, :]

    height,width,channels = frame.shape
    dst = pipeline(frame)

    kernel = np.ones((1,1), np.uint8)
    dst = cv2.erode(dst,kernel,iterations = 1)

    dst_size = (dst.shape[1], dst.shape[0])
    #dst = change_perspective(dst, dst_size)
    dst, M, Minv = perspective_warp(frame, dst, dst_size)

    out_img, curves, lanes, ploty, success = sliding_window(dst)
    if success == False:
        return frame, prev_offset, -1, -1
    left = curves[0][len(curves[0])-1]
    right = curves[1][len(curves[1])-1]
    midpoint = (right+left)/2
    car_position = width/2
    offset = car_position - midpoint

    #curverad = get_curve(frame, curves[0], curves[1])
    img = draw_lanes(frame, curves[0], curves[1])

    # Here we will detect traffic signs and trash
    '''
'''
    retCode, resolution, frameb = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    while not resolution:
        retCode, resolution, frameb = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    
    frameb=np.array(frameb,dtype=np.uint8)
    frameb.resize([resolution[1],resolution[0],3])
    frameb = cv2.rotate(frameb, cv2.ROTATE_90_CLOCKWISE)
    frameb = cv2.flip(frameb, 1)
    frameb = im2double(frameb)
    
    speed = detect_signs(frameb, model)
    '''
'''
    retCode, resolution, frameb = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    if retCode == -1: raise ValueError("Couldn't retrieve the frame.")
    while not resolution:
        retCode, resolution, frameb = sim.simxGetVisionSensorImage(clientID, crk3.picam2, 0, sim.simx_opmode_oneshot_wait)
    
    frameb=np.array(frameb,dtype=np.uint8)
    frameb.resize([resolution[1],resolution[0],3])
    frameb = cv2.rotate(frameb, cv2.ROTATE_90_CLOCKWISE)
    frameb = cv2.flip(frameb, 1)
    frameb = im2double(frameb)
    speed = detect_signs_simple_version(frameb)
    '''
'''
    model_detect_signs()
    '''
    
    
'''
    #counter_i = get_dataset(crk3, clientID, counter_i)
    for i in range(1,50):
        imatge = plt.imread('claw_rover_k3/movement/signs_dataset/4/imatge4 ('+str(i)+').png')
        imatge = cut_sign(imatge)
        if type(imatge) != int:
            plt.imsave('claw_rover_k3/movement/signs_dataset/4/retalls/imatge4_'+str(i)+'.png', imatge)
    '''
        
'''
    #Detect presence of trash nearby
    trash = Detect(frame)
    
    # print("--------------------------------------")
    # if trash == -1:
    #     print("No")
    # elif trash == 0:
    #     print("Esquerra")
    # elif trash == 1:
    #     print("Dreta")
    # elif trash == 2:
    #     print("Ambdos costats")
    # print("--------------------------------------")
    
    
    # switch trash =
    # -1: No detected trash
    #  0: Trash detected to the left
    #  1: Trash detected to the right
    #  2: Trash detected to the right and left
    
    # PLACEHOLDERs
    detected_trash = False
    if trash == 1:
        detected_trash = True
    '''
'''    
    
    detected_trash = Detect(frame)
    return img, offset, detected_trash, speed
'''