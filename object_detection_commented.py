# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
#open cv is not used for detection but here it is only used for drawing of the rectangles around dog.
import cv2
#VOC classes are used for mapping.
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
#used for processing of the images.
import imageio

# Defining a function that will do the detections
#1st is the image on which this detect function is going to be applied.
#2nd is the SSD Neural Network
#3rd is the transformation (correct format for the neural network)
def detect(frame, net, transform): # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
    
    #The shape will give us three outputs and last one is the number of channels i.e. for black and white frame it is 1 and for coloured image it is 3 - RGB
    height, width = frame.shape[:2] # We get the height and the width of the frame.
    
    frame_t = transform(frame)[0] # We apply the transformation to our frame.
    
    #Remember frame_t is a numpy array now we have transform it into torch tensor.
    #Here the neural network was trained under convention Green Red Blue so we need to make it into the correct format.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor.
    
    #unsqueeze function is just used before feeding the neural network , as neural nework works only on batch and not on the individual data.
    x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding to the batch.
    
    y = net(x) # We feed the neural network ssd with the image and we get the output y.
    
    #Tensor variable is composed of two elements torch tensor and gradient to get the torch tensor we are using .data
    # detections = [batch, number of class , number of occurance of the class, tuple of (score,x0,y0,x1,y1)] for each occurance of the object we get the score(or threshold) and the co-ordinates.
    detections = y.data # We create the detections tensor contained in the output y.
    
    #used for nomalization.
    #first width and height is uses for upper left corner and next one is used for lower right corner.
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height].
    
    
    #detections.size(1) will give us the number of classes.
    for i in range(detections.size(1)): # For every class:
        
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        
        while detections[0, i, j, 0] >= 0.6: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
            
            #openCV works with numpy array so we will change it back to numpy from torch tensor.
            pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.
            
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle.
            j += 1 # We increment j to get to the next occurrence.
    return frame # We return the original frame with the detector rectangle and the label around the detected object.

# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.

# Doing some Object Detection on a video
#Change name of the input file to generate an output for that corresponding input.
reader = imageio.get_reader('general_video.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('output_general.mp4', fps = fps) # We create an output video with this same fps frequence.

for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.