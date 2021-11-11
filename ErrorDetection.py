import os
import numpy as np
import pandas as pd
from PIL import Image as PImage
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from detect import run

def error_detection(image_path, true_count, dataf, a):

    thresh = 0.20

    # run model detection
    count = run(weights="best.pt", 
                imgsz=[416, 416], 
                source=image_path, 
                return_result=True, 
                conf_thres=thresh,
                save_txt=True)
  
    count1=count.copy()
    dataf.append(count1)    
    print(count)

    image = PImage.open(image_path)
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
    # c=0 is basically a variable. We will only increase its value if the true count is equal to count for any of the four objects
    # For example: if nut is not the same then c will stay 0, if the value of screw is the same then c will be 1
    #              if bracket is not same then c will stay at 1 and if same then c will be 2 
    c=0
    cc=a
    dd=0
    for obj in count.keys():
        if count[obj] != true_count[obj]:
            diff = true_count[obj]-count[obj]
            if diff>=0:
                print("Error! {} {} is/are missing".format(diff,obj))
                dataf.append("Error! {} {} is/are missing".format(diff,obj))
                # count[obj] = count[obj]+diff
                    
            else:
                print("Error! {} {} is/are excess".format(diff*-1,obj))
                dataf.append("Error! {} {} is/are excess".format(diff*-1,obj))
                # count[obj] = count[obj] - (diff*-1)
            
        else:
            # count=true_count for all the categories
            c=c+1

    for obj in count.keys():
        
        if count[obj] != true_count[obj]:
            diff = true_count[obj]-count[obj]
            if obj=="other" and diff!=0:
                print("Remove the other/unknown object and then move to the staging area")
                dataf.append("Remove the other/unknown object")
                break
            if diff>0:
                #print("Error! {} {} are missing".format(diff,obj))
                #count[obj] = count[obj]+diff
                print(" {} {} added to the CURRENT BIN from the PARTS BIN ".format(diff,obj))
                dataf.append(" {} {} added to the CURRENT BIN from the PARTS BIN ".format(diff,obj))
            else: 
                #print("Error! {} {} are excess".format(diff*-1,obj))
                #count[obj] = count[obj] - (diff*-1)
                print(" {} {} removed from the CURRENT BIN to the PARTS BIN".format(diff*-1,obj))
                dataf.append(" {} {} removed from the CURRENT BIN to the PARTS BIN".format(diff*-1,obj))
  
    if c==len(count):
        # Here basically we are checking if the length of the count is 4 and if it is 4 then it means that the
        # true count and count is same for all the 4 parts. In that case we can move to the staging area
        print("Entire order is here, move to the staging area")
        dataf.append("Entire order is here, move to the staging area")
        # cc=a=0, so now cc=0+1=1. Then it will perform image processing again in the next cell
        cc=cc+1

    return (count, image_np, cc, dd) # ignore dd but basically we are returning these values from the error detection function

#add new variable to pred_images, like what you already have example: im_path
#def pred_images(im_path,t_count,dataf,order):

#order is a class object that contains the number of I,L, and T parts we need.
#order is how many exact parts we need. Intead of putting in a function, put it in a variable and then pass that variable into a function
#Inside of pred_images it figures out how many of each part we need by pulling that information
#out of the order variable

#Created a class called order and then basically assigned values to an instance variable: nut, screw, bracket, other
class order:
  def __init__(self, I, L, T):
    self.I = I
    self.L = L
    self.T = T

def pred_images(im_path, t_count, dataf):
  
    # Checking if its a directory or a single image 
    error = pd.DataFrame(columns = list(t_count.keys()))
    if os.path.isdir(im_path):
    # We go to the else part basically if its a single image. 
    # For both scenarios, it asks the user for the input and then goes to the error detection function 
    # Dataf.append is basically a data frame that will later be converted into a txt form
        images = os.listdir(im_path)
        img_names = []
        for img in images:
        
            try:
                a=0
                b=0
                if img.endswith("jpg"):
                
                    img_path = im_path+"/"+img
                    print('Name:{}'.format(img))
                    dataf.append('Name:{}'.format(img))
                    dataf.append("I needed: {}".format(t_count["I"]))
                    dataf.append("L needed: {}".format(t_count["L"]))
                    dataf.append("T needed: {}".format(t_count["T"]))
                    count,image_np,c1,d1 = error_detection(img_path,t_count,dataf,a)
                    if c1==1: # C1 is a flag to re check if this is 1 then we have the 4th scenario 
                        print("Performing image processing again")
                        del dataf[-1]
                        del dataf[-1]
                        del dataf[-1]
                        del dataf[-1]
                        del dataf[-1]
                        del dataf[-1]
                        dataf.append(("Performed image processing again"))
                        dataf.append("I needed: {}".format(t_count["I"]))
                        dataf.append("L needed: {}".format(t_count["L"]))
                        dataf.append("T needed: {}".format(t_count["T"]))      # calling error detection function again for rechecking
                        count,image_np,c2,d1 = error_detection(im_path,t_count,dataf,a) # a is initially 0 but if true count and count are the same then a becomes 1 and #d1 is just variable for any further changes
                    dataf.append("-------------------------------------------------------------")
            except :
                continue
        error["Image"] = img_names
        #error.to_csv("/content/output/error.txt",index=False)
    else :
        a=0  # this is for the 4th scenario
        b=0  # we don't need b
        print('Name:{}'.format(im_path))
        dataf.append("Name: {}".format(im_path))
        dataf.append("I needed: {}".format(t_count["I"]))
        dataf.append("L needed: {}".format(t_count["L"]))
        dataf.append("T needed: {}".format(t_count["T"]))
        
        count2, image_np, c1, d1 = error_detection(im_path, t_count, dataf, a)
        if c1==1:
            print("\n")
            print("Performing image processing again")
            print("\n")
            del dataf[-1] #deleting wrong values ie values before double check so that we only get the latest values on the txt file
            del dataf[-1]
            del dataf[-1]
            del dataf[-1]
            del dataf[-1]
            del dataf[-1]
            dataf.append(("Performed image processing again"))
            dataf.append("I needed: {}".format(t_count["I"]))
            dataf.append("L needed: {}".format(t_count["L"]))
            dataf.append("T needed: {}".format(t_count["T"]))
            
            count, image_np, c3, d1 = error_detection(im_path, t_count, dataf, a) # rechecking, basically performing image processing again
                
        dataf.append("-------------------------------------------------------------")
        plt.figure(figsize=(12, 8))
        plt.imshow(image_np)
    return dataf
# Storing the old scenarios in dq so everything prints out in the txt file one after another
if __name__ == "__main__":
    dq=[]

    #orderNumber1 = order(2, 4, 6)
    #orderNumber2 = order(1, 0, 0)
    #orderNumber3 = order(1, 0, 0)

    #true_count = {"I":orderNumber3.I, "L":orderNumber3.L, "T":orderNumber3.T}

    orderNumber1 = order(2, 4, 6)
    orderNumber2 = order(3, 5, 8)
    orderNumber3 = order(1, 0, 0)

    #true_count = {"I":orderNumber1.I,"L":orderNumber1.L,"T":orderNumber1.T}
    true_count = {"I":orderNumber2.I,"L":orderNumber2.L,"T":orderNumber2.T}
    #true_count = {"I":orderNumber3.I,"L":orderNumber3.L,"T":orderNumber3.T}

    dataf=dq

    # image with multiple parts
    #img_path = "/content/yolov5/20211102_004826.jpg"
    #for filename in os.listdir("data/images"):

    img_path = "data/images/I-shape.jpg"

    pred_images(img_path,true_count,dataf)
    # Put the data in the df and then output the df in a txt file. Basically we have everything stored in dataf
    df = pd.DataFrame(dataf)
    # Here dq has the old data and dataf has the new data
    dq = dataf
    df.to_csv("error.txt", index=False) 
