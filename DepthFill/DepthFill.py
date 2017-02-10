import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os

N = 300
RGB_folder = "\\\\poplin\\tmp\\khoa\\DIW\\RGB\\"
DNNd_folder = "\\\\poplin\\tmp\\khoa\\DIW\\DNNdepth\\"
DNNds_folder = "\\\\poplin\\tmp\\khoa\\DIW\\DNNdepthscaled\\"
Knct_folder = "\\\\poplin\\tmp\\khoa\\DIW\\Knctdepth\\"

RGBfullsizeBMP_folder = "\\\\poplin\\tmp\\khoa\\DIW\\RGBfullsizeBMP\\"
RGBfullsize_folder = "\\\\poplin\\tmp\\khoa\\DIW\\RGBfullsize\\"
DEPTH_H = 424
DEPTH_W = 512

def preprocess_extract_dnn_depth(src_folder = RGB_folder, out_folder = DNNd_folder, size = (640, 480), out_file_prefix = "out_color", src_format = "jpg"):
    """"""
    for i in range(N):
        src_file = out_file_prefix + "00%03d." % i + src_format
        img = Image.open(src_folder + src_file)
        img.crop((size[0], 0, size[0]*2, size[1])).save(out_folder + src_file)
        #imgplot = plt.imshow(img)
        #plt.show()

def preprocess_extract_dnn_full_depth():
    preprocess_extract_dnn_depth(src_folder = "\\\\poplin\\tmp\\khoa\\DIW\\DNNout\\", size = (1920, 1080), out_file_prefix = "dnn_depth", src_format = "png")

def mass_rename(folder, old_name, new_name, format):
    for i in range(N):
        old_file = old_name + "00%03d." % i + format
        new_file = new_name + "00%03d." % i + format
        os.rename(folder + old_file, folder + new_file)


def test2():
    file1 = "seq_depth00000.png"
    file2 = "dnn_depth00000.jpg"
    img1 = Image.open(Knct_folder + file1)
    img2 = Image.open(DNNd_folder + file2)
    i = 10

def mapping():
    with open("Filedepth_20.bin", "rb") as filedepth:
        byte = filedepth.read()
        deps = np.fromstring(byte, dtype=np.uint16).reshape(424, -1)
    with open("FileMapp_20.bin", "rb") as filemapp:
        byte = filemapp.read()
        mapp = np.fromstring(byte, dtype=np.uint16).reshape(424, -1)  
    RGBim = Image.open("ds20.png")
    RGB = RGBim.load()
    RGBiDim = Image.new(RGBim.mode, (DEPTH_W, DEPTH_H), "black")
    RGBiD = RGBiDim.load()

    for i in range(DEPTH_H):
        for j in range(DEPTH_W):
            if deps[i, j] != 0:
                ii, jj = int(mapp[i, 2*j]), int(mapp[i, 2*j+1])
                if (ii < 1080) & (jj < 1920):                    
                     RGBiD[j, i] = RGB[jj, ii]
                    
                else: print("mapp for [{0} {1}] = {2} {3} OOB".format(i, j, ii, jj))
            else: 
                #print("deps[{0} {1}] = {2} <=0".format(i, j, deps[i, j]))
                RGBiD[j, i] = 0 #(0, 0, 0)
    #img = Image.fromarray(RGBiDim, 'RGB')
    #img.save('20.png')
    #img.show()
    RGBiDim.save('filled20.png')
    RGBiDim.show()
    i = 10

def positive_min(a):
    a = a.reshape(-1)
    min = a[0]+1
    for x in a:
        if 0 < x < min: min = x
    return min

def rescale(file_in, file_ref, file_out, debug = False):
    """ Rescales the depth values of a DNN-based depth image so that it fits with the actual measured depth image.
    Particularly, it rescales file_in so that its min and max values become the same with those of file_ref, and writes out into file_out
    file_in: dnn_fullsize, file_ref: seq_depth, file_out: dnn_fullscaled """
    img_in = Image.open(file_in)
    size_in = img_in.size
    mat_in = np.asarray(img_in)[:,:,0].reshape(-1).reshape(img_in.size)
    img_ref = Image.open(file_ref)
    mat_ref = np.asarray(img_ref).reshape(-1)
    img_out = Image.new(img_ref.mode, size_in, "black")
    mat_out = img_out.load()
    min_in, min_ref = positive_min(mat_in), positive_min(mat_ref)
    scale = (max(mat_ref) - min_ref) / (max(mat_in.reshape(-1)) - min_in)
    out = np.zeros_like(mat_in, dtype = np.uint16)
    for i in range(size_in[0]):
        for j in range(size_in[1]):
            out[i, j] = int( (mat_in[i, j] - min_in)*scale + min_ref )
    #out = np.asarray(mat_out, dtype = np.uint16)
    #img_out.save(file_out)
    cv2.imwrite(file_out, out)
    if debug: img_out.show()

def rescale2(file_in, file_ref, file_out, debug = False):
    im_in = cv2.imread(file_in)
    size_in = im_in.shape
    mat_in = np.asarray(im_in)[:, :, 0]
    im_ref = cv2.imread(file_ref)
    mat_ref = np.asarray(im_ref)[:, :, 0]
    #im_out = cv2.imread(file_ref)
    min_in, min_ref = positive_min(mat_in), positive_min(mat_ref)
    scale = (max(mat_ref.reshape(-1)) - min_ref) / (max(mat_in.reshape(-1)) - min_in)
    out = mat_in * scale
    #out = np.zeros_like(mat_in, dtype = np.uint16)

    #for i in range(size_in[0]):
    #    for j in range(size_in[1]):
    #        out[i, j] = int( (mat_in[i, j] - min_in)*scale + min_ref )
    cv2.imwrite(file_out, out)
    #cv2.imwrite("ref.png", mat_ref)


def rescale_unittest():
    """ Unit test for rescale"""
    rescale2(DNNd_folder + "dnn_depth00000.png", Knct_folder + "seq_depth00000.png", DNNds_folder + "dnn_dscaled00000.png")

def rescale_all():
    for i in range(54,N):
        rescale2(DNNd_folder + "dnn_depth00%03d.png" %i, Knct_folder + "seq_depth00%03d.png" %i, DNNds_folder + "dnn_dscaled00%03d.png" %i)



def convert_1(file_in, file_out = None):
    """ Convert image specified in file_in into file_out. If not specified, output is (file_in).PNG"""
    img = Image.open(file_in)
    if file_out == None:
        img.save(file_in[:file_in.find(".")] + ".png")
    else: 
        img.save(file_out)

def convert_RGBfullsizeBMP():
    for i in range(N):
        convert_1(RGBfullsizeBMP_folder + "KinectScreenshot_RGB%d.bmp" % i, RGB_folder + "rgb_full%05d.png" % i)

def filled_depth(file_dnn, file_ref, file_out, debug = False):
    im_in = cv2.imread(file_dnn)
    size_in = im_in.shape
    mat_in = np.asarray(im_in)[:, :, 0]
    im_ref = cv2.imread(file_ref)
    mat_ref = np.asarray(im_ref)[:, :, 0]

    #out = np.zeros_like(mat_in, dtype = np.uint16)
    out = mat_in

    for i in range(size_in[0]):
        for j in range(size_in[1]):
            if mat_in[i, j] == 0: out[i, j] = mat_ref[i, j]
            #else: out[i, j] =  mat_in[i, j]
    cv2.imwrite(file_out, out)

def fill_1():
    i = 20
    filled_depth(Knct_folder + "seq_depth00%03d.png" %i, DNNds_folder + "dnn_dscaled00%03d.png" %i, DNNd_folder + "filled_depth00%03d.png" %i)

def fill_all():
    for i in range(N):
        filled_depth(Knct_folder + "seq_depth00%03d.png" %i, DNNds_folder + "dnn_dscaled00%03d.png" %i, DNNd_folder + "filled_depth00%03d.png" %i)

# DONE preprocess_extract_dnn_depth()
# DONE mass_rename(DNNd_folder, "DNNdepth", "dnn_depth", "jpg")
#test()
#mapping()
#convert_RGBfullsizeBMP()
#preprocess_extract_dnn_full_depth()
#rescale_unittest()
#fill_1()
#rescale_all()
fill_all()
