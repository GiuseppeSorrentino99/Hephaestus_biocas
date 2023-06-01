from __future__ import print_function
import SimpleITK as sitk
import pydicom
import cv2
import glob

import registration_callbacks
import os
import time
import numpy as np
import math
import time
import pandas as pd
import argparse
import re


# import matplotlib.pyplot as plt
# from ipywidgets import interact, fixed
#from IPython.display import clear_output
OUTPUT_DIR = 'Output'
dim = 512
################################################################################
# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
# def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
#     # Create a figure with two subplots and the specified size.
#     plt.subplots(1,2,figsize=(10,8))
    
#     # Draw the fixed image in the first subplot.
#     plt.subplot(1,2,1)
#     plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
#     plt.title('fixed image')
#     plt.axis('off')
    
#     # Draw the moving image in the second subplot.
#     plt.subplot(1,2,2)
#     plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
#     plt.title('moving image')
#     plt.axis('off')
    
#     plt.show()

# # Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# # of an image stack of two images that occupy the same physical space. 
# def display_images_with_alpha(image_z, alpha, fixed, moving):
#     img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
#     plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
#     plt.axis('off')
#     plt.show()
    
# # Callback invoked when the StartEvent happens, sets up our new data.
# def start_plot():
#     global metric_values, multires_iterations
    
#     metric_values = []
#     multires_iterations = []

# # Callback invoked when the EndEvent happens, do cleanup of data and figure.
# def end_plot():
#     global metric_values, multires_iterations
    
#     del metric_values
#     del multires_iterations
#     # Close figure, we don't want to get a duplicate of the plot latter on.
#     plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
# def plot_values(registration_method):
#     global metric_values, multires_iterations
    
#     metric_values.append(registration_method.GetMetricValue())                                       
#     # Clear the output area (wait=True, to reduce flickering), and plot current data
#     clear_output(wait=True)
#     # Plot the similarity metric values
#     plt.plot(metric_values, 'r')
#     plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
#     plt.xlabel('Iteration Number',fontsize=12)
#     plt.ylabel('Metric Value',fontsize=12)
#     plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
# def update_multires_iterations():
#     global metric_values, multires_iterations
#     multires_iterations.append(len(metric_values))

def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    #sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

#input_path='/home/davide/SITK_POW_MSE/'
def save_data(list_name,OUT_STAK,res_path):
    for i in range(len(OUT_STAK)):
        
        b=list_name[i].split('/')
        c=b.pop()
        d=c.split('.')
        cv2.imwrite(res_path+'/'+d[0][0:2]+str(int(d[0][2:5])+1)+'.png', OUT_STAK[i])
        
def command_iteration(filter):
    print(f"{filter.GetOptimizerIteration():3} = {filter.GetMetricValue():10.5f}")

def register(filename,fix, mov, metric, optimizer,input_path,moving_image,res_path, name_list, iterations):
    start_single_sw = time.time()
    t=0
    OUT_STAK=[]
    #print(name_list)
    #print("Computing initial transform")
    x = sitk.CenteredTransformInitializer(fix,
                                          mov,
                                          sitk.Euler3DTransform(), 
                                          sitk.CenteredTransformInitializerFilter.MOMENTS)                                           
    #print(x)
    #print("Resampling")
    #resa=sitk.Resample(mov, fix, x, sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID())
    #resa=sitk.Resample(mov, fix, x, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    #res2D=sitk.GetArrayFromImage(resa)
    #print(res2D.shape)
    #mov_res2D=sitk.GetImageFromArray(res2D)
    #print("Starting image registration")
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings.
    if metric=="mse":
        registration_method.SetMetricAsMeanSquares()
    elif metric=="prz":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
    elif metric=="mi":
        registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=256,varianceForJointPDFSmoothing=0)
        #registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=20,varianceForJointPDFSmoothing=0)
    elif metric=="cc":
        registration_method.SetMetricAsCorrelation()
    else:
        return -100
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1)

    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    # Optimizer settings.
    if optimizer=="gd":
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=iterations, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    elif optimizer=="plone":
        registration_method.SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations=iterations, epsilon=1.5e-4, initialRadius=1.01, growthFactor=-1.05, shrinkFactor=-0.99,seed=12345)
    elif optimizer=="pow":
        registration_method.SetOptimizerAsPowell(numberOfIterations=iterations, maximumLineIterations=iterations, stepLength=1, stepTolerance=1e-06, valueTolerance=0.00005)
    else:
        return -100
    #
    #

    registration_method.SetOptimizerScalesFromPhysicalShift()
    #registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    # Setup for the multi-resolution framework.            
    #registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(x, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    #registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    #registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    #registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    #registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    start_time = time.time()
    final_transform = registration_method.Execute(sitk.Cast(fix, sitk.sitkFloat32), 
                                                   sitk.Cast(mov, sitk.sitkFloat32))
    #final_transform = registration_method.Execute(fix, mov)

    end_time= time.time()
    t=t+(end_time - start_time)
    
    FINALE=resample(mov, final_transform)

    #FINALE = sitk.Cast(FINALE, sitk.sitkFloat32)
    #print("type: ", FINALE.dtype) 
    print(FINALE.GetSize())
    naaa=sitk.GetArrayFromImage(FINALE)
    print(naaa.shape)
    OUT_STAK=cv2.normalize(naaa,None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    end_single_sw = time.time()
    print('Final time: ', end_single_sw - start_single_sw)
    with open(filename, 'a') as file2:
        file2.write("%s\n" % (end_single_sw - start_single_sw)) 
    # print(naaa.shape)
    # print(name_list)
    save_data(name_list,OUT_STAK,res_path)
    return t






def compute_wrapper(args, num_threads=1):
    #for k in range(args.offset, args.patient):
    curr_prefix = args.prefix#+str(k)
    curr_ct = os.path.join(curr_prefix,args.ct_path)
    curr_pet = os.path.join(curr_prefix,args.pet_path)
    curr_res = os.path.join(curr_prefix,args.res_path)
    os.makedirs(curr_res,exist_ok=True)
    CT=glob.glob(curr_ct+'/*dcm')
    PET=glob.glob(curr_pet+'/*dcm')
    PET.sort(key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    CT.sort(key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    assert len(CT) == len(PET)
    times=[]
    #images_per_thread = len(CT) // num_threads
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames(args.ct_path)
    dicom_names = np.asarray(dicom_names)
    dicom_names = sorted(dicom_names, key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    
    reader.SetFileNames(dicom_names)
    reader.SetOutputPixelType(sitk.sitkFloat32)

    fixed = reader.Execute()
    fixed1 = sitk.GetArrayFromImage(fixed)
    #save_data(CT,fixed1,"./input/fixed")
    #size = image.GetSize()
    # #dicom_names = reader.GetGDCMSeriesFileNames(ct_path)
    # ct_array = np.array(CT)
    # reader.SetFileNames(ct_array)
    # fixed = reader.Execute()
    # fixed = sitk.GetArrayFromImage(fixed)
    # save_data(CT,fixed,"./input/fixed")
    #size = fixed.GetSize()
    #print(size)
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames(args.pet_path)
    dicom_names = np.asarray(dicom_names)
    dicom_names = sorted(dicom_names, key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    reader.SetFileNames(dicom_names)
    reader.SetOutputPixelType(sitk.sitkFloat32)

    moving = reader.Execute()

    moving1 = sitk.GetArrayFromImage(moving)
    #save_data(CT,moving1,"./input/moving")
    #size = image.GetSize()
    # print("moving", moving.GetPixelIDValue(), '\n')
    # print(moving.GetPixelIDTypeAsString())
    #size = moving.GetSize()
    #print(size)
    #fixed_image = sitk.Cast(fixed, sitk.sitkFloat32)
    #moving_image = sitk.Cast(moving, sitk.sitkFloat32) 

    # f=sitk.GetArrayFromImage(fixed_image)
    # m=sitk.GetArrayFromImage(moving_image)
    
    f=sitk.GetArrayFromImage(fixed)
    m=sitk.GetArrayFromImage(moving)
    #print(m.shape)
    #print(f.shape)
    #print(type(m))
    fix=sitk.GetImageFromArray(f)
    mov=sitk.GetImageFromArray(m)
    #print(fix.GetSize())
    #print(mov.GetSize())
    #print(moving_image.GetSize())
    t=register(args.filename,fix, mov, args.metric, args.optimizer,curr_res,moving,args.res_path, CT, args.iterations)
    
    #times.append(t)
    """
    df = pd.DataFrame([\
        ["total time ",np.sum(times)],\
        ["mean time hw",np.mean(times)],\
        ["std time hw",np.std(times)]],\
                    columns=['Label','Test'+str(args.metric)+"_"+str(args.optimizer)])
    df_path = os.path.join(curr_res,'Time'+str(args.metric)+"_"+str(args.optimizer)+'.csv')
    df.to_csv(df_path, index=False)
    """
    # for i in range(num_threads):
    #     start = images_per_thread * i
    #     end = images_per_thread * (i + 1) if i < num_threads - 1 else len(CT)
    #     name = "t%02d" % (i)
    #     pool.append(multiprocessing.Process(target=compute, args=(CT[start:end], PET[start:end], name, curr_res, i, k)))
    # for t in pool:
    #     t.start()
    # for t in pool:
    #     t.join()


def main():

    parser = argparse.ArgumentParser(description='Iron software for IR onto a python env')
    parser.add_argument("-t", "--thread_number", nargs='?', help='Number of // threads', default=1, type=int)
    parser.add_argument("-mtr", "--metric", nargs='?', help='Metric to be tested, available mse, cc, mi, prz', default='mi')
    parser.add_argument("-opt", "--optimizer", nargs='?', help='optimizer to be tested, available plone, gd, pow', default='plone')
    parser.add_argument("-cp", "--ct_path", nargs='?', help='Path of the CT Images', default='./')
    parser.add_argument("-pp", "--pet_path", nargs='?', help='Path of the PET Images', default='./')
    parser.add_argument("-rp", "--res_path", nargs='?', help='Path of the Results, with the / at the end', default='./')
    parser.add_argument("-f", "--filename", nargs='?', help='Name of the file in which to write times', default='test.csv')
    parser.add_argument("-px", "--prefix", nargs='?', help='prefix Path of patients folder, e.g. ~/faber_at_dac_/Data/Test/St,\
     it will be added a 0 after the prefix', default='./')
    parser.add_argument("-it", "--iterations", nargs='?', help='number of iterations to perform', default=100, type=int)
    #parser.add_argument("-vol", "--volume", nargs='?', help="Volume of the image to analyze, expressed as number of slices", default=1, type=int)
    

    args = parser.parse_args()
    #patient_number=args.patient
    #ct_path=args.ct_path
    #pet_path=args.pet_path
    #res_path=args.res_path
    num_threads = args.thread_number
    #print(args.config)
    compute_wrapper(args, num_threads)

    print("Faber-testing-simpleitk.py is at the end :)")



if __name__== "__main__":
    main()
