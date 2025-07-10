## Demonstration - How to quickly create a classification model

This demonstration will show how to start with an unclassified set of images and turn it into a yoloV 5,8,11 model.
It is expected you are aware of how to use the yolo models from Ultralytics in the first place. This tool is just something to help more quickly annotate and validate your dataset and in the end ultimately train your model.

NOTE: ROOT_FOLDER == The same folder as the Yolo_TrainerTools.py file. 
NOTE: It is assumed you have already gone through installaton before this demo starts.
REMINDER: If you have a GPU, make sure to look at the instalation instructions in the readme.md around how to make sure your GPU is used.

1. Prepare your dataset of images and decide how many classes you are detecting. For this example, we are going to use a publically available pre-existing dataset. It has been downloaded to this repo within the \tutorial_test_set\barcode_set folder.

2. Copy ONLY THE IMAGE FILES to the ROOT_FOLDER\auto_annotation_demo. This dataset already has the text annotations, but for the tutorial we are starting fresh and making our own.

3. Decide on the classifications you are using for this dataset. In this case we are going to be teaching the system how to recognize QR and DataMatrix codes so we will have the following classes, 0:QR and 1:DATAMATRIX.

4. In order to allow the system to auto-annotate to some degree, we'll have to annotate a small sample of the dataset. We are going to annotate 30 examples of each class type. While the system does allow for multiple annotations per image, we are going to specifically pick images that have 1 example of the class we are detecting. This makes it easier to start with a balanced dataset of say 25 annotations of each class.

5. run: python Yolo_TrainerTools.py

6. We first need to make our dataset data.yaml which is used to tell the training system where the train, test, and val folders are. Go to the Dataset Editor tab and click on "Create Dataset" and set the fields as follows:
    - Save Location: Browse to your ROOT_FOLDER, in my case it is "G:\sandbox_ai\yolo_training_tools".
    - Root folder: This will be the working folder that is parented by the ROOT Folder, in my case it is "G:\sandbox_ai\yolo_training_tools\working".
    - The other 3 folders are relative to the working folder and can remain the same.
    - Add the classes we determined in step 3. Add a class 0:QR and class 1:DATAMATRIX
    - Click on Save.

7. We can now start to annotate images. Lets go to the "Folder Browser" tab. 

8. Here we will first load the folder with the barcode images by clicking on the "Browse Folder" button and selecting the "auto_annotation_demo" demo folder. Then click on the "Load Dataset" button and load the data.yaml file you just created in step 6.

9. You are now ready to annotate. Find 25 clear examples of each classification. For the first round, avoid picking images that have multiple classifications on them and stick to images with just one clear example of the classification you are going for. I find it easy to pick 1 classification first and find 25, then do 25 of the other one. Note: If you come across images in the dataset which appear to be duplicates or you do not feel appropriate to be in the dataset, you can either hit the DEL button or click on the "Reject selected images" button which will move the image to the auto_annotation_demo/rejected folder. When completed, click on the "Save all Changes" button.
    - NOTE: In the case of the barcode example, since there isn't as many QR codes in this dataset, we will break the rules a bit and classify a number of multi-categorization examples just to make sure we can capture what we need. Just roughly annotate 25 of each if possible. 

10. We now need to split this dataset into the train and val folders (we are skipping test here since we have a very limited dataset). This will prepare for training. Go to the "Dataset Split" tab. For the "Source Dataset", browse to the "auto_annotation_demo" where your images, and the annotations you just created exist. For the "Output Directory" browse to your ROOT_FOLDER/working (as set originally in the data.yaml). If you do not have the working folder, please create it now.

11. Change the Split Configuration to 80% train, 20% validation and 0% test. Leave the rest of the options as they are and hit "Execute Split"

12. [optional] Verify that your dataset split worked, go to the "Dataset Editor" tab and click on "Load Dataset" and select your data.yaml file you created earlier. If it is setup correctly, you should see some images in your Val folder (which it starts in). To see your training folder, change the "split" dropdown to "Train". This will let you confirm you have drawn the annotations, that the data.yaml is setup correctly. This screen also allows you to edit your annotations further if needed.

13. It's time to do your first training. Click on the "Training" tab and "Load Dataset". Change the training parameters as follows: Batch Size: leave at 16, unless you do not have enough GPU memory or you are on CPU, then lower to say 4. Epoches:50. Click on "Enable Data Augmentation" to open the Augmentation panel. We are going to use augmentation here to help since this is a small dataset. You can use the default augmentation settings.

14. Click on "Start Training" and behold the beautiful graphs while you wait for your first quick training to finish. With a GPU active, 50 epochs here should only take about 2 minutes at most (on a NVIDIA 4060 with 16GB GPU). You will see during the training a number of Charts you can view to see how the training is going in the charts tab. The charts all have a hover over to explain what you are looking at. When the training is complete, the display goes to the "Results" tab. Much of these metrics won't make much sense initially as this is way too small of a dataset, you just need it good enough to help auto-annotate the rest of the dataset.

15. Auto-annotation: In order to get this started, we need to load the last model you just trained. Click on the "Model Management" tab. Then click "Find Models in Project" and the available models list will update. Double-click on the top line with will be a "best.pt" file.

16. Click on the "Auto-Annotation" tab and "Select Folder". This folder is the "auto_annotation_demo" with the images and the annotations. The auto-annotation system will ignore the images with annotations and only work against the images without annotations, with each iteration dumping more annotations in the "auto_annotation_demo" folder, splitting, training and manual validation of the results. Wash, rinse, repeat.

17. Click on the "Load Dataset" button and load your data.yaml.

18. Click on "Start Auto-Annotation". You can leave all the auto-annotation training and augmentation settings (unles you have a lower GPU memory or CPU, then lower batch size).

19. This will cause the system to run inference against your non-annoated images. You may need to adjust the confidence thresholds on your first iteration to something like High:0.80 and Medium:0.25, otherwise you may not see many images. You will start out on the "Review" filter. There may be a smaller amount of images to select from at first. I typically go to the Approved folder to see images that higher confidence levels and check all of them to make sure they look good. If any of them look off, either correct them by selecting the image and modifying the annotation, or just hit "R" with the image selected to send it to the Rejected filter to be annotated in the next iteraton. Once I'm happy with the Approved filtered, I take a look at the "Review" filter and see if there are any easy wins, then just draw a mouse selector around the ones that look good and hit the "A" key to approve the selected group.

20. When you are happy with your first group of images that you have in your "Approved" filter, click on the Approved button, then "Select All" (or CTRL-A). We are now going to "Export Selected Annotations" to the file system, which will also trigger another iteration of the auto-annotation system.

