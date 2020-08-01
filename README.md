# Custom Object Detection using Tensorflow On Jetson Nano

This tutorial is inspired by following tutorials:
[Step by Step: Build Your Custom Real-Time Object Detector](https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d)
[How to run TensorFlow Object Detection model on Jetson Nano](https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/)

The following steps are for making a custom dataset, if anyone wants to save image from a camera, the following repository will be useful:
[Dataset Builder](https://github.com/MahimanaGIT/DatasetBuilder)

Annotate the images of the dataset using [LabelImg](https://github.com/tzutalin/labelImg)


The repository should be of the following for:
> Object Detection Repostiory:
>   - data/
>        - images
>            - ".jpg" files
>       - annotations
>            - ".xml" files from LabelImg
>        - test_labels
>            - Separate the  test labels
>        - train_labels
>            - Separate the train labels
>   - models
>        - pretrained_model
>        - fine_tune_model
>        - training
>        - trt_model
    

Step 1: Saving custom images for the dataset

Step 2: Saving corresponding xml files for every images of the dataset using LabelImg

Step 3: Saving Images in data/images folder and separate corresponding xml files in "test_labels" and "training_labels" folder in ./data folder

Step 4: Converting corresponding xml files for "test_labels" and "training_labels" to CSV files and label_map.pbtxt file using the script "Preprocessing.py"

Step 5: Exporting the python path to add the object detection from tensorflow object detection API: 

    "export PYTHONPATH=$PYTHONPATH:~/repo/object_detection/object_detection/models/research/:~/repo/object_detection/object_detection/models/research/slim/"

Step 6: Run the following command in terminal from the directory "object_detection/models/research/":

    "protoc object_detection/protos/*.proto --python_out=."

Step 7: Run the script to check if everything is OK

    "python3 object_detection/builders/model_builder_test.py"

Step 8: Generate TF Records using "GeneratingTFRecords.py"

Step 9: Select and download the model using the "SelectingAndDownloadingModel.py"

Step 10: Configure Model Training Pipeline using "ConfigureModelTrainingPipeline.py"

Step 11: To launch tensorboard, execute the following command:

    "tensorboard --logdir=./models/training"

Step 12: Configuring the "pipeline_1.config" to copy from the existing sample of config file "/object_detection/models/research/object_detection/samples/config/ssd_mobilenet_v2_coco.config"

Step 13: Start the training running the script and giving the arguments:
    "python model_main.py --pipeline_config_path=./models/pretrained_model/pipeline_1.config --model_dir=./models/training"

Step 14: Change the classes and all the class id number in the corresponding files.

Step 15: Export the best model using the script "ExportTrainedModel.py".

Step 16: Use the frozen path from "models/fine_tuned_model" with the file name "frozen_inference_graph.pb" and label_map.pbtxt and run the UseModel.py

Change classes when training new model in:

1. pipeline.config
2. GeneratingTFRecords.py

Debugging:

Using TFRecord viewer for verifying the .record file, execute:

    "python3 tfviewer.py /home/mahimana/Documents/Deep_Learning/ObjectDetection/data/train_labels.record --labels-to-highlight='cubesat, rock, processing_plant'"


#Problems Faced: 
1. Error Generating TF Records, each bounding box was covering the whole image, "tf.io.TFRecordWriter" was causing problem, using "tf.python_io.TFRecordWriter" instead.
