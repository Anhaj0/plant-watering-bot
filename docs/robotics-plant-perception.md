# Plant perception for robotics: from a camera frame to an edge-AI decision

**By Anhaj Uwaisulkarni · Colombo, Sri Lanka**

A plant-care robot needs to turn camera observations into information that a control system can use. My public plant-perception project explores the first part of that problem: classifying image regions, exporting a model and connecting inference to a camera loop.

I'm an AI Computer Vision Engineer at EVOQ and a Computer Science with Applied AI student at Coventry University. Robotics, edge AI and IoT are the direction I want to develop professionally. This article explains the implementation available in my repository and the checks needed before using that perception component in a physical system.

The two source files are [Train.py](https://github.com/Anhaj0/plant-watering-bot/blob/master/Train.py) and [webcam.py](https://github.com/Anhaj0/plant-watering-bot/blob/master/webcam.py). This is a source walkthrough; it does not report a new training run or hardware benchmark.

## Give the camera a limited, useful question

The camera runner divides each frame into four vertical regions. Each region is resized to the model's input dimensions and classified independently. The display draws a region border and a plant score for each result.

This is regional image classification. A positive result says that a strip looks like the plant class; it does not provide an object's bounding box, distance or precise position. That distinction matters when deciding what a robot could safely infer from the output.

For an early perception component, dividing the scene into regions gives the interface an understandable structure. A control system would still need to establish how those regions relate to camera geometry, robot position and a target plant.

## The training pipeline

The current implementation uses **MobileNetV2**, initialized with ImageNet weights, as its feature extractor. Global average pooling, dropout and a single sigmoid output form the classification head. Images are resized to 224 by 224 pixels, with separate plant and not-plant directories.

The script reserves 30 percent of the images for validation. Its training sequence contains RGB phases followed by grayscale phases, with transformations including rotation, brightness changes and noise. The grayscale conversion writes back to the dataset files, so a separate copy of original images would be needed to preserve the original collection.

The code also contains a fine-tuning stage. Its presence alone is not proof that every intended layer was trained; the model's trainable variables would need inspection during an actual run.

These source details explain the training design. They do not establish accuracy under a moving robot's camera, unusual lighting or plants absent from the dataset. That needs an evaluation set with examples that were kept out of training.

## Exporting the model for edge AI

The converter requests integer-only TensorFlow Lite operations, with uint8 inputs and outputs. It supplies up to 100 training examples through a representative-data generator.

Representative data is used to estimate numerical ranges during quantization. It should reflect the input distribution the deployed model will receive. [Google's API documentation](https://ai.google.dev/edge/api/tflite/python/tf/lite/RepresentativeDataset) explains its calibration role.

For this project, conversion is a step toward deployment. Compatibility, memory consumption and latency on a chosen embedded device remain separate questions. A desktop camera runner does not demonstrate that the model executes on a microcontroller.

## Keep capture and inference understandable

The OpenCV loop requests a 640 by 480 camera image and passes frames to a background worker. Both frame and result queues have a capacity of two. When the frame queue is full, the capture loop drops the new frame instead of waiting to submit it.

That design bounds the number of queued frames. It does not guarantee that the displayed result belongs to the newest captured image. The result queue uses a blocking write, so the worker can also wait if results are not being consumed. Both behaviors matter when reasoning about observation age.

The runner displays a frame-rate estimate. The rate at which the display loop runs is not necessarily the rate at which new predictions complete. A useful benchmark would separately measure capture, preprocessing, inference and the age of the prediction displayed to the user.

## Three checks before connecting a physical action

**Input consistency.** Training rescales image values to 0–1. The uint8 camera path applies the quantization scale to raw camera values. Those paths need a numerical comparison using the same image before the predictions can be trusted. Color order and grayscale handling should be checked at the same time.

**Class meaning.** The runner flips the dequantized sigmoid output and applies a 0.80 threshold. The label mapping from the training dataset, the output interpretation and examples from both classes should agree. A threshold needs validation against false positives and false negatives for the intended use.

**Timing and behavior.** A watering action needs more than a positive strip classification. The system needs a valid target, suitable sensor observations, limits on actuation and a way to handle uncertain or stale camera results. These belong to the next stage of integration, rather than being demonstrated by the current camera code.

## Where this connects to IoT

The current public source covers the vision component. A future connected plant-care system could record observations and make them available through an interface: image region, prediction, timestamp and system state. Telemetry would help a person understand why a device acted or why it waited.

That is the connection I want to develop between computer vision, robotics and IoT: observable components whose inputs, outputs and limitations can be inspected. Network telemetry, moisture sensing and watering control are future integration work for this project, not implemented features established by these two files.

If you are working on robotics perception or edge-AI products, the [project repository](https://github.com/Anhaj0/plant-watering-bot) is available to inspect. My broader background is on my [portfolio](https://anhaj0.github.io/), and you can find me on [LinkedIn](https://www.linkedin.com/in/anhaj-uwaisulkarni).

**About the author:** Anhaj Uwaisulkarni is an AI Computer Vision Engineer at EVOQ, a Coventry University Applied AI student and a co-founder of AstriX Digital Systems, based in Colombo, Sri Lanka. His career interests include robotics perception, edge AI and connected devices.


