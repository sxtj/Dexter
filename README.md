# Dexter

Dexter is a machine learning project that recognizes static American Sign Language
(ASL) alphabet gestures from images and a live webcam feed.

I built this project in high school to learn more about computer vision, neural
networks, and how trained models can be used in a real-time application. The model
is trained with TensorFlow/Keras and the webcam demo uses OpenCV to capture hand
gestures and display predictions on screen.

## What it does

- Trains a convolutional neural network to classify ASL alphabet images
- Supports 29 classes: A-Z, `space`, `del`, and `nothing`
- Uses image augmentation to make the model more robust to small changes in hand
  position, lighting, and scale
- Applies class weighting to help handle uneven class distributions
- Saves the best model during training
- Runs live webcam inference with a region of interest and confidence score

## Tech stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

## Project files

- `model.py` - trains the ASL image classifier and saves the model
- `cam.py` - loads the trained model and runs real-time webcam prediction
- `README.md` - project overview and setup notes

## Model approach

The model is a custom convolutional neural network built with Keras. It uses
residual blocks, batch normalization, dropout, global average pooling, and dense
classification layers.

During training, the image pipeline uses:

- Rescaling
- Rotation
- Width and height shifts
- Shearing
- Zoom
- Brightness variation
- Validation split
- Early stopping
- Learning rate reduction
- Model checkpointing

Horizontal flipping is disabled because mirroring a hand sign can change its
meaning.

## Dataset format

The training script expects the dataset to be organized like this:

```text
a_d/
  train/
    A/
    B/
    C/
    ...
    space/
    del/
    nothing/
  test/
    A/
    B/
    C/
    ...
    space/
    del/
    nothing/
```

Each class should have its own folder containing images for that label.

## How to run

Install the required Python packages:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```

Train the model:

```bash
python model.py
```

This saves:

- `best_asl_model.h5` - best checkpoint based on validation accuracy
- `final.h5` - final model after training

Run the webcam demo:

```bash
python cam.py
```

Place your hand inside the box shown on the webcam feed. Press `q` to quit.

## Results

The training script prints the final test accuracy and loss after evaluation.

If using this project for a resume or portfolio, add the final accuracy here:

```text
Test accuracy: TODO
Test loss: TODO
```

## Limitations

This project recognizes static ASL alphabet gestures, not full ASL sentences or
grammar. Performance can also depend on lighting, camera quality, background, hand
position, and whether the webcam input looks similar to the training data.

## What I learned

This project helped me practice:

- Building and training CNNs with TensorFlow/Keras
- Designing a computer vision preprocessing pipeline
- Using callbacks to improve training
- Evaluating model performance
- Connecting a trained model to a real-time webcam application
- Thinking about the difference between a model that works on a dataset and one
  that works reliably in the real world
