# Image Caption Generator — Project Report

## Abstract

This project explores automatic image captioning, where a model converts an
image into a short natural-language description. The deep-learning training
implementation uses a ResNet50 convolutional neural network (CNN) to encode
spatial image features, additive visual attention to select image regions, and
an LSTM decoder to predict a caption word by word. The project also provides a
web demo. To meet the implementation deadline, the current demo uses a
pretrained Salesforce BLIP captioning model; it is a separate inference path
from the CNN-attention-LSTM training implementation.

## Objective

The goal is to build and document an image-captioning pipeline that combines
computer vision and sequence modelling. The training code demonstrates
transfer learning, spatial feature extraction, attention, teacher forcing,
sequence padding, cross-entropy optimization, and validation-loss monitoring.

## Dataset and split

Flickr8k contains 8,091 images, each paired with five human-written captions.
The training script reads the image-caption file and splits by unique image so
captions for one image do not appear in both subsets. In the recorded run,
36,405 captions were used for training and 4,050 for validation. Images and
captions are downloaded separately using KaggleHub; they are not stored in this
repository.

## CNN + attention + LSTM implementation

### CNN encoder

An ImageNet-pretrained ResNet50 processes each image. The final convolutional
block produces a 7 × 7 feature map with 2,048 values per spatial location. The
49 locations are retained rather than averaged into one vector, so attention
can use different image regions at different caption steps. The CNN is frozen
while its features are extracted and cached.

### Additive visual attention

At each decoding step, Bahdanau-style additive attention combines each spatial
feature with the current LSTM hidden state. A softmax produces 49 attention
weights, and their weighted sum forms the visual context vector for that step.

### LSTM decoder

The decoder embeds the preceding word and combines that embedding with the
visual context. An LSTMCell updates the hidden and cell states, then predicts a
distribution over the vocabulary for the next word. Training uses teacher
forcing and cross-entropy loss, ignoring padded target tokens. Inference uses
greedy decoding until the end token or the caption-length limit.

The recorded training configuration used 128-dimensional word embeddings, a
256-dimensional LSTM hidden state, a 128-dimensional attention layer, batch
size 64, and the Adam optimizer.

## Training results

The best saved checkpoint is from epoch 5, with validation cross-entropy loss
**2.8446**. Epoch 6 had validation loss **2.8554**, so it did not replace the
epoch-5 checkpoint. The training script saves the checkpoint with the lowest
validation loss to `artifacts/flickr8k_attention.pt`.

These are validation losses, not BLEU scores or test-set results. No held-out
test metrics are claimed. Flickr8k is relatively small, and this short run may
produce imperfect captions.

## Web demo implementation

The Gradio frontend accepts an uploaded image and displays a generated caption.
For the deadline-focused demo, its current backend uses the pretrained
`Salesforce/blip-image-captioning-base` model. The frontend does not display
which model is behind the captioning service. BLIP is not the CNN-attention-LSTM
checkpoint; the report and project code distinguish the demo backend from the
separate training implementation above.

The first app run downloads BLIP weights from Hugging Face. On CPU, model loading
and inference can take longer than on a GPU.

## Limitations and next steps

- The demo's BLIP backend and the trained CNN-attention-LSTM implementation are
  separate model paths.
- The CNN-attention-LSTM model was evaluated using training and validation
  cross-entropy only; a held-out test evaluation and BLEU measurement remain to
  be done.
- Captions may omit objects, actions, or relationships visible in an image.
- Future work could compare both models using the same held-out image split and
  report BLEU alongside qualitative examples.
