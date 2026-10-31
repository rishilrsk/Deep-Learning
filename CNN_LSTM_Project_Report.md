# Image Caption Generation Using a CNN Encoder, Visual Attention, and LSTM Decoder

## Deep Learning Course Project Report

**Student name(s):** [Add name(s)]  
**Student ID(s):** [Add ID(s)]  
**Department / Institution:** [Add department and institution]  
**Academic year:** [Add academic year]

---

## Abstract

Image caption generation is the task of producing a short natural-language description from an image. This project implements an encoder–decoder deep-learning system that combines a convolutional neural network (CNN), additive visual attention, and a long short-term memory (LSTM) decoder. The CNN encoder is a pretrained ResNet50 whose final classification layer is removed. It converts each image into a grid of spatial visual features. At every word-generation step, an additive attention module assigns weights to those spatial features and builds a context vector. An LSTMCell uses this context and the preceding word to predict the next word in the caption.

The model is trained and evaluated using the Flickr8k image-caption dataset. Captions are normalized and tokenized, special tokens mark the start and end of each sequence, and images are split into training and validation sets so captions for a given image remain in the same split. The CNN features are cached to reduce repeated image computation during decoder training. The implemented decoder uses a 128-dimensional word embedding, 256-dimensional hidden state, and 128-dimensional attention layer. Training uses Adam and token-level cross-entropy loss. The best saved checkpoint was produced at epoch 5 with a validation loss of approximately 2.8446. The project demonstrates how spatial visual features, attention, and sequence modeling can be combined for image captioning. The recorded loss is a training diagnostic; no separate held-out test metric or human evaluation was recorded in this run.

**Keywords:** Image captioning, deep learning, ResNet50, CNN encoder, visual attention, LSTM, Flickr8k, sequence generation.

## 1. Introduction

Images contain visual information about objects, people, actions, attributes, and relationships. A computer-vision classifier typically assigns one or more labels to an image, while an image-captioning system aims to express relevant visual content as a sentence. Caption generation therefore combines visual feature extraction with language modeling.

This project uses an encoder–decoder design. The encoder processes an image and creates features representing different spatial regions. The decoder generates a caption one word at a time. Visual attention connects these two stages: before predicting each word, it estimates which encoded image regions are most useful for that step. The model can therefore condition different parts of a sentence on different portions of the image.

The CNN encoder is ResNet50, a deep residual convolutional network. Its pretrained visual representations provide a practical starting point when the project dataset is modest in size. The decoder is an LSTM, a recurrent neural network designed to maintain information across a sequence. The system is trained with image-caption pairs from Flickr8k.

### 1.1 Motivation

Automatically describing images is useful for accessibility, image search, photo organization, and human-computer interaction. It is also a useful deep-learning project because it combines convolutional networks, recurrent networks, attention, sequence preprocessing, and supervised learning in one end-to-end task.

### 1.2 Project objectives

1. Build an image-captioning pipeline using a CNN encoder and an LSTM decoder.
2. Preserve spatial feature locations from the CNN so the decoder can attend to image regions.
3. Implement additive attention that computes a visual context vector for each generated word.
4. Prepare Flickr8k images and captions for supervised training.
5. Train the decoder, monitor validation cross-entropy loss, and save the best checkpoint.
6. Generate a caption from a new image by predicting words until an end token is produced or a maximum length is reached.

## 2. Problem statement

Given an image, the system generates a fluent caption describing its main content. It extracts visual features from the image and uses them, together with the words already generated, to predict the next word in the caption.

The main challenges are:

- **Visual representation:** Relevant objects and actions may appear at different image locations and scales.
- **Visual-language alignment:** The model must connect words to image evidence rather than generate a generic sentence.
- **Sequence dependencies:** A caption’s next word depends on its preceding words and the visual context.
- **Limited training data:** Flickr8k is much smaller than modern web-scale image-text datasets, so the model may not describe unfamiliar scenes reliably.
- **Evaluation:** A low token-level loss does not by itself guarantee that a generated sentence is factually correct, fluent, or preferred by a human.

## 3. Literature review

### 3.1 Convolutional image encoders

Convolutional neural networks learn layered visual representations from image pixels. Early layers respond to simple patterns such as edges and textures, while deeper layers capture more complex visual structures. ResNet introduced residual connections that help train substantially deeper networks by providing shortcut paths through the layers [1]. In this project, a pretrained ResNet50 supplies the image feature extractor.

### 3.2 Recurrent language generation

An LSTM maintains a hidden state and cell state as it processes a sequence. Its gated updates help preserve useful information over multiple time steps [2]. In caption generation, the LSTM receives a word embedding and image context at each time step and predicts a distribution over the next word. Earlier CNN–RNN captioning systems established the encoder–decoder pattern for generating a sentence from an image [3].

### 3.3 Visual attention for captioning

Using one fixed image vector for an entire sentence can make it difficult to connect individual words with specific visual regions. “Show, Attend and Tell” introduced an attention-based captioning approach that learns to focus on different image regions while generating a caption [4]. This project follows that general idea with additive attention: the current decoder hidden state is compared with each projected spatial feature, and a softmax converts the scores into region weights.

### 3.4 Dataset background

Flickr8k contains photographs paired with multiple human-written descriptions. The dataset was introduced as a resource for image description and caption retrieval research [5]. Multiple captions per image provide alternative language descriptions of the same visual content, which is useful for supervised caption-generation training.

## 4. Existing system and limitations

Basic image-description systems often use image classification: they identify likely objects and return labels such as “dog,” “street,” or “car.” Such a list does not express relationships or actions in a sentence. A simple CNN-to-single-vector and LSTM design can generate sentences, but the decoder may have difficulty using a single compressed visual representation for every word.

Common limitations include:

- Captions can omit secondary objects or relationships.
- The model can produce generic wording that is plausible for many images.
- A generated object or action may be incorrect even when the sentence is grammatically valid.
- Training and inference quality depend on vocabulary coverage and dataset composition.
- Greedy decoding chooses the highest-probability next token at each step and may miss a better complete sentence.
- The current experiment records cross-entropy loss but does not report BLEU, METEOR, CIDEr, SPICE, or human-evaluation results.

## 5. Proposed system

The proposed system generates a caption using three cooperating components:

1. **CNN encoder:** A pretrained ResNet50 processes a resized and normalized RGB image. The final classification layers are removed so the network returns a spatial feature map rather than class scores.
2. **Additive attention:** The feature map is reshaped into a set of spatial region vectors. At each decoding step, attention compares every region with the decoder’s current hidden state, normalizes the scores, and combines the regions into a context vector.
3. **LSTM decoder:** The previous caption token is embedded and concatenated with the attention context. An LSTMCell updates its hidden and cell states. A linear output layer maps the hidden state and context to vocabulary scores for the next token.

### 5.1 Advantages of the proposed system

- Retains spatial information instead of representing the entire image with only one feature vector.
- Recomputes attention at every word step, allowing the visual context to change through the caption.
- Uses a pretrained CNN to provide useful visual features without training the full CNN from scratch.
- Uses captions paired with images for supervised, end-to-end decoder learning.
- Saves the checkpoint with the best validation loss observed during training.

## 6. System requirements

### 6.1 Software

- Python 3.10 or newer
- PyTorch and Torchvision
- Pillow for image loading and image transformations
- KaggleHub for downloading Flickr8k when using the supplied download script
- A command-line terminal for dataset preparation and model training

The exact dependencies are listed in the project’s `requirements.txt`.

### 6.2 Hardware

- CPU: supported; training is slower because the model processes many caption examples.
- GPU: CUDA-compatible GPU recommended to reduce feature-extraction and training time.
- Storage: approximately 1 GB for the dataset, plus space for cached CNN features, model weights, and Python dependencies.

## 7. Dataset description

The project uses the Flickr8k dataset downloaded through KaggleHub from the `adityajn105/flickr8k` dataset. It contains approximately 8,000 images and five captions per image, represented in this project as 8,091 image files and 40,455 image-caption pairs. The image directory and captions file are supplied to `train.py`.

The implementation sorts the unique image names and assigns 90% of the image names to training and the remaining 10% to validation. All captions for an image remain together in one split. In the completed run, the training set contained 36,405 captions and the validation set contained 4,050 captions. This is an image-level train-validation split; the run did not create a separate test set.

### 7.1 Caption representation

Captions are converted to lowercase and tokenized into alphanumeric word tokens. The vocabulary is built from training captions, retaining words that occur at least twice. Special tokens represent padding, unknown words, caption start, and caption end. Each sequence is represented as:

`<start> word_1 word_2 ... word_n <end>`

Shorter sequences in a batch are padded to the batch’s maximum caption length. Padding targets are ignored by the cross-entropy loss.

## 8. Methodology

### 8.1 Image preprocessing and feature extraction

Each source image is orientation-corrected using its EXIF metadata, converted to RGB, and transformed using the preprocessing associated with the pretrained ResNet50 weights. ResNet50’s final average-pooling and classification layers are removed. Its last convolutional block returns a feature map with 2,048 channels and a 7 × 7 spatial grid for a standard 224 × 224 input.

The grid is flattened into 49 region vectors, each of dimension 2,048. These vectors are saved in a feature cache as half-precision tensors. During decoder training, cached features are converted to floating point and batched with caption token sequences. Caching avoids running the frozen CNN repeatedly for every caption associated with an image.

### 8.2 CNN encoder

The encoder is ResNet50 initialized with pretrained ImageNet weights. It is used in evaluation mode to extract image features, and the training loop optimizes the caption decoder rather than fine-tuning the ResNet50. The spatial output is represented as:

\[
V = \{v_1, v_2, \ldots, v_{49}\}, \qquad v_i \in \mathbb{R}^{2048}
\]

The decoder projects each region vector to a 256-dimensional representation. The mean of the projected regions is used to initialize the LSTM hidden and cell states.

### 8.3 Additive visual attention

At decoding step (t), the current hidden state (h_{t-1}) is combined with each projected image region. The implementation computes an additive score of the form:

\[
e_{t,i} = w^T \tanh(W_v v_i + W_h h_{t-1})
\]

The score is normalized over all 49 spatial regions:

\[
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{49}\exp(e_{t,j})}
\]

The visual context is the weighted sum of region features:

\[
c_t = \sum_{i=1}^{49} \alpha_{t,i}v_i
\]

Here, (\alpha_{t,i}) is the attention weight assigned to region (i) at step (t). The same attention mechanism is used during training and caption generation. The attention weights are internal model values; the current application does not provide a user-facing visualization of them.

### 8.4 LSTM decoder and next-word prediction

The previous token (y_{t-1}) is mapped to a 128-dimensional word embedding. The embedding is concatenated with the 256-dimensional attention context and passed to an LSTMCell with a 256-dimensional hidden state. The output layer combines the updated hidden state and visual context, then produces scores over the vocabulary:

\[
P(y_t \mid y_{<t}, I) = \operatorname{softmax}(W_o[ h_t; c_t ] + b_o)
\]

During training, the decoder receives the ground-truth previous token at each step (teacher forcing) and learns to predict the next token. During inference, it starts with `<start>`, selects the highest-scoring token at each step (greedy decoding), and stops at `<end>` or after 32 generated tokens.

### 8.5 Training objective and optimization

The model minimizes token-level categorical cross-entropy between the predicted vocabulary distribution and the next ground-truth caption token. Padding positions are excluded from the loss. The implementation uses Adam with a learning rate of 0.001, batch size 64, and gradient-norm clipping at 5.0. Decoder dropout is 0.3. The best checkpoint is saved when validation loss improves.

## 9. Flow chart

The processing pipeline is:

1. Read an image-caption pair from Flickr8k.
2. Preprocess the image and tokenize its caption.
3. Extract and cache the image’s 7 × 7 × 2,048 ResNet50 feature map.
4. Convert the feature map into 49 region vectors and project them to the decoder dimension.
5. Initialize the LSTM hidden state and cell state from the pooled visual features.
6. For each caption token, calculate additive attention weights and a context vector.
7. Combine the previous word embedding with the context vector and update the LSTM.
8. Predict the next word and calculate cross-entropy loss during training.
9. Validate after each epoch and save the checkpoint with the lowest validation loss.
10. For a new image, repeat the encoder and attention-decoder steps, greedily generating tokens until `<end>` or the maximum length.

### Napkin AI prompt

> Create a clear academic flowchart for an image-caption generation project using a ResNet50 CNN encoder, additive visual attention, and an LSTM decoder. Use a white background, readable labels, tidy spacing, and a restrained navy, teal, and light-gray palette. Show two connected lanes labeled TRAINING and INFERENCE.
>
> In the TRAINING lane, show: Flickr8k image-caption pairs → image preprocessing (EXIF orientation, RGB, ResNet50 resize and normalization) and caption preprocessing (lowercase tokenization, vocabulary, `<start>`, `<end>`, padding) → pretrained ResNet50 CNN encoder with classification head removed → 7 × 7 × 2,048 spatial feature map → flatten to 49 region vectors → cache features → project features and initialize LSTM hidden and cell states → teacher-forced previous word embedding + current hidden state → additive attention scores over 49 regions → softmax attention weights → weighted visual context vector → LSTMCell → vocabulary softmax for next-word prediction → cross-entropy loss with padding ignored → Adam optimizer updates decoder → validation loss → save best checkpoint.
>
> In the INFERENCE lane, show: user image → same image preprocessing → ResNet50 spatial feature extraction → projected 49 region vectors → initialize LSTM states → start token → additive attention over regions → context vector + previous generated word embedding → LSTMCell → greedy next-token prediction → loop back for the next token until `<end>` or the 32-token limit → generated caption. Connect the training checkpoint to the inference decoder. Label the CNN as frozen pretrained ResNet50 and the decoder dimensions as embedding 128, hidden 256, attention 128. Use arrows to make the sequence and loop explicit. Do not add other architectures, third-party model names, product logos, or unsupported evaluation metrics.

## 10. Implementation details

The model components are implemented in `caption_model.py`, with dataset preparation, feature extraction, training, validation, and checkpoint saving handled in `train.py`. The caption model consists of an additive attention module and an `AttentionLSTMCaptioner`. The training script accepts the image directory and caption file as command-line arguments. It supports resuming from a checkpoint when the saved vocabulary matches the current training data.

The training configuration used for the saved model was:

| Setting | Value |
|---|---:|
| Dataset | Flickr8k |
| Split | 90% train / 10% validation by image |
| Training captions | 36,405 |
| Validation captions | 4,050 |
| CNN encoder | Pretrained ResNet50, frozen for decoder training |
| Spatial feature grid | 7 × 7 = 49 regions |
| Region feature dimension | 2,048 |
| Word embedding dimension | 128 |
| LSTM hidden dimension | 256 |
| Attention dimension | 128 |
| Batch size | 64 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Cross-entropy, padding ignored |
| Decoder dropout | 0.3 |
| Gradient clipping | Maximum norm 5.0 |
| Inference decoding | Greedy, maximum 32 tokens |

## 11. Results and discussion

The recorded run continued training from an existing checkpoint and completed epochs 4–6. The best checkpoint was saved at epoch 5 because it had the lowest validation cross-entropy loss observed in that run.

| Epoch | Training loss | Validation loss | Checkpoint status |
|---:|---:|---:|---|
| 4 | 2.5380 | 2.8614 | Not best |
| 5 | 2.3083 | 2.8446 | Best checkpoint saved |
| 6 | 2.1288 | 2.8554 | Validation loss increased |

Training loss decreased from epoch 4 through epoch 6, while validation loss improved through epoch 5 and then rose slightly. This pattern suggests that further training did not improve the validation objective during the final epoch. The checkpoint therefore retains the epoch-5 parameters rather than the epoch-6 parameters.

These values are cross-entropy losses, not caption-quality scores. The available run did not compute a separate test-set score, BLEU, METEOR, CIDEr, SPICE, or a human rating. Claims about caption accuracy should therefore be based on qualitative examples and, for a formal evaluation, additional metrics computed on a held-out test set.

## 12. Applications, advantages, and limitations

### 12.1 Applications

- Supporting image descriptions for accessibility workflows.
- Adding searchable text to photo collections.
- Producing draft descriptions for image-management systems.
- Demonstrating CNN feature extraction, attention, and recurrent sequence generation in a deep-learning course project.

### 12.2 Advantages

- The spatial CNN representation gives the attention module multiple image locations to consider.
- The decoder recalculates attention as the caption grows.
- The LSTM maintains a recurrent state that carries information from preceding words.
- A cached, frozen encoder reduces repeated computation during decoder training.
- Validation loss is monitored, and the best observed checkpoint is retained.

### 12.3 Limitations

- Flickr8k is small and may not cover all image domains, objects, or writing styles.
- The vocabulary and captions constrain which words and descriptions the model can produce.
- Greedy decoding is locally optimal at each step and may yield repetitive or incomplete captions.
- A pretrained ImageNet encoder may not represent every caption-relevant relationship or activity.
- The attention weights are not a guarantee of faithful explanations and are not currently shown in an interface.
- The current documented experiment lacks held-out test metrics and systematic human evaluation.
- CPU training can take substantial time; feature caching reduces repeated CNN work but requires disk space.

## 13. Future enhancements

1. Reserve an image-level test set and report caption metrics such as BLEU, METEOR, CIDEr, and SPICE.
2. Include qualitative examples with the source image, reference captions, generated caption, and discussion of errors.
3. Compare greedy decoding with beam search and evaluate changes on the same held-out set.
4. Experiment with data augmentation and regularization to improve generalization.
5. Fine-tune selected CNN layers after the decoder has stabilized, if compute resources allow.
6. Visualize the learned attention weights as a research aid, while avoiding claims that they fully explain the generated sentence.
7. Evaluate performance on images outside Flickr8k to understand domain-shift limitations.

## 14. Conclusion

This project implements an image-captioning model that combines a pretrained ResNet50 CNN encoder, additive visual attention, and an LSTM decoder. The CNN represents an image as 49 spatial feature vectors. The attention layer forms a step-specific visual context, and the LSTM uses that context along with the preceding token to predict the next word. The system was trained on Flickr8k image-caption pairs with token-level cross-entropy loss. The best checkpoint from the recorded run was saved at epoch 5 with a validation loss of approximately 2.8446.

The project demonstrates the central deep-learning components of an image-captioning pipeline and provides a foundation for further evaluation. A held-out test set, standard caption metrics, and careful human review would be needed to make stronger claims about generated-caption quality.

## References

1. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
2. S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. [DOI](https://doi.org/10.1162/neco.1997.9.8.1735).
3. O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, “Show and Tell: A Neural Image Caption Generator,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015. [arXiv](https://arxiv.org/abs/1411.4555).
4. K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov, R. Zemel, and Y. Bengio, “Show, Attend and Tell: Neural Image Caption Generation with Visual Attention,” *Proceedings of the 32nd International Conference on Machine Learning*, PMLR 37, pp. 2048–2057, 2015. [PMLR](https://proceedings.mlr.press/v37/xuc15.html).
5. M. Hodosh, P. Young, and J. Hockenmaier, “Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics,” *Journal of Artificial Intelligence Research*, vol. 47, pp. 853–899, 2013. [JAIR article and Flickr8k information](https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html).

---

**End of report**
