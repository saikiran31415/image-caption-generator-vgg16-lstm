# Image Caption Generator using VGG16 and LSTM

A deep learning project that automatically generates descriptive captions for images using a combination of VGG16 (for image feature extraction) and LSTM (for text generation). The model is trained on the Flickr8K dataset and can generate human-like captions for any input image.

## Features

- **Image Feature Extraction**: Uses pre-trained VGG16 model to extract rich visual features from images
- **Text Generation**: Employs LSTM networks for sequential caption generation
- **End-to-End Pipeline**: Complete workflow from data preprocessing to model inference
- **BLEU Score Evaluation**: Implements standard evaluation metrics for caption quality assessment
- **Interactive Inference**: Easy-to-use inference notebook for generating captions on new images

## Architecture

The model follows an **Encoder-Decoder** architecture:

### Encoder (VGG16)
- Pre-trained VGG16 model (excluding the final classification layer)
- Extracts 4096-dimensional feature vectors from images
- Features are passed through a dropout layer and dense layer for dimensionality reduction

### Decoder (LSTM)
- Embedding layer for word representations (256 dimensions)
- LSTM layer for sequential processing (256 units)
- Dense layer with softmax activation for word prediction
- Combines image features with text features using an addition layer

```
Image (224x224x3) → VGG16 → Features (4096) → Dense (256)
                                                    ↓
Text Sequence → Embedding (256) → LSTM (256) → Add → Dense → Softmax → Next Word
```

## Project Structure

```
image-caption-generator-vgg16-lstm/
├── training_and_testing.ipynb    # Main training notebook
├── model_inference.ipynb         # Inference and testing notebook
├── features.pkl                  # Pre-extracted image features
├── all_captions.pkl             # Processed caption data
├── best_model.h5                # Trained model weights
└── model.png                    # Model architecture visualization
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saikiran31415/image-caption-generator-vgg16-lstm.git
   cd image-caption-generator-vgg16-lstm
   ```

2. **Install required dependencies**
   ```bash
   pip install tensorflow keras numpy pandas matplotlib pillow tqdm nltk
   ```

3. **Download the Flickr8K dataset** (if training from scratch)
   - Images: [Flickr8K Dataset](https://www.kaggle.com/adityajn105/flickr8k)
   - Captions: Usually included with the dataset as `captions.txt`

## Usage

### Training the Model

1. **Open the training notebook**
   ```bash
   jupyter notebook training_and_testing.ipynb
   ```

2. **Update dataset paths** in the notebook:
   ```python
   BASE_DIR = '/path/to/your/flickr8k/dataset'
   WORKING_DIR = '/path/to/your/working/directory'
   ```

3. **Run the notebook cells sequentially** to:
   - Extract image features using VGG16
   - Preprocess caption data
   - Train the LSTM model
   - Evaluate model performance

### Generating Captions

1. **Open the inference notebook**
   ```bash
   jupyter notebook model_inference.ipynb
   ```

2. **Load your trained model and generate captions**:
   ```python
   # The notebook includes functions to:
   # - Load pre-trained model
   # - Process new images
   # - Generate captions
   # - Display results
   ```

### Quick Inference Example

```python
# Load required components
model = load_model('best_model.h5')
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)
with open('all_captions.pkl', 'rb') as f:
    all_captions = pickle.load(f)

# Generate caption for an image
caption = predict_caption(model, image_features, tokenizer, max_length)
print(f"Generated Caption: {caption}")
```

## Model Performance

The model is evaluated using BLEU scores:
- **BLEU-1**: Measures unigram precision
- **BLEU-2**: Measures bigram precision

*Note: Specific performance metrics depend on training duration and dataset size*

## Technical Details

### Hyperparameters
- **Vocabulary Size**: Determined from training data (~8000-9000 words)
- **Maximum Caption Length**: ~34 words
- **Embedding Dimension**: 256
- **LSTM Units**: 256
- **Batch Size**: 32
- **Training Epochs**: 50
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

### Data Preprocessing
- Images resized to 224×224 pixels (VGG16 input requirement)
- Captions are lowercased and cleaned
- Special tokens added: `startseq` and `endseq`
- Words shorter than 2 characters are removed

## Dataset

The model is trained on the **Flickr8K dataset** which contains:
- 8,000 images
- 5 captions per image (40,000 total captions)
- Diverse range of subjects and scenes
- Split: 90% training, 10% testing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- tqdm
- NLTK

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **VGG16 Model**: Pre-trained on ImageNet dataset
- **Flickr8K Dataset**: For providing the training data
- **TensorFlow/Keras**: For the deep learning framework

## Contact

**Sai Kiran Chary Parvigari**
- GitHub: [@saikiran31415](https://github.com/saikiran31415)
- Project Link: [https://github.com/saikiran31415/image-caption-generator-vgg16-lstm](https://github.com/saikiran31415/image-caption-generator-vgg16-lstm)
