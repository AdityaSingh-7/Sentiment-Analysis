🎭 Multimodal Emotion Detection

An advanced deep learning project for detecting emotions from text, audio, and video using a multimodal neural network architecture. This model is designed to improve emotion recognition accuracy by leveraging complementary information from multiple modalities.
📌 Overview

This project detects human emotions using input from:

    📝 Text (e.g., transcripts)

    🔊 Audio (tone, pitch)

    🎥 Video (facial expressions)

Built with the MELD dataset (Multimodal EmotionLines Dataset), the model uses separate encoders for each modality and a joint classifier to predict the emotion.
🧠 Model Architecture

    Text Encoder: BERT / DistilBERT for contextual understanding.

    Audio Encoder: 1D CNN over spectrogram features (e.g., MFCCs).

    Video Encoder: 3D ResNet for spatiotemporal facial expression features.

    Fusion Layer: Concatenates all encoded modalities.

    Classifier: Fully connected layers for final emotion classification.

📊 Dataset

    MELD (Multimodal EmotionLines Dataset)

    7 Emotion Labels: anger, disgust, fear, joy, neutral, sadness, surprise

    Preprocessing steps include:

        Tokenizing text

        Extracting MFCC features for audio

        Extracting frames and faces from video

🔧 Tech Stack

    🐍 Python, PyTorch

    🤗 HuggingFace Transformers (for BERT/DistilBERT)

    🧪 Librosa (for audio processing)

    🎞️ OpenCV, FFmpeg (for video frame extraction)

    📈 TensorBoard (for training logs and visualizations)

    Google Colab (for training/testing)
