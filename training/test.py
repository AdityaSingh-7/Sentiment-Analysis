from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import torch 
import numpy as np
import subprocess
import librosa
import librosa.display


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6,
        }

        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise FileNotFoundError(f"File {video_path} not found")
            
            ret, frame= cap.read()
            if not ret or frame is None:    
                raise ValueError(f"Cannot read frame from {video_path}")
            
            #reset the video capture to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame/255.0
                frames.append(frame)
            
        except Exception as e:
            raise ValueError(f"Error reading video {video_path}: {e}")
        finally:    
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        #pad or truncate frames
        if len(frames) < 30:
            padding = [np.zeros_like(frames[0]) for _ in range(30 - len(frames))]
            frames.extend(padding)
        else:    
            frames = frames[:30]

        #before permute the shape is (30, 224, 224, 3){30 frames, 224x224 resolution, 3 channels}
        #after permute the shape is (30, 3, 224, 224){30 frames, 3 channels, 224x224 resolution}
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)
    
    def extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')
        try:
            # Convert video to audio using ffmpeg
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_fft=1024, 
                hop_length=512, 
                n_mels=64
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Normalize the spectrogram
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
            
            # Convert to tensor
            mel_tensor = torch.FloatTensor(log_mel_spec)
            
            # Handle padding/truncation to fixed size (300 frames)
            if mel_tensor.size(1) < 300:
                padding = 300 - mel_tensor.size(1)
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, padding))
            else:
                mel_tensor = mel_tensor[:, :300]
            
            # Add batch dimension to match torchaudio output
            mel_tensor = mel_tensor.unsqueeze(0)
            
            return mel_tensor
        
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error extracting audio from video: {e}")
        except Exception as e:
            raise ValueError(f"Error converting video to audio: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]

        try:
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            video_path_exists = os.path.exists(video_path)

            if video_path_exists == False:
                raise FileNotFoundError(f"File {video_path} not found")
            
            text_inputs = self.tokenizer(row['Utterance'],
                                        padding="max_length",
                                        truncation=True,
                                        max_length=128, 
                                        return_tensors='pt')
            
            video_frames = self.load_video_frames(video_path)
            audio_features = self.extract_audio_features(video_path)
            print(audio_features)

            #map sentiment and emotion labels
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs':{
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing row {video_path}:{e}")
            return None
        
def collate_fn(batch):
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_dataloader(train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloader(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )
    
    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'].shape)
        print(batch['sentiment_label'].shape)
        break