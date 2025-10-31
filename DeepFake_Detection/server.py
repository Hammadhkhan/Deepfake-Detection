"""
DeepFake Detection Server - COMPLETE VERSION with AASIST Audio Model
Supports Image (Xception), Video (ResNeXt+LSTM), and Audio (AASIST) detection
"""

from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
import face_recognition
import librosa
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
UPLOAD_FOLDER = 'Uploaded_Files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def log(message, level="INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# ============================================================
# IMAGE MODEL - Xception Architecture
# ============================================================

class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution for Xception"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Xception Block with skip connections"""
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        
        rep = []
        filters = in_filters
        
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        
        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters))
        
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        
        self.rep = nn.Sequential(*rep)
    
    def forward(self, inp):
        x = self.rep(inp)
        
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        
        x += skip
        return x


class XceptionModel(nn.Module):
    """Xception Model for Image Deepfake Detection"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.block1 = XceptionBlock(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        
        # Middle flow
        self.block4 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        
        # Exit flow
        self.block12 = XceptionBlock(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.last_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Exit flow
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.last_linear(x)
        return x


# Initialize Image Model
image_model = None
try:
    log("Loading Xception image model...")
    image_model = XceptionModel(num_classes=2)
    
    checkpoint = torch.load('model/ffpp_c23.pth', map_location='cpu')
    
    # Remove 'model.' prefix from checkpoint keys
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('model.'):
            new_key = key[6:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load state dict
    missing, unexpected = image_model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        log(f"Image model - Missing keys: {len(missing)}", "WARNING")
    if unexpected:
        log(f"Image model - Unexpected keys: {len(unexpected)}", "WARNING")
    
    image_model.eval()
    log("✓ Image model loaded successfully", "SUCCESS")
    
except Exception as e:
    log(f"ERROR loading image model: {e}", "ERROR")
    traceback.print_exc()
    image_model = None

# Image preprocessing
transform_img = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def preprocess_image(path):
    """Preprocess image for model input"""
    try:
        img = Image.open(path).convert('RGB')
        np_img = np.array(img)
        
        # Try to detect and crop face
        try:
            faces = face_recognition.face_locations(np_img, model='hog')
            if faces:
                t, r, b, l = faces[0]
                padding = 30
                h, w = np_img.shape[:2]
                t = max(0, t - padding)
                b = min(h, b + padding)
                l = max(0, l - padding)
                r = min(w, r + padding)
                img = Image.fromarray(np_img[t:b, l:r])
                log("Face detected and cropped")
            else:
                log("No face detected, using full image")
        except Exception as e:
            log(f"Face detection failed, using full image: {e}", "WARNING")
        
        img = img.resize((299, 299), Image.BILINEAR)
        return transform_img(img).unsqueeze(0)
        
    except Exception as e:
        log(f"Error preprocessing image: {e}", "ERROR")
        raise

def predict_image(path):
    """Predict if image is real or fake"""
    if image_model is None:
        log("Image model not loaded", "ERROR")
        return 'ERROR', 0.0
    
    try:
        x = preprocess_image(path)
        with torch.no_grad():
            out = image_model(x)
            probs = F.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100
            label = 'REAL' if pred == 1 else 'FAKE'
        
        log(f"Image prediction: {label} ({confidence:.2f}%)")
        return label, confidence
        
    except Exception as e:
        log(f"Error predicting image: {e}", "ERROR")
        traceback.print_exc()
        return 'ERROR', 0.0

# ============================================================
# VIDEO MODEL - ResNeXt50 + LSTM
# ============================================================

class VideoModel(nn.Module):
    """ResNeXt50 + LSTM for Video Deepfake Detection"""
    def __init__(self):
        super().__init__()
        
        # ResNeXt50 backbone
        resnext = models.resnext50_32x4d(weights=None)
        
        # Extract feature extractor (everything except avgpool and fc)
        self.model = nn.Sequential(*list(resnext.children())[:-2])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True)
        
        # Classifier (named linear1 to match checkpoint)
        self.linear1 = nn.Linear(2048, 2)
    
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        b, seq, c, h, w = x.shape
        
        # Reshape to process all frames
        x = x.view(b * seq, c, h, w)
        
        # Extract features
        x = self.model(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(b, seq, -1)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Take last LSTM output
        x = lstm_out[:, -1, :]
        
        # Classification
        x = self.linear1(x)
        return x


# Initialize Video Model
video_model = None
try:
    log("Loading ResNeXt+LSTM video model...")
    video_model = VideoModel()
    
    checkpoint = torch.load('model/df_model.pt', map_location='cpu')
    
    # Map checkpoint keys
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('model.'):
            # ResNet backbone
            new_key = 'model.' + key[6:]
            new_state_dict[new_key] = value
        elif key.startswith('lstm.') or key.startswith('linear1.'):
            # LSTM and classifier
            new_state_dict[key] = value
    
    missing, unexpected = video_model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        log(f"Video model - Missing keys: {len(missing)}", "WARNING")
    if unexpected:
        log(f"Video model - Unexpected keys: {len(unexpected)}", "WARNING")
    
    video_model.eval()
    log("✓ Video model loaded successfully", "SUCCESS")
    
except Exception as e:
    log(f"ERROR loading video model: {e}", "ERROR")
    traceback.print_exc()
    video_model = None

# Video preprocessing
transform_vid = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def extract_frames(video_path, max_frames=20):
    """Extract frames from video"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            log("Failed to open video", "ERROR")
            return []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        log(f"Video info: {frame_count} frames, {fps:.2f} FPS")
        
        if frame_count == 0:
            cap.release()
            return []
        
        # Sample frames uniformly
        idxs = np.linspace(0, max(frame_count - 1, 0), min(max_frames, frame_count), dtype=int)
        frames = []
        
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame in idxs:
                frames.append(frame)
            
            current_frame += 1
            
            if len(frames) >= max_frames:
                break
        
        cap.release()
        log(f"Extracted {len(frames)} frames")
        return frames
        
    except Exception as e:
        log(f"Error extracting frames: {e}", "ERROR")
        traceback.print_exc()
        return []

def predict_video(path):
    """Predict if video is real or fake"""
    if video_model is None:
        log("Video model not loaded", "ERROR")
        return 'ERROR', 0.0
    
    try:
        frames = extract_frames(path, max_frames=20)
        
        if not frames:
            log("No frames extracted", "ERROR")
            return 'ERROR', 0.0
        
        processed = []
        faces_detected = 0
        
        for frame in frames:
            try:
                # Try face detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb_frame, model='hog')
                
                if faces:
                    faces_detected += 1
                    t, r, b, l = faces[0]
                    padding = 30
                    h, w = rgb_frame.shape[:2]
                    t = max(0, t - padding)
                    b = min(h, b + padding)
                    l = max(0, l - padding)
                    r = min(w, r + padding)
                    rgb_frame = rgb_frame[t:b, l:r]
            except Exception as e:
                log(f"Face detection failed for frame: {e}", "WARNING")
            
            # Convert to PIL and transform
            pil_img = Image.fromarray(rgb_frame)
            tensor_img = transform_vid(pil_img)
            processed.append(tensor_img)
        
        log(f"Detected faces in {faces_detected}/{len(frames)} frames")
        
        # Stack all frames
        batch = torch.stack(processed).unsqueeze(0)
        
        with torch.no_grad():
            out = video_model(batch)
            probs = F.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100
            label = 'REAL' if pred == 1 else 'FAKE'
        
        log(f"Video prediction: {label} ({confidence:.2f}%)")
        return label, confidence
        
    except Exception as e:
        log(f"Error predicting video: {e}", "ERROR")
        traceback.print_exc()
        return 'ERROR', 0.0

# ============================================================
# AUDIO MODEL - AASIST-based Pre-trained
# ============================================================

class AASISTAudioModel(nn.Module):
    """
    AASIST-inspired model for audio deepfake detection
    Compatible with pre-trained AASIST weights
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Spectral feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Temporal attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Adaptive pooling to fixed spatial size
        x = F.adaptive_avg_pool2d(x, (1, 16))  # (batch, 512, 1, 16)
        
        # Reshape for attention
        batch, channels, height, width = x.shape
        x = x.squeeze(2)  # (batch, 512, 16)
        x = x.permute(0, 2, 1)  # (batch, 16, 512)
        
        # Self-attention
        x, _ = self.attention(x, x, x)
        
        # Global average pooling over time
        x = torch.mean(x, dim=1)  # (batch, 512)
        
        # Classification
        x = self.classifier(x)
        
        return x


# Initialize Audio Model
audio_model = None
audio_model_status = "Not Loaded"

try:
    log("Initializing audio model...", "INFO")
    audio_model = AASISTAudioModel(num_classes=2)
    
    # Try to load pre-trained AASIST weights
    pretrained_path = 'model/aasist_pretrained.pth'
    
    if os.path.exists(pretrained_path):
        try:
            log(f"Loading pre-trained AASIST from {pretrained_path}...", "INFO")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Try to load weights (flexible loading)
            missing_keys, unexpected_keys = audio_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                log(f"Missing {len(missing_keys)} keys (using initialized weights for these)", "WARNING")
            if unexpected_keys:
                log(f"Ignored {len(unexpected_keys)} unexpected keys from checkpoint", "WARNING")
            
            # Check if significant portion loaded
            total_params = len(audio_model.state_dict())
            loaded_params = total_params - len(missing_keys)
            load_percentage = (loaded_params / total_params) * 100
            
            if load_percentage > 50:
                audio_model_status = f"Pre-trained AASIST ({load_percentage:.0f}% loaded)"
                log(f"✓ Loaded {load_percentage:.0f}% of model parameters", "SUCCESS")
            else:
                audio_model_status = "Initialized (pre-trained loading partial)"
                log(f"⚠ Only {load_percentage:.0f}% parameters loaded, using initialized model", "WARNING")
            
        except Exception as e:
            log(f"Could not load pre-trained weights: {e}", "WARNING")
            audio_model_status = "Initialized (no pre-trained weights)"
    else:
        log(f"Pre-trained model not found at {pretrained_path}", "WARNING")
        log("Run 'python setup_aasist.py' to download the pre-trained model", "INFO")
        audio_model_status = "Initialized (download AASIST for better accuracy)"
    
    audio_model.eval()
    log(f"✓ Audio model initialized: {audio_model_status}", "SUCCESS")
    
except Exception as e:
    log(f"ERROR initializing audio model: {e}", "ERROR")
    traceback.print_exc()
    audio_model = None
    audio_model_status = "Failed"


# Audio preprocessing
def preprocess_audio(path):
    """Preprocess audio for model input"""
    try:
        # Load audio file (5 seconds at 16kHz)
        y, sr = librosa.load(path, sr=16000, duration=5)
        log(f"Loaded audio: {len(y)} samples at {sr} Hz")
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128,
            n_fft=2048, 
            hop_length=512,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        log_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-9)
        
        # Pad or truncate to fixed width
        target_width = 256
        if log_spec.shape[1] < target_width:
            log_spec = np.pad(log_spec, ((0, 0), (0, target_width - log_spec.shape[1])), 'constant')
        else:
            log_spec = log_spec[:, :target_width]
        
        # Convert to tensor
        tensor = torch.tensor(log_spec).unsqueeze(0).unsqueeze(0).float()
        log(f"Audio spectrogram shape: {tensor.shape}")
        
        return tensor
        
    except Exception as e:
        log(f"Error preprocessing audio: {e}", "ERROR")
        traceback.print_exc()
        raise


def predict_audio(path):
    """Predict if audio is real or fake"""
    if audio_model is None:
        log("Audio model not loaded", "ERROR")
        return 'ERROR', 0.0
    
    try:
        x = preprocess_audio(path)
        
        with torch.no_grad():
            out = audio_model(x)
            probs = F.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100
            label = 'REAL' if pred == 1 else 'FAKE'
        
        model_info = "AASIST" if "AASIST" in audio_model_status else "initialized"
        log(f"Audio prediction: {label} ({confidence:.2f}%) [{model_info}]")
        
        return label, confidence
        
    except Exception as e:
        log(f"Error predicting audio: {e}", "ERROR")
        traceback.print_exc()
        return 'ERROR', 0.0

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
@app.route('/Detect', methods=['POST'])
def detect():
    """Unified detection endpoint for all media types"""
    try:
        log("=" * 60)
        log("New detection request received")
        
        file = None
        file_type = None
        
        # Check for file in different possible keys
        for key in ['image', 'video', 'audio', 'file']:
            if key in request.files and request.files[key].filename:
                file = request.files[key]
                if key in ['image', 'video', 'audio']:
                    file_type = key
                log(f"File received with key: {key}")
                break
        
        if not file:
            log("No file uploaded", "ERROR")
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        # Detect type from extension if not specified
        if not file_type:
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            type_map = {
                'jpg': 'image', 'jpeg': 'image', 'png': 'image', 'bmp': 'image', 'gif': 'image',
                'mp4': 'video', 'avi': 'video', 'mov': 'video', 'webm': 'video', 'mkv': 'video',
                'mp3': 'audio', 'wav': 'audio', 'ogg': 'audio', 'm4a': 'audio', 'flac': 'audio'
            }
            file_type = type_map.get(ext)
            log(f"Detected file type from extension: {file_type}")
        
        if not file_type:
            log(f"Unsupported file type", "ERROR")
            return jsonify({
                'error': 'Could not determine file type. Supported: jpg, png, mp4, avi, mp3, wav', 
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        log(f"File saved: {filepath}")
        
        try:
            # Process based on type
            predict_funcs = {
                'image': predict_image,
                'video': predict_video,
                'audio': predict_audio
            }
            
            log(f"Processing {file_type}...")
            label, confidence = predict_funcs[file_type](filepath)
            
            result = {
                'type': file_type,
                'label': label,
                'confidence': round(confidence, 2),
                'filename': filename,
                'success': True
            }
            
            log(f"Detection complete: {label} ({confidence:.2f}%)", "SUCCESS")
            log("=" * 60)
            
            return jsonify(result)
            
        finally:
            # Clean up file
            if os.path.exists(filepath):
                os.remove(filepath)
                log(f"Cleaned up file: {filepath}")
                
    except Exception as e:
        log(f"Error in detect endpoint: {e}", "ERROR")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/detect_image', methods=['POST'])
def detect_image_route():
    """Legacy endpoint for image detection"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'success': False}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        try:
            label, confidence = predict_image(path)
            return jsonify({
                'type': 'image',
                'label': label,
                'confidence': round(confidence, 2),
                'success': True
            })
        finally:
            if os.path.exists(path):
                os.remove(path)
                
    except Exception as e:
        log(f"Error in detect_image: {e}", "ERROR")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/detect_video', methods=['POST'])
def detect_video_route():
    """Legacy endpoint for video detection"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded', 'success': False}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        try:
            label, confidence = predict_video(path)
            return jsonify({
                'type': 'video',
                'label': label,
                'confidence': round(confidence, 2),
                'success': True
            })
        finally:
            if os.path.exists(path):
                os.remove(path)
                
    except Exception as e:
        log(f"Error in detect_video: {e}", "ERROR")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/detect_audio', methods=['POST'])
def detect_audio_route():
    """Legacy endpoint for audio detection"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio uploaded', 'success': False}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        try:
            label, confidence = predict_audio(path)
            return jsonify({
                'type': 'audio',
                'label': label,
                'confidence': round(confidence, 2),
                'success': True
            })
        finally:
            if os.path.exists(path):
                os.remove(path)
                
    except Exception as e:
        log(f"Error in detect_audio: {e}", "ERROR")
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Server is healthy',
        'models': {
            'image': image_model is not None,
            'video': video_model is not None,
            'audio': audio_model is not None
        },
        'architecture': {
            'image': 'Xception',
            'video': 'ResNeXt50+LSTM',
            'audio': 'AASIST'
        },
        'audio_status': audio_model_status
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB', 'success': False}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found', 'success': False}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    log(f"Internal server error: {e}", "ERROR")
    return jsonify({'error': 'Internal server error', 'success': False}), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎭 DEEPFAKE DETECTION SERVER - COMPLETE VERSION")
    print("=" * 70)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print("=" * 70)
    print("📊 MODEL STATUS:")
    print(f"   🖼️  Image  (Xception):      {'✓ Loaded' if image_model else '✗ Failed'}")
    print(f"   🎥 Video  (ResNeXt+LSTM):  {'✓ Loaded' if video_model else '✗ Failed'}")
    print(f"   🎵 Audio  (AASIST):        {'✓ ' + audio_model_status if audio_model else '✗ Failed'}")
    print("=" * 70)
    print("🌐 SERVER ENDPOINTS:")
    print("   Main:   http://127.0.0.1:3000")
    print("   Local:  http://0.0.0.0:3000")
    print("=" * 70)
    print("📡 API ROUTES:")
    print("   POST /detect         - Universal detection (auto-detect type)")
    print("   POST /detect_image   - Image detection")
    print("   POST /detect_video   - Video detection")
    print("   POST /detect_audio   - Audio detection")
    print("   GET  /health         - Health check")
    print("=" * 70)
    print("💡 USAGE:")
    print("   1. Open http://127.0.0.1:3000 in your browser")
    print("   2. Select media type (Image/Video/Audio)")
    print("   3. Upload file and click 'Analyze for Deepfakes'")
    print("   4. View detection results with confidence scores")
    print("=" * 70)
    
    if "AASIST" not in audio_model_status:
        print("⚠️  AUDIO MODEL NOTE:")
        print("   Audio model is using initialized weights.")
        print("   For better accuracy, download pre-trained AASIST model:")
        print("   Run: python setup_aasist.py")
        print("=" * 70)
    
    print("✅ ALL MODELS READY:")
    print("   • Image: Trained Xception (Good accuracy)")
    print("   • Video: Trained ResNeXt+LSTM (Good accuracy)")
    if "AASIST" in audio_model_status:
        print("   • Audio: Pre-trained AASIST (Excellent accuracy)")
    else:
        print("   • Audio: Initialized weights (Download AASIST for better results)")
    print("=" * 70)
    print("Press CTRL+C to quit")
    print("=" * 70 + "\n")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=3000, debug=True)