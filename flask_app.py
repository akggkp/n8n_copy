from flask import Flask, render_template_string, request, jsonify
import os
import requests
from pathlib import Path
from datetime import datetime
import subprocess
import threading

app = Flask(__name__)
UPLOAD_FOLDER = '/data/videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'webm', 'mov', 'flv'}

# Ensure upload folder exists
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_youtube_url(url):
    """Check if URL is a valid YouTube URL"""
    youtube_patterns = [
        'youtube.com/watch',
        'youtu.be/',
        'youtube.com/playlist',
        'youtube.com/channel',
    ]
    return any(pattern in url for pattern in youtube_patterns)

def download_youtube_video(youtube_url, output_path):
    """Download video from YouTube using yt-dlp"""
    try:
        # Check if yt-dlp is installed
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Exception("yt-dlp not installed. Run: pip install yt-dlp")
    
    try:
        # Download video
        cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]',  # Download best MP4 quality
            '-o', output_path,
            '--quiet',
            '--no-warnings',
            youtube_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            raise Exception(f"Download failed: {result.stderr}")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Download timeout - video too large or connection issue")
    except Exception as e:
        raise Exception(f"YouTube download error: {str(e)}")

def send_to_n8n_webhook(webhook_data):
    """Send processed video data to n8n webhook"""
    try:
        response = requests.post(
            'http://localhost:5678/webhook/upload-video',
            json=webhook_data,
            timeout=10
        )
        print(f"[n8n] Status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"[n8n] Error: {e}")
        return False

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading AI - Video Upload & YouTube Downloader</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 900px;
                margin: 0 auto;
            }
            
            .header {
                text-align: center;
                color: white;
                margin-bottom: 40px;
            }
            
            .header h1 {
                font-size: 32px;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 14px;
                opacity: 0.9;
            }
            
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            
            .tab-btn {
                flex: 1;
                padding: 12px;
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s;
            }
            
            .tab-btn.active {
                background: white;
                color: #667eea;
                border-color: white;
            }
            
            .tab-btn:hover {
                background: rgba(255, 255, 255, 0.3);
                border-color: white;
            }
            
            .tab-content {
                display: none;
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            }
            
            .tab-content.active {
                display: block;
            }
            
            .upload-area {
                border: 2px dashed #667eea;
                padding: 40px;
                text-align: center;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
                background: #f9f9f9;
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background: #f5f5f5;
            }
            
            .upload-area.dragover {
                border-color: #764ba2;
                background: #eee;
            }
            
            .upload-icon {
                font-size: 40px;
                margin-bottom: 10px;
            }
            
            .upload-text {
                color: #333;
                font-weight: 600;
                margin-bottom: 5px;
            }
            
            .upload-subtext {
                color: #999;
                font-size: 12px;
            }
            
            input[type="file"],
            input[type="text"],
            textarea {
                display: block;
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                font-family: inherit;
            }
            
            input[type="file"] {
                display: none;
            }
            
            input[type="text"]:focus,
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
            }
            
            label {
                display: block;
                color: #333;
                font-weight: 600;
                margin-top: 15px;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            .file-info {
                background: #f0f4ff;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-size: 12px;
                color: #667eea;
                display: none;
            }
            
            .file-info.show {
                display: block;
            }
            
            .buttons {
                display: flex;
                gap: 10px;
                margin-top: 25px;
            }
            
            button {
                flex: 1;
                padding: 12px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .btn-submit {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .btn-submit:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            }
            
            .btn-submit:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .btn-clear {
                background: #f0f0f0;
                color: #333;
            }
            
            .btn-clear:hover {
                background: #e0e0e0;
            }
            
            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                font-size: 14px;
                display: none;
            }
            
            .status.show {
                display: block;
            }
            
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .status.loading {
                background: #cfe2ff;
                color: #084298;
                border: 1px solid #b6d4fe;
            }
            
            .spinner {
                display: inline-block;
                width: 12px;
                height: 12px;
                border: 2px solid #084298;
                border-radius: 50%;
                border-top-color: transparent;
                animation: spin 0.6s linear infinite;
                margin-right: 8px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎬 Trading AI Video Uploader</h1>
                <p>Upload local videos or download from YouTube for AI processing</p>
            </div>
            
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('upload')">📁 Upload Local Video</button>
                <button class="tab-btn" onclick="switchTab('youtube')">🎥 YouTube Download</button>
            </div>
            
            <!-- Upload Tab -->
            <div id="upload" class="tab-content active">
                <form id="uploadForm">
                    <label>Select Video File</label>
                    <div class="upload-area" id="uploadArea">
                        <div class="upload-icon">📂</div>
                        <div class="upload-text">Click to upload or drag and drop</div>
                        <div class="upload-subtext">MP4, WebM, AVI, MOV (Max 500MB)</div>
                        <input type="file" id="videoFile" accept=".mp4,.webm,.avi,.mov,.flv" />
                    </div>
                    <div class="file-info" id="fileInfo"></div>
                    
                    <label>Video ID (Optional)</label>
                    <input type="text" id="videoId" placeholder="e.g., nifty-analysis-001" />
                    
                    <label>Description (Optional)</label>
                    <textarea id="description" placeholder="Add notes about this video..."></textarea>
                    
                    <div class="buttons">
                        <button type="submit" class="btn-submit" id="submitBtn">Upload & Process</button>
                        <button type="button" class="btn-clear" onclick="clearForm('uploadForm', 'fileInfo')">Clear</button>
                    </div>
                    
                    <div class="status" id="uploadStatus"></div>
                </form>
            </div>
            
            <!-- YouTube Tab -->
            <div id="youtube" class="tab-content">
                <form id="youtubeForm">
                    <label>YouTube URL</label>
                    <input type="text" id="youtubeUrl" placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/..." />
                    
                    <label>Video ID (Optional)</label>
                    <input type="text" id="youtubeVideoId" placeholder="e.g., youtube-trading-001" />
                    
                    <label>Description (Optional)</label>
                    <textarea id="youtubeDescription" placeholder="Add notes about this video..."></textarea>
                    
                    <div class="buttons">
                        <button type="submit" class="btn-submit" id="youtubeSubmitBtn">Download & Process</button>
                        <button type="button" class="btn-clear" onclick="clearForm('youtubeForm')">Clear</button>
                    </div>
                    
                    <div class="status" id="youtubeStatus"></div>
                </form>
            </div>
        </div>
        
        <script>
            function switchTab(tab) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById(tab).classList.add('active');
                event.target.classList.add('active');
            }
            
            function clearForm(formId, infoId) {
                document.getElementById(formId).reset();
                if (infoId) document.getElementById(infoId).classList.remove('show');
            }
            
            // Upload form
            const uploadArea = document.getElementById('uploadArea');
            const videoFile = document.getElementById('videoFile');
            const fileInfo = document.getElementById('fileInfo');
            const uploadForm = document.getElementById('uploadForm');
            const uploadStatus = document.getElementById('uploadStatus');
            
            let selectedFile = null;
            
            uploadArea.addEventListener('click', () => videoFile.click());
            videoFile.addEventListener('change', handleFileSelect);
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                videoFile.files = e.dataTransfer.files;
                handleFileSelect();
            });
            
            function handleFileSelect() {
                selectedFile = videoFile.files[0];
                if (selectedFile) {
                    const sizeMB = (selectedFile.size / (1024 * 1024)).toFixed(2);
                    fileInfo.textContent = '✓ Selected: ' + selectedFile.name + ' (' + sizeMB + 'MB)';
                    fileInfo.classList.add('show');
                }
            }
            
            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                if (!selectedFile) {
                    showStatus(uploadStatus, 'Please select a video file', 'error');
                    return;
                }
                
                showStatus(uploadStatus, '<span class="spinner"></span>Uploading...', 'loading');
                document.getElementById('submitBtn').disabled = true;
                
                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('video_id', document.getElementById('videoId').value);
                    formData.append('description', document.getElementById('description').value);
                    
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showStatus(uploadStatus, '✓ Uploaded! ID: ' + result.video_id + '\\nStatus: Queued for processing', 'success');
                        uploadForm.reset();
                        fileInfo.classList.remove('show');
                    } else {
                        showStatus(uploadStatus, '✗ Error: ' + result.error, 'error');
                    }
                } catch (error) {
                    showStatus(uploadStatus, '✗ Upload failed: ' + error.message, 'error');
                } finally {
                    document.getElementById('submitBtn').disabled = false;
                }
            });
            
            // YouTube form
            const youtubeForm = document.getElementById('youtubeForm');
            const youtubeStatus = document.getElementById('youtubeStatus');
            
            youtubeForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const youtubeUrl = document.getElementById('youtubeUrl').value;
                if (!youtubeUrl) {
                    showStatus(youtubeStatus, 'Please enter a YouTube URL', 'error');
                    return;
                }
                
                showStatus(youtubeStatus, '<span class="spinner"></span>Downloading and processing...', 'loading');
                document.getElementById('youtubeSubmitBtn').disabled = true;
                
                try {
                    const response = await fetch('/download-youtube', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            youtube_url: youtubeUrl,
                            video_id: document.getElementById('youtubeVideoId').value,
                            description: document.getElementById('youtubeDescription').value
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showStatus(youtubeStatus, '✓ Downloaded! ID: ' + result.video_id + '\\nFilename: ' + result.filename + '\\nStatus: Queued for processing', 'success');
                        youtubeForm.reset();
                    } else {
                        showStatus(youtubeStatus, '✗ Error: ' + result.error, 'error');
                    }
                } catch (error) {
                    showStatus(youtubeStatus, '✗ Download failed: ' + error.message, 'error');
                } finally {
                    document.getElementById('youtubeSubmitBtn').disabled = false;
                }
            });
            
            function showStatus(element, message, type) {
                element.className = 'status show ' + type;
                element.innerHTML = message;
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle local file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate video ID
        video_id = request.form.get('video_id', f'local-{int(datetime.now().timestamp())}')
        
        # Save file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Get file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Send to n8n webhook
        webhook_data = {
            'video_id': video_id,
            'filename': file.filename,
            'file_path': filepath,
            'file_size_mb': round(file_size_mb, 2),
            'description': request.form.get('description', ''),
            'source': 'local_upload'
        }
        
        # Send to n8n in background
        threading.Thread(target=send_to_n8n_webhook, args=(webhook_data,)).start()
        
        print(f"[upload] {video_id} - {file.filename} ({file_size_mb:.2f}MB)")
        
        return jsonify({
            'status': 'success',
            'video_id': video_id,
            'filename': file.filename,
            'file_size_mb': round(file_size_mb, 2),
            'message': 'Video queued for processing'
        }), 200
        
    except Exception as e:
        print(f"[upload_error] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-youtube', methods=['POST'])
def download_youtube():
    """Handle YouTube video downloads"""
    try:
        data = request.get_json()
        youtube_url = data.get('youtube_url', '').strip()
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        if not is_youtube_url(youtube_url):
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        # Generate video ID
        video_id = data.get('video_id', f'youtube-{int(datetime.now().timestamp())}')
        
        # Generate output filename
        output_filename = f"{video_id}.mp4"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        print(f"[youtube] Downloading {video_id}...")
        
        # Download video
        try:
            download_youtube_video(youtube_url, output_path)
        except Exception as e:
            print(f"[youtube_error] {str(e)}")
            return jsonify({'error': f'YouTube download failed: {str(e)}'}), 400
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        # Send to n8n webhook
        webhook_data = {
            'video_id': video_id,
            'filename': output_filename,
            'file_path': output_path,
            'file_size_mb': round(file_size_mb, 2),
            'youtube_url': youtube_url,
            'description': data.get('description', ''),
            'source': 'youtube_download'
        }
        
        # Send to n8n in background
        threading.Thread(target=send_to_n8n_webhook, args=(webhook_data,)).start()
        
        print(f"[youtube] {video_id} - {output_filename} ({file_size_mb:.2f}MB)")
        
        return jsonify({
            'status': 'success',
            'video_id': video_id,
            'filename': output_filename,
            'file_size_mb': round(file_size_mb, 2),
            'youtube_url': youtube_url,
            'message': 'Video downloaded and queued for processing'
        }), 200
        
    except Exception as e:
        print(f"[youtube_main_error] {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🎬 Trading AI Video Server")
    print("=" * 50)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("Running on: http://0.0.0.0:8888")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8888, debug=True)