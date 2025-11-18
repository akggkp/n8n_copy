// Dashboard JavaScript

const API_BASE = '';

// Auto-refresh interval (30 seconds)
let autoRefreshInterval;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadVideos();
    loadStrategies();
    
    // Auto-refresh every 30 seconds
    autoRefreshInterval = setInterval(function() {
        loadStats();
        loadVideos();
        loadStrategies();
    }, 30000);
});

// Load statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        
        document.getElementById('total-videos').textContent = data.total_videos;
        document.getElementById('completed').textContent = data.completed;
        document.getElementById('processing').textContent = data.processing;
        document.getElementById('profitable').textContent = data.profitable_strategies;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load videos
async function loadVideos() {
    try {
        const response = await fetch(`${API_BASE}/api/videos?limit=20`);
        const data = await response.json();
        
        const tbody = document.getElementById('videos-tbody');
        
        if (data.videos.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading">No videos found</td></tr>';
            return;
        }
        
        tbody.innerHTML = data.videos.map(video => `
            <tr>
                <td><code>${video.video_id}</code></td>
                <td>${video.filename}</td>
                <td><span class="status-badge status-${video.status}">${video.status}</span></td>
                <td>${video.processing_time ? video.processing_time.toFixed(1) + 's' : '-'}</td>
                <td>${video.processed_at ? new Date(video.processed_at).toLocaleString() : '-'}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading videos:', error);
        document.getElementById('videos-tbody').innerHTML = 
            '<tr><td colspan="5" class="loading">Error loading videos</td></tr>';
    }
}

// Load profitable strategies
async function loadStrategies() {
    try {
        const response = await fetch(`${API_BASE}/api/profitable-strategies`);
        const data = await response.json();
        
        const tbody = document.getElementById('strategies-tbody');
        
        if (data.strategies.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading">No profitable strategies yet</td></tr>';
            return;
        }
        
        tbody.innerHTML = data.strategies.map(strategy => `
            <tr>
                <td><code>${strategy.video_id}</code></td>
                <td>${strategy.strategy_name}</td>
                <td><strong style="color: #28a745;">${strategy.win_rate}%</strong></td>
                <td><strong style="color: #667eea;">${strategy.profit_factor}</strong></td>
                <td>${strategy.created_at ? new Date(strategy.created_at).toLocaleString() : '-'}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading strategies:', error);
        document.getElementById('strategies-tbody').innerHTML = 
            '<tr><td colspan="5" class="loading">Error loading strategies</td></tr>';
    }
}

// Refresh functions
function refreshVideos() {
    loadVideos();
}

function refreshStrategies() {
    loadStrategies();
}

// Trigger processing
async function triggerProcessing(event) {
    event.preventDefault();
    
    const videoId = document.getElementById('video-id').value;
    const filename = document.getElementById('filename').value;
    const filePath = document.getElementById('file-path').value;
    
    const resultDiv = document.getElementById('trigger-result');
    resultDiv.style.display = 'block';
    resultDiv.className = '';
    resultDiv.textContent = 'Triggering processing...';
    
    try {
        const response = await fetch(`${API_BASE}/api/trigger-processing`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_id: videoId,
                filename: filename,
                file_path: filePath
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultDiv.className = 'result-success';
            resultDiv.textContent = `✓ Processing started! Task ID: ${data.task_id}`;
            
            // Clear form
            document.getElementById('trigger-form').reset();
            
            // Refresh videos after 2 seconds
            setTimeout(loadVideos, 2000);
        } else {
            resultDiv.className = 'result-error';
            resultDiv.textContent = `✗ Error: ${data.error}`;
        }
    } catch (error) {
        resultDiv.className = 'result-error';
        resultDiv.textContent = `✗ Error: ${error.message}`;
    }
    
    // Hide result after 10 seconds
    setTimeout(() => {
        resultDiv.style.display = 'none';
    }, 10000);
}

// Upload video
async function uploadVideo(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('upload-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }
    
    const resultDiv = document.getElementById('upload-result');
    resultDiv.style.display = 'block';
    resultDiv.className = '';
    resultDiv.textContent = 'Uploading video...';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultDiv.className = 'result-success';
            resultDiv.textContent = `✓ Upload successful! Task ID: ${data.task_id}`;
            
            // Clear form
            document.getElementById('upload-form').reset();
            
            // Refresh videos after 2 seconds
            setTimeout(loadVideos, 2000);
        } else {
            resultDiv.className = 'result-error';
            resultDiv.textContent = `✗ Error: ${data.error}`;
        }
    } catch (error) {
        resultDiv.className = 'result-error';
        resultDiv.textContent = `✗ Error: ${error.message}`;
    }
    
    // Hide result after 10 seconds
    setTimeout(() => {
        resultDiv.style.display = 'none';
    }, 10000);
}