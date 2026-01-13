// ============================================================================
// Main Application Module - Avatar System Orchestrator
// ============================================================================

const App = {
    // Application state
    state: {
        imageFile: null,
        audioFile: null,
        videoFile: null,
        imagePath: null,
        audioPath: null,
        videoPath: null,
        extractedImagePath: null,
        taskId: null,
        isGenerating: false,
        generatedVideoUrl: null
    },

    // Configuration
    config: {
        maxImageSize: 10, // MB
        maxAudioSize: 50, // MB
        maxVideoSize: 100, // MB
        allowedImageTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'],
        allowedAudioTypes: ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/x-m4a', 'audio/flac'],
        allowedVideoTypes: ['video/mp4', 'video/avi', 'video/webm', 'video/quicktime', 'video/x-matroska'],
        statusPollInterval: 2000 // ms
    },

    // DOM elements
    elements: {},

    /**
     * Initialize application
     */
    init() {
        console.log('[App] Initializing Avatar System Orchestrator...');

        this.cacheElements();
        this.setupEventListeners();
        this.checkBrowserSupport();

        console.log('[App] Application initialized successfully');
    },

    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            // Upload zones
            imageUploadZone: document.getElementById('imageUploadZone'),
            audioUploadZone: document.getElementById('audioUploadZone'),
            videoUploadZone: document.getElementById('videoUploadZone'),
            imageInput: document.getElementById('imageInput'),
            audioInput: document.getElementById('audioInput'),
            videoInput: document.getElementById('videoInput'),
            imagePreview: document.getElementById('imagePreview'),
            audioPreview: document.getElementById('audioPreview'),
            videoPreview: document.getElementById('videoPreview'),
            audioPlayer: document.getElementById('audioPlayer'),

            // Configuration
            fpsSelect: document.getElementById('fpsSelect'),
            resolutionSelect: document.getElementById('resolutionSelect'),
            generateBtn: document.getElementById('generateBtn'),

            // Result
            resultVideo: document.getElementById('resultVideo'),
            videoOverlay: document.getElementById('videoOverlay'),
            playBtn: document.getElementById('playBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            resetBtn: document.getElementById('resetBtn')
        };
    },

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Image upload
        this.elements.imageUploadZone.addEventListener('click', () => {
            this.elements.imageInput.click();
        });

        this.elements.imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0], 'image');
            }
        });

        this.setupDragAndDrop(this.elements.imageUploadZone, 'image');

        // Audio upload
        this.elements.audioUploadZone.addEventListener('click', () => {
            this.elements.audioInput.click();
        });

        this.elements.audioInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0], 'audio');
            }
        });

        this.setupDragAndDrop(this.elements.audioUploadZone, 'audio');

        // Video upload
        this.elements.videoUploadZone.addEventListener('click', () => {
            this.elements.videoInput.click();
        });

        this.elements.videoInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0], 'video');
            }
        });

        this.setupDragAndDrop(this.elements.videoUploadZone, 'video');

        // Remove buttons
        document.querySelectorAll('.btn-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const target = btn.getAttribute('data-target');
                this.removeFile(target);
            });
        });

        // Generate button
        this.elements.generateBtn.addEventListener('click', () => {
            this.startGeneration();
        });

        // Video controls
        if (this.elements.playBtn) {
            this.elements.playBtn.addEventListener('click', () => {
                this.elements.resultVideo.play();
            });
        }

        if (this.elements.resultVideo) {
            this.elements.resultVideo.addEventListener('play', () => {
                if (this.elements.videoOverlay) {
                    this.elements.videoOverlay.style.opacity = '0';
                }
            });

            this.elements.resultVideo.addEventListener('pause', () => {
                if (this.elements.videoOverlay) {
                    this.elements.videoOverlay.style.opacity = '1';
                }
            });
        }

        // Download button
        if (this.elements.downloadBtn) {
            this.elements.downloadBtn.addEventListener('click', () => {
                this.downloadVideo();
            });
        }

        // Reset button
        if (this.elements.resetBtn) {
            this.elements.resetBtn.addEventListener('click', () => {
                this.resetApplication();
            });
        }
    },

    /**
     * Setup drag and drop for upload zone
     * @param {HTMLElement} zone - Upload zone element
     * @param {string} type - File type (image or audio)
     */
    setupDragAndDrop(zone, type) {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('drag-over');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0], type);
            }
        });
    },

    /**
     * Handle file selection
     * @param {File} file - Selected file
     * @param {string} type - File type (image or audio)
     */
    async handleFileSelect(file, type) {
        console.log(`[App] File selected: ${file.name} (${type})`);

        // Validate file type
        const allowedTypes = type === 'image'
            ? this.config.allowedImageTypes
            : type === 'audio'
                ? this.config.allowedAudioTypes
                : this.config.allowedVideoTypes;

        if (!UI.validateFileType(file, allowedTypes)) {
            UI.showToast('error', 'Invalid File Type',
                `Please select a valid ${type} file.`);
            return;
        }

        // Validate file size
        const maxSize = type === 'image'
            ? this.config.maxImageSize
            : type === 'audio'
                ? this.config.maxAudioSize
                : this.config.maxVideoSize;

        if (!UI.validateFileSize(file, maxSize)) {
            UI.showToast('error', 'File Too Large',
                `File size must be less than ${maxSize}MB.`);
            return;
        }

        // Store file and show preview
        if (type === 'image') {
            this.state.imageFile = file;
            await this.showImagePreview(file);
        } else if (type === 'audio') {
            this.state.audioFile = file;
            this.showAudioPreview(file);
        } else if (type === 'video') {
            this.state.videoFile = file;
            await this.showVideoPreview(file);
        }

        // Update generate button state
        this.updateGenerateButton();
    },

    /**
     * Show image preview
     * @param {File} file - Image file
     */
    async showImagePreview(file) {
        try {
            const preview = this.elements.imagePreview;
            const img = preview.querySelector('img');
            const fileName = preview.querySelector('.file-name');
            const fileSize = preview.querySelector('.file-size');

            const dataUrl = await UI.createImagePreview(file);
            img.src = dataUrl;
            fileName.textContent = file.name;
            fileSize.textContent = UI.formatFileSize(file.size);

            preview.style.display = 'block';
            this.elements.imageUploadZone.querySelector('.upload-content').style.display = 'none';

            console.log('[App] Image preview displayed');

        } catch (error) {
            console.error('[App] Failed to show image preview:', error);
            UI.showToast('error', 'Preview Error', 'Failed to display image preview.');
        }
    },

    /**
     * Show audio preview
     * @param {File} file - Audio file
     */
    showAudioPreview(file) {
        const preview = this.elements.audioPreview;
        const fileName = preview.querySelector('.file-name');
        const fileSize = preview.querySelector('.file-size');
        const player = this.elements.audioPlayer;

        const url = URL.createObjectURL(file);
        player.src = url;
        fileName.textContent = file.name;
        fileSize.textContent = UI.formatFileSize(file.size);

        preview.style.display = 'block';
        this.elements.audioUploadZone.querySelector('.upload-content').style.display = 'none';

        console.log('[App] Audio preview displayed');
    },

    /**
     * Show video preview
     * @param {File} file - Video file
     */
    async showVideoPreview(file) {
        const preview = this.elements.videoPreview;
        const fileName = preview.querySelector('.file-name');
        const fileSize = preview.querySelector('.file-size');
        const video = preview.querySelector('video');

        const url = URL.createObjectURL(file);
        video.src = url;
        fileName.textContent = file.name;
        fileSize.textContent = UI.formatFileSize(file.size);

        preview.style.display = 'block';
        this.elements.videoUploadZone.querySelector('.upload-content').style.display = 'none';

        console.log('[App] Video preview displayed');
    },

    /**
     * Remove uploaded file
     * @param {string} type - File type (image or audio)
     */
    removeFile(type) {
        if (type === 'image') {
            this.state.imageFile = null;
            this.state.imagePath = null;
            this.state.extractedImagePath = null;
            this.elements.imagePreview.style.display = 'none';
            this.elements.imageUploadZone.querySelector('.upload-content').style.display = 'flex';
            this.elements.imageInput.value = '';
            console.log('[App] Image file removed');
        } else if (type === 'audio') {
            this.state.audioFile = null;
            this.state.audioPath = null;
            this.elements.audioPreview.style.display = 'none';
            this.elements.audioUploadZone.querySelector('.upload-content').style.display = 'flex';
            this.elements.audioInput.value = '';
            if (this.elements.audioPlayer.src) {
                URL.revokeObjectURL(this.elements.audioPlayer.src);
                this.elements.audioPlayer.src = '';
            }
            console.log('[App] Audio file removed');
        } else if (type === 'video') {
            this.state.videoFile = null;
            this.state.videoPath = null;
            this.state.extractedImagePath = null;
            this.elements.videoPreview.style.display = 'none';
            this.elements.videoUploadZone.querySelector('.upload-content').style.display = 'flex';
            this.elements.videoInput.value = '';
            const video = this.elements.videoPreview.querySelector('video');
            if (video && video.src) {
                URL.revokeObjectURL(video.src);
                video.src = '';
            }
            console.log('[App] Video file removed');
        }

        this.updateGenerateButton();
    },

    /**
     * Update generate button enabled state
     */
    updateGenerateButton() {
        // Allow generation with either: (image + audio) OR (video + audio)
        const hasImageAndAudio = this.state.imageFile && this.state.audioFile;
        const hasVideoAndAudio = this.state.videoFile && this.state.audioFile;
        const canGenerate = (hasImageAndAudio || hasVideoAndAudio) && !this.state.isGenerating;
        this.elements.generateBtn.disabled = !canGenerate;
    },

    /**
     * Start avatar generation
     */
    async startGeneration() {
        if (this.state.isGenerating) return;

        this.state.isGenerating = true;
        this.updateGenerateButton();
        UI.hideResult();
        UI.showProgress();

        try {
            // Determine input mode: video or image+audio
            const useVideo = this.state.videoFile !== null;

            if (useVideo) {
                // Upload video first
                UI.updateProgress(10, 'Uploading video...');
                const videoResult = await API.uploadVideo(this.state.videoFile, (progress) => {
                    UI.updateProgress(10 + (progress * 0.3), 'Uploading video...');
                });
                this.state.videoPath = videoResult.path;

                // Use extracted image if available
                if (videoResult.extracted_image) {
                    this.state.extractedImagePath = videoResult.extracted_image;
                    this.state.imagePath = videoResult.extracted_image;
                    console.log('[App] Using extracted frame from video:', videoResult.extracted_image);
                } else {
                    UI.showToast('warning', 'Frame Extraction Failed',
                        'Could not extract frame from video. Please upload an image separately.');
                    this.state.isGenerating = false;
                    this.updateGenerateButton();
                    UI.hideProgress();
                    return;
                }
            } else {
                // Upload image
                UI.updateProgress(10, 'Uploading image...');
                const imageResult = await API.uploadImage(this.state.imageFile, (progress) => {
                    UI.updateProgress(10 + (progress * 0.2), 'Uploading image...');
                });
                this.state.imagePath = imageResult.path;
                console.log('[App] Image uploaded:', imageResult);
            }

            // Upload audio
            UI.updateProgress(30, 'Uploading audio...');
            const audioResult = await API.uploadAudio(this.state.audioFile, (progress) => {
                UI.updateProgress(30 + (progress * 0.2), 'Uploading audio...');
            });
            this.state.audioPath = audioResult.path;
            console.log('[App] Audio uploaded:', audioResult);

            // Get generation parameters
            const fps = parseInt(this.elements.fpsSelect.value);
            const resolution = this.elements.resolutionSelect.value.split('x').map(Number);

            // Start generation
            UI.updateProgress(50, 'Generating avatar...');
            const generateParams = {
                audio_path: this.state.audioPath,
                image_path: this.state.imagePath,
                fps: fps,
                resolution: resolution
            };

            const generateResult = await API.generateAvatar(generateParams);
            console.log('[App] Generation started:', generateResult);

            // Simulate progress (since we don't have real-time updates yet)
            await this.simulateProgress();

            // Show result
            this.showGenerationResult(generateResult);

            UI.showToast('success', 'Success!', 'Avatar generated successfully.');

        } catch (error) {
            console.error('[App] Generation failed:', error);
            UI.showToast('error', 'Generation Failed', API.getErrorMessage(error));
            UI.hideProgress();
        } finally {
            this.state.isGenerating = false;
            this.updateGenerateButton();
        }
    },

    /**
     * Simulate progress updates
     */
    async simulateProgress() {
        const steps = [
            { progress: 60, status: 'Processing audio features...' },
            { progress: 70, status: 'Analyzing facial landmarks...' },
            { progress: 80, status: 'Generating lip movements...' },
            { progress: 90, status: 'Rendering video frames...' },
            { progress: 95, status: 'Finalizing video...' }
        ];

        for (const step of steps) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            UI.updateProgress(step.progress, step.status);
        }

        UI.updateProgress(100, 'Complete!');
    },

    /**
     * Show generation result
     * @param {Object} result - Generation result
     */
    showGenerationResult(result) {
        UI.hideProgress();
        UI.showResult();

        // Set video source (using placeholder for now since generation is not implemented)
        const videoUrl = result.video_path || '/static/demo-video.mp4';
        this.state.generatedVideoUrl = videoUrl;
        this.elements.resultVideo.src = videoUrl;

        console.log('[App] Displaying result:', result);
        UI.scrollToElement('resultSection');
    },

    /**
     * Download generated video
     */
    async downloadVideo() {
        if (!this.state.generatedVideoUrl) {
            UI.showToast('warning', 'No Video', 'No video available to download.');
            return;
        }

        try {
            const filename = `avatar_${Date.now()}.mp4`;
            await API.downloadFile(this.state.generatedVideoUrl, filename);
            UI.showToast('success', 'Downloaded', 'Video downloaded successfully.');
        } catch (error) {
            console.error('[App] Download failed:', error);
            UI.showToast('error', 'Download Failed', 'Failed to download video.');
        }
    },

    /**
     * Reset application to initial state
     */
    resetApplication() {
        // Clear state
        this.removeFile('image');
        this.removeFile('audio');
        this.removeFile('video');
        this.state.taskId = null;
        this.state.generatedVideoUrl = null;
        this.state.extractedImagePath = null;

        // Reset UI
        UI.hideProgress();
        UI.hideResult();
        this.elements.resultVideo.src = '';

        // Reset to defaults
        this.elements.fpsSelect.value = '25';
        this.elements.resolutionSelect.value = '512x512';

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });

        console.log('[App] Application reset');
        UI.showToast('info', 'Reset', 'Application reset to initial state');
    },

    /**
     * Check browser support for required features
     */
    checkBrowserSupport() {
        const support = UI.checkBrowserSupport();

        if (!support.allSupported) {
            const missing = Object.entries(support)
                .filter(([key, value]) => !value && key !== 'allSupported')
                .map(([key]) => key);

            UI.showToast('warning', 'Browser Compatibility',
                `Some features may not work. Missing: ${missing.join(', ')}`, 0);

            console.warn('[App] Browser support issues:', missing);
        }
    }
};

// Initialize application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}

// Export for debugging
window.App = App;

// Debug helper
window.debugApp = () => {
    console.log('Application State:', App.state);
    console.log('Application Config:', App.config);
    console.log('API Status:', API);
    console.log('UI Status:', UI);
};
