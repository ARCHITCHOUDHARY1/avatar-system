// ============================================================================
// Enhanced App Controller - Microphone + Processing Visualizer Integration
// ============================================================================

const EnhancedApp = {
    // Extend existing App module
    ...App,

    // Enhanced state
    enhancedState: {
        microphoneEnabled: false,
        recordedAudio: null,
        processingActive: false,
        debugMode: false,
        consoleLogs: []
    },

    /**
     * Initialize enhanced features
     */
    initEnhanced() {
        console.log('[EnhancedApp] Initializing enhanced features...');

        this.setupMicrophoneUI();
        this.setupProcessingVisualizerUI();
        this.setupIOComparisonUI();
        this.setupErrorConsole();
        this.checkMicrophoneSupport();

        console.log('[EnhancedApp] Enhanced features initialized');
    },

    /**
     * Setup microphone UI
     */
    setupMicrophoneUI() {
        const btnStartRecording = document.getElementById('btnStartRecording');
        const btnPauseRecording = document.getElementById('btnPauseRecording');
        const btnResumeRecording = document.getElementById('btnResumeRecording');
        const btnStopRecording = document.getElementById('btnStopRecording');
        const btnCancelRecording = document.getElementById('btnCancelRecording');
        const btnUseRecording = document.getElementById('btnUseRecording');

        if (!btnStartRecording) return;

        // Start recording
        btnStartRecording.addEventListener('click', async () => {
            try {
                await Microphone.startRecording();
                this.onRecordingStarted();
            } catch (error) {
                console.error('[EnhancedApp] Failed to start recording:', error);
                this.logToConsole('error', `Failed to start recording: ${error.message}`);
                UI.showToast('error', 'Recording Failed', error.message);
            }
        });

        // Pause recording
        btnPauseRecording.addEventListener('click', () => {
            Microphone.pauseRecording();
            this.onRecordingPaused();
        });

        // Resume recording
        btnResumeRecording.addEventListener('click', () => {
            Microphone.resumeRecording();
            this.onRecordingResumed();
        });

        // Stop recording
        btnStopRecording.addEventListener('click', async () => {
            const file = await Microphone.stopRecording();
            this.onRecordingStopped(file);
        });

        // Cancel recording
        btnCancelRecording.addEventListener('click', () => {
            Microphone.cancelRecording();
            this.onRecordingCancelled();
        });

        // Use recording
        btnUseRecording.addEventListener('click', () => {
            if (this.enhancedState.recordedAudio) {
                this.state.audioFile = this.enhancedState.recordedAudio;
                this.showAudioPreview(this.enhancedState.recordedAudio);
                this.updateGenerateButton();
                UI.showToast('success', 'Audio Added', 'Recorded audio is ready for avatar generation.');

                // Hide recording info
                document.getElementById('recordingInfo').style.display = 'none';
            }
        });

        // Listen for audio level updates
        window.addEventListener('microphoneAudioLevel', (event) => {
            this.updateAudioLevelDisplay(event.detail.level);
        });

        // Update recording duration
        setInterval(() => {
            if (Microphone.state.isRecording) {
                const duration = Microphone.getRecordingDuration();
                this.updateRecordingDuration(duration);
            }
        }, 1000);
    },

    /**
     * Check microphone support
     */
    checkMicrophoneSupport() {
        const btnStartRecording = document.getElementById('btnStartRecording');
        if (!btnStartRecording) return;

        if (Microphone.isSupported()) {
            btnStartRecording.disabled = false;
            this.enhancedState.microphoneEnabled = true;
            this.logToConsole('info', 'Microphone support detected');
        } else {
            btnStartRecording.disabled = true;
            btnStartRecording.textContent = 'Microphone Not Supported';
            this.logToConsole('warning', 'Microphone not supported in this browser');
        }
    },

    /**
     * Handle recording started
     */
    onRecordingStarted() {
        // Show/hide buttons
        document.getElementById('btnStartRecording').style.display = 'none';
        document.getElementById('btnPauseRecording').style.display = 'inline-flex';
        document.getElementById('btnStopRecording').style.display = 'inline-flex';
        document.getElementById('btnCancelRecording').style.display = 'inline-flex';

        // Show visualizer
        document.getElementById('microphoneVisualizer').style.display = 'block';
        document.getElementById('recordingInfo').style.display = 'none';

        this.logToConsole('info', 'Recording started');
        this.startWaveformVisualization();
    },

    /**
     * Handle recording paused
     */
    onRecordingPaused() {
        document.getElementById('btnPauseRecording').style.display = 'none';
        document.getElementById('btnResumeRecording').style.display = 'inline-flex';
        this.logToConsole('info', 'Recording paused');
    },

    /**
     * Handle recording resumed
     */
    onRecordingResumed() {
        document.getElementById('btnResumeRecording').style.display = 'none';
        document.getElementById('btnPauseRecording').style.display = 'inline-flex';
        this.logToConsole('info', 'Recording resumed');
    },

    /**
     * Handle recording stopped
     * @param {File} file - Recorded audio file
     */
    onRecordingStopped(file) {
        // Hide/show buttons
        document.getElementById('btnPauseRecording').style.display = 'none';
        document.getElementById('btnResumeRecording').style.display = 'none';
        document.getElementById('btnStopRecording').style.display = 'none';
        document.getElementById('btnCancelRecording').style.display = 'none';
        document.getElementById('btnStartRecording').style.display = 'inline-flex';

        // Hide visualizer
        document.getElementById('microphoneVisualizer').style.display = 'none';

        if (file) {
            this.enhancedState.recordedAudio = file;

            // Show recording info
            const recordingInfo = document.getElementById('recordingInfo');
            recordingInfo.style.display = 'block';

            document.getElementById('recordingFileName').textContent = file.name;
            document.getElementById('recordingFileDuration').textContent =
                `${Microphone.getRecordingDuration()}s`;
            document.getElementById('recordingFileSize').textContent =
                UI.formatFileSize(file.size);

            this.logToConsole('info', `Recording completed: ${file.name}`);
            UI.showToast('success', 'Recording Complete', 'Audio recorded successfully!');
        }

        this.stopWaveformVisualization();
    },

    /**
     * Handle recording cancelled
     */
    onRecordingCancelled() {
        // Reset UI
        document.getElementById('btnPauseRecording').style.display = 'none';
        document.getElementById('btnResumeRecording').style.display = 'none';
        document.getElementById('btnStopRecording').style.display = 'none';
        document.getElementById('btnCancelRecording').style.display = 'none';
        document.getElementById('btnStartRecording').style.display = 'inline-flex';

        document.getElementById('microphoneVisualizer').style.display = 'none';
        document.getElementById('recordingInfo').style.display = 'none';

        this.enhancedState.recordedAudio = null;
        this.logToConsole('info', 'Recording cancelled');
        this.stopWaveformVisualization();
    },

    /**
     * Update audio level display
     * @param {number} level - Audio level (0-100)
     */
    updateAudioLevelDisplay(level) {
        const audioLevelFill = document.getElementById('audioLevelFill');
        const audioLevelText = document.getElementById('audioLevelText');

        if (audioLevelFill) {
            audioLevelFill.style.width = `${level}%`;
        }

        if (audioLevelText) {
            audioLevelText.textContent = `${Math.round(level)}%`;
        }
    },

    /**
     * Update recording duration display
     * @param {number} duration - Duration in seconds
     */
    updateRecordingDuration(duration) {
        const recordingDuration = document.getElementById('recordingDuration');
        if (recordingDuration) {
            const minutes = Math.floor(duration / 60);
            const seconds = duration % 60;
            recordingDuration.textContent =
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    },

    /**
     * Start waveform visualization
     */
    startWaveformVisualization() {
        const canvas = document.getElementById('waveformCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        const drawWaveform = () => {
            if (!Microphone.state.isRecording) return;

            const frequencyData = Microphone.getFrequencyData();

            ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
            ctx.fillRect(0, 0, width, height);

            const barWidth = (width / frequencyData.length) * 2.5;
            let x = 0;

            for (let i = 0; i < frequencyData.length; i++) {
                const barHeight = (frequencyData[i] / 255) * height;

                const gradient = ctx.createLinearGradient(0, height - barHeight, 0, height);
                gradient.addColorStop(0, '#667eea');
                gradient.addColorStop(0.5, '#764ba2');
                gradient.addColorStop(1, '#f093fb');

                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth, barHeight);

                x += barWidth + 1;
            }

            this.waveformAnimationId = requestAnimationFrame(drawWaveform);
        };

        drawWaveform();
    },

    /**
     * Stop waveform visualization
     */
    stopWaveformVisualization() {
        if (this.waveformAnimationId) {
            cancelAnimationFrame(this.waveformAnimationId);
            this.waveformAnimationId = null;
        }

        const canvas = document.getElementById('waveformCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    },

    /**
     * Setup processing visualizer UI
     */
    setupProcessingVisualizerUI() {
        // Processing visualizer is initialized automatically by ProcessingVisualizer module
        this.logToConsole('info', 'Processing visualizer initialized');
    },

    /**
     * Setup I/O comparison UI
     */
    setupIOComparisonUI() {
        // Will be populated when generation completes
        this.logToConsole('info', 'I/O comparison UI initialized');
    },

    /**
     * Setup error console
     */
    setupErrorConsole() {
        const btnClearConsole = document.getElementById('btnClearConsole');
        const btnExportLogs = document.getElementById('btnExportLogs');
        const toggleDebug = document.getElementById('filterDebug');

        if (btnClearConsole) {
            btnClearConsole.addEventListener('click', () => {
                this.clearConsole();
            });
        }

        if (btnExportLogs) {
            btnExportLogs.addEventListener('click', () => {
                this.exportLogs();
            });
        }

        if (toggleDebug) {
            toggleDebug.addEventListener('change', (e) => {
                this.enhancedState.debugMode = e.target.checked;
                this.logToConsole('info', `Debug mode ${e.target.checked ? 'enabled' : 'disabled'}`);
            });
        }

        // Setup console filters
        ['filterErrors', 'filterWarnings', 'filterInfo', 'filterDebug'].forEach(filterId => {
            const filter = document.getElementById(filterId);
            if (filter) {
                filter.addEventListener('change', () => {
                    this.applyConsoleFilters();
                });
            }
        });

        // Intercept console methods
        this.interceptConsole();

        this.logToConsole('info', 'Error console initialized');
    },

    /**
     * Log message to console
     * @param {string} type - Log type (error, warning, info, debug)
     * @param {string} message - Log message
     */
    logToConsole(type, message) {
        const timestamp = new Date().toISOString();
        const log = { type, message, timestamp };

        this.enhancedState.consoleLogs.push(log);

        // Display in console
        const consoleOutput = document.getElementById('consoleOutput');
        if (consoleOutput) {
            const entry = document.createElement('div');
            entry.className = `console-entry ${type}`;
            entry.innerHTML = `
                <span class="console-timestamp">${new Date(timestamp).toLocaleTimeString()}</span>
                <span class="console-type">[${type.toUpperCase()}]</span>
                <span class="console-message">${message}</span>
            `;
            consoleOutput.appendChild(entry);
            consoleOutput.scrollTop = consoleOutput.scrollHeight;

            // Show error console section if error or warning
            if (type === 'error' || type === 'warning') {
                document.getElementById('errorConsoleSection').style.display = 'block';
            }
        }
    },

    /**
     * Clear console
     */
    clearConsole() {
        this.enhancedState.consoleLogs = [];
        const consoleOutput = document.getElementById('consoleOutput');
        if (consoleOutput) {
            consoleOutput.innerHTML = '';
        }
        this.logToConsole('info', 'Console cleared');
    },

    /**
     * Export logs
     */
    exportLogs() {
        const logs = this.enhancedState.consoleLogs.map(log =>
            `${log.timestamp} [${log.type.toUpperCase()}] ${log.message}`
        ).join('\n');

        const blob = new Blob([logs], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `avatar-system-logs-${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);

        this.logToConsole('info', 'Logs exported');
    },

    /**
     * Apply console filters
     */
    applyConsoleFilters() {
        const filters = {
            error: document.getElementById('filterErrors')?.checked,
            warning: document.getElementById('filterWarnings')?.checked,
            info: document.getElementById('filterInfo')?.checked,
            debug: document.getElementById('filterDebug')?.checked
        };

        const entries = document.querySelectorAll('.console-entry');
        entries.forEach(entry => {
            const type = Array.from(entry.classList).find(c =>
                ['error', 'warning', 'info', 'debug'].includes(c)
            );
            entry.style.display = filters[type] ? 'block' : 'none';
        });
    },

    /**
     * Intercept browser console for logging
     */
    interceptConsole() {
        const originalConsole = {
            log: console.log,
            warn: console.warn,
            error: console.error
        };

        console.log = (...args) => {
            originalConsole.log(...args);
            if (this.enhancedState.debugMode) {
                this.logToConsole('debug', args.join(' '));
            }
        };

        console.warn = (...args) => {
            originalConsole.warn(...args);
            this.logToConsole('warning', args.join(' '));
        };

        console.error = (...args) => {
            originalConsole.error(...args);
            this.logToConsole('error', args.join(' '));
        };
    },

    /**
     * Enhanced start generation with processing visualization
     */
    async startGenerationEnhanced() {
        // Call original startGeneration
        await this.startGeneration();

        // Show processing visualizer
        ProcessingVisualizer.start(this.state.taskId || 'task-' + Date.now());
        this.enhancedState.processingActive = true;
    },

    /**
     * Show I/O comparison after generation
     * @param {Object} result - Generation result
     */
    showIOComparison(result) {
        const ioSection = document.getElementById('ioComparisonSection');
        if (!ioSection) return;

        ioSection.style.display = 'block';

        // Input image
        const ioInputImage = document.getElementById('ioInputImage');
        if (ioInputImage && this.state.imageFile) {
            const img = ioInputImage.querySelector('img');
            img.src = URL.createObjectURL(this.state.imageFile);
        }

        // Input audio
        const ioInputAudio = document.getElementById('ioInputAudio');
        if (ioInputAudio && this.state.audioFile) {
            ioInputAudio.src = URL.createObjectURL(this.state.audioFile);
        }

        // Output video
        const ioOutputVideo = document.getElementById('ioOutputVideo');
        if (ioOutputVideo && result.video_path) {
            ioOutputVideo.src = result.video_path;
        }

        // Emotion
        if (result.emotion) {
            document.getElementById('ioEmotionPrimary').textContent = result.emotion;
        }
        if (result.confidence) {
            document.getElementById('ioEmotionConfidence').textContent =
                `${(result.confidence * 100).toFixed(1)}% Confidence`;
        }

        // Avatar controls
        if (result.avatar_control) {
            const controlsDisplay = document.getElementById('ioAvatarControls');
            controlsDisplay.innerHTML = '';
            Object.entries(result.avatar_control).forEach(([key, value]) => {
                const item = document.createElement('div');
                item.className = 'control-item';
                item.innerHTML = `
                    <span>${key.replace('_', ' ').toUpperCase()}</span>
                    <span>${(value * 100).toFixed(1)}%</span>
                `;
                controlsDisplay.appendChild(item);
            });
        }

        // Performance
        if (result.performance) {
            const perfDisplay = document.getElementById('ioPerformance');
            perfDisplay.innerHTML = '';
            Object.entries(result.performance).forEach(([key, value]) => {
                const item = document.createElement('div');
                item.className = 'perf-item';
                item.innerHTML = `
                    <span>${key.replace('_', ' ').toUpperCase()}</span>
                    <span>${value.toFixed(2)}s</span>
                `;
                perfDisplay.appendChild(item);
            });
        }

        this.logToConsole('info', 'I/O comparison displayed');
        UI.scrollToElement('ioComparisonSection');
    }
};

// Initialize enhanced features when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait for base App to initialize first
        setTimeout(() => EnhancedApp.initEnhanced(), 100);
    });
} else {
    setTimeout(() => EnhancedApp.initEnhanced(), 100);
}

// Export for debugging
window.EnhancedApp = EnhancedApp;
