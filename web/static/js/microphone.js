// ============================================================================
// Microphone Module - Real-time Audio Recording
// ============================================================================

const Microphone = {
    // State
    state: {
        isRecording: false,
        isPaused: false,
        mediaRecorder: null,
        audioStream: null,
        audioChunks: [],
        startTime: null,
        pausedDuration: 0,
        lastPauseTime: null
    },

    // Configuration
    config: {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 128000,
        maxDuration: 300000, // 5 minutes
        visualizerFFTSize: 2048,
        visualizerUpdateInterval: 50 // ms
    },

    // Audio context and analyzer
    audioContext: null,
    analyser: null,
    dataArray: null,
    timeDataArray: null,
    animationId: null,

    // Frequency analysis
    frequencyBands: {
        bass: { min: 0, max: 250 },      // 0-250 Hz
        mid: { min: 250, max: 2000 },    // 250-2000 Hz
        treble: { min: 2000, max: 8000 } // 2000-8000 Hz
    },

    /**
     * Check microphone support
     * @returns {boolean}
     */
    isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    },

    /**
     * Request microphone permission and start recording
     * @returns {Promise<void>}
     */
    async startRecording() {
        if (!this.isSupported()) {
            throw new Error('Microphone not supported in this browser');
        }

        if (this.state.isRecording) {
            console.warn('[Microphone] Already recording');
            return;
        }

        try {
            // Request microphone access
            console.log('[Microphone] Requesting microphone access...');
            this.state.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 48000,
                    channelCount: 1
                }
            });

            // Create audio context for visualization
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(this.state.audioStream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.config.visualizerFFTSize;
            this.analyser.smoothingTimeConstant = 0.8;
            source.connect(this.analyser);

            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.timeDataArray = new Uint8Array(this.analyser.fftSize);

            // Determine MIME type support
            let mimeType = this.config.mimeType;
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = ''; // Browser will choose
                }
            }

            // Create media recorder
            const options = {
                audioBitsPerSecond: this.config.audioBitsPerSecond
            };
            if (mimeType) {
                options.mimeType = mimeType;
            }

            this.state.mediaRecorder = new MediaRecorder(this.state.audioStream, options);

            // Setup event handlers
            this.state.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.state.audioChunks.push(event.data);
                }
            };

            this.state.mediaRecorder.onstop = () => {
                this.handleRecordingStop();
            };

            this.state.mediaRecorder.onerror = (event) => {
                console.error('[Microphone] Recording error:', event.error);
                this.stopRecording();
            };

            // Start recording
            this.state.mediaRecorder.start(100); // Collect data every 100ms
            this.state.isRecording = true;
            this.state.isPaused = false;
            this.state.startTime = Date.now();
            this.state.audioChunks = [];
            this.state.pausedDuration = 0;

            console.log('[Microphone] Recording started');

            // Start visualization
            this.startVisualization();

            // Auto-stop after max duration
            setTimeout(() => {
                if (this.state.isRecording) {
                    console.log('[Microphone] Max duration reached, stopping...');
                    this.stopRecording();
                }
            }, this.config.maxDuration);

        } catch (error) {
            console.error('[Microphone] Failed to start recording:', error);
            this.cleanup();
            throw error;
        }
    },

    /**
     * Pause recording
     */
    pauseRecording() {
        if (!this.state.isRecording || this.state.isPaused) {
            return;
        }

        this.state.mediaRecorder.pause();
        this.state.isPaused = true;
        this.state.lastPauseTime = Date.now();
        console.log('[Microphone] Recording paused');
    },

    /**
     * Resume recording
     */
    resumeRecording() {
        if (!this.state.isRecording || !this.state.isPaused) {
            return;
        }

        this.state.mediaRecorder.resume();
        this.state.isPaused = false;
        if (this.state.lastPauseTime) {
            this.state.pausedDuration += Date.now() - this.state.lastPauseTime;
        }
        console.log('[Microphone] Recording resumed');
    },

    /**
     * Stop recording
     * @returns {Promise<File>}
     */
    async stopRecording() {
        if (!this.state.isRecording) {
            return null;
        }

        return new Promise((resolve) => {
            this.state.mediaRecorder.addEventListener('stop', () => {
                resolve(this.state.recordedFile);
            }, { once: true });

            this.state.mediaRecorder.stop();
            this.state.isRecording = false;
            this.stopVisualization();

            // Stop all tracks
            if (this.state.audioStream) {
                this.state.audioStream.getTracks().forEach(track => track.stop());
            }

            console.log('[Microphone] Recording stopped');
        });
    },

    /**
     * Handle recording stop event
     */
    handleRecordingStop() {
        const blob = new Blob(this.state.audioChunks, {
            type: this.state.mediaRecorder.mimeType || 'audio/webm'
        });

        const duration = this.getRecordingDuration();
        const filename = `recording_${Date.now()}.webm`;

        this.state.recordedFile = new File([blob], filename, {
            type: blob.type,
            lastModified: Date.now()
        });

        console.log('[Microphone] Recording processed:', {
            size: blob.size,
            duration: duration,
            filename: filename
        });

        this.cleanup();
    },

    /**
     * Get recording duration in seconds
     * @returns {number}
     */
    getRecordingDuration() {
        if (!this.state.startTime) return 0;
        const endTime = Date.now();
        return Math.floor((endTime - this.state.startTime - this.state.pausedDuration) / 1000);
    },

    /**
     * Get current audio level (0-100)
     * @returns {number}
     */
    getAudioLevel() {
        if (!this.analyser || !this.dataArray) return 0;

        this.analyser.getByteFrequencyData(this.dataArray);

        const average = this.dataArray.reduce((a, b) => a + b, 0) / this.dataArray.length;
        return Math.min(100, Math.floor((average / 255) * 100));
    },

    /**
     * Get frequency data for visualization
     * @returns {Uint8Array}
     */
    getFrequencyData() {
        if (!this.analyser || !this.dataArray) return new Uint8Array(0);

        this.analyser.getByteFrequencyData(this.dataArray);
        return this.dataArray;
    },

    /**
     * Get time domain data for waveform
     * @returns {Uint8Array}
     */
    getTimeDomainData() {
        if (!this.analyser || !this.timeDataArray) return new Uint8Array(0);

        this.analyser.getByteTimeDomainData(this.timeDataArray);
        return this.timeDataArray;
    },

    /**
     * Get frequency band levels (bass, mid, treble)
     * @returns {Object}
     */
    getFrequencyBands() {
        if (!this.analyser || !this.dataArray) {
            return { bass: 0, mid: 0, treble: 0 };
        }

        this.analyser.getByteFrequencyData(this.dataArray);

        const nyquist = this.audioContext.sampleRate / 2;
        const frequencyPerBin = nyquist / this.dataArray.length;

        const getBandLevel = (minFreq, maxFreq) => {
            const minBin = Math.floor(minFreq / frequencyPerBin);
            const maxBin = Math.floor(maxFreq / frequencyPerBin);

            let sum = 0;
            let count = 0;
            for (let i = minBin; i <= maxBin && i < this.dataArray.length; i++) {
                sum += this.dataArray[i];
                count++;
            }

            return count > 0 ? Math.floor((sum / count / 255) * 100) : 0;
        };

        return {
            bass: getBandLevel(this.frequencyBands.bass.min, this.frequencyBands.bass.max),
            mid: getBandLevel(this.frequencyBands.mid.min, this.frequencyBands.mid.max),
            treble: getBandLevel(this.frequencyBands.treble.min, this.frequencyBands.treble.max)
        };
    },

    /**
     * Get audio energy level in dB
     * @returns {number}
     */
    getAudioEnergy() {
        if (!this.analyser || !this.dataArray) return -100;

        this.analyser.getByteFrequencyData(this.dataArray);

        const sum = this.dataArray.reduce((a, b) => a + b, 0);
        const average = sum / this.dataArray.length;

        // Convert to dB (approximate)
        const db = 20 * Math.log10(average / 255);
        return Math.max(-60, Math.min(0, db));
    },

    /**
     * Detect dominant frequency (pitch)
     * @returns {number} frequency in Hz
     */
    getDominantFrequency() {
        if (!this.analyser || !this.dataArray) return 0;

        this.analyser.getByteFrequencyData(this.dataArray);

        let maxIndex = 0;
        let maxValue = 0;

        for (let i = 0; i < this.dataArray.length; i++) {
            if (this.dataArray[i] > maxValue) {
                maxValue = this.dataArray[i];
                maxIndex = i;
            }
        }

        const nyquist = this.audioContext.sampleRate / 2;
        const frequencyPerBin = nyquist / this.dataArray.length;

        return Math.floor(maxIndex * frequencyPerBin);
    },

    /**
     * Get comprehensive audio metrics
     * @returns {Object}
     */
    getAudioMetrics() {
        return {
            level: this.getAudioLevel(),
            energy: this.getAudioEnergy(),
            pitch: this.getDominantFrequency(),
            bands: this.getFrequencyBands(),
            frequencyData: this.getFrequencyData(),
            waveformData: this.getTimeDomainData()
        };
    },

    /**
     * Start audio visualization
     */
    startVisualization() {
        const updateVisualizer = () => {
            if (!this.state.isRecording) return;

            const metrics = this.getAudioMetrics();

            // Dispatch custom event with comprehensive audio metrics
            window.dispatchEvent(new CustomEvent('microphoneAudioLevel', {
                detail: metrics
            }));

            this.animationId = requestAnimationFrame(updateVisualizer);
        };

        updateVisualizer();
    },

    /**
     * Stop audio visualization
     */
    stopVisualization() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    },

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopVisualization();

        if (this.state.audioStream) {
            this.state.audioStream.getTracks().forEach(track => track.stop());
            this.state.audioStream = null;
        }

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.analyser = null;
        this.dataArray = null;
        this.state.audioChunks = [];
    },

    /**
     * Cancel recording (discard audio)
     */
    cancelRecording() {
        if (this.state.isRecording) {
            this.state.mediaRecorder.stop();
            this.state.isRecording = false;
        }

        this.state.audioChunks = [];
        this.state.recordedFile = null;
        this.cleanup();

        console.log('[Microphone] Recording cancelled');
    }
};

// Export for use in other modules
window.Microphone = Microphone;
