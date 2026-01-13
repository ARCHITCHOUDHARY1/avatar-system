// ============================================================================
// Processing Visualizer - Real-time Pipeline Progress & Metrics
// ============================================================================

const ProcessingVisualizer = {
    // State
    state: {
        startTime: null,
        currentStage: null,
        stageStartTime: null,
        stageTimes: {},
        totalStages: 5,
        completedStages: 0,
        currentProgress: 0,
        isProcessing: false,
        taskId: null,
        pollingInterval: null
    },

    // Pipeline stages configuration
    stages: [
        {
            id: 'audio_processing',
            name: 'Audio Processing',
            description: 'Extracting audio features and transcription',
            icon: '🔊',
            expectedDuration: 3000 // ms
        },
        {
            id: 'emotion_detection',
            name: 'Emotion Detection',
            description: 'Analyzing emotional content',
            icon: '😊',
            expectedDuration: 2000
        },
        {
            id: 'mistral_controller',
            name: 'Mistral AI Controller',
            description: 'Generating avatar control parameters',
            icon: '🧠',
            expectedDuration: 1500
        },
        {
            id: 'video_generation',
            name: 'Video Generation',
            description: 'Rendering talking avatar video',
            icon: '🎬',
            expectedDuration: 45000
        },
        {
            id: 'quality_enhancement',
            name: 'Quality Enhancement',
            description: 'Enhancing video with GFPGAN',
            icon: '✨',
            expectedDuration: 12000
        }
    ],

    // DOM elements
    elements: null,

    /**
     * Initialize visualizer
     */
    init() {
        this.cacheElements();
        this.createStageCards();
        console.log('[ProcessingVisualizer] Initialized');
    },

    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            container: document.getElementById('processingVisualizer'),
            stagesContainer: document.getElementById('pipelineStages'),
            timelineContainer: document.getElementById('processingTimeline'),
            metricsContainer: document.getElementById('processingMetrics'),
            overallProgress: document.getElementById('overallProgress'),
            progressBar: document.getElementById('overallProgressBar'),
            progressText: document.getElementById('overallProgressText'),
            elapsedTime: document.getElementById('elapsedTime'),
            estimatedTime: document.getElementById('estimatedTime'),
            currentStage: document.getElementById('currentStageName'),
            stageDescription: document.getElementById('currentStageDescription')
        };
    },

    /**
     * Create stage progress cards
     */
    createStageCards() {
        if (!this.elements.stagesContainer) return;

        this.elements.stagesContainer.innerHTML = '';

        this.stages.forEach((stage, index) => {
            const card = document.createElement('div');
            card.className = 'stage-card';
            card.id = `stage-${stage.id}`;
            card.setAttribute('data-stage-id', stage.id');

            card.innerHTML = `
                <div class="stage-header">
                    <div class="stage-icon">${stage.icon}</div>
                    <div class="stage-info">
                        <div class="stage-name">${stage.name}</div>
                        <div class="stage-description">${stage.description}</div>
                    </div>
                    <div class="stage-status">
                        <div class="status-icon pending">⏳</div>
                    </div>
                </div>
                <div class="stage-progress">
                    <div class="progress-bar-mini">
                        <div class="progress-fill" style="width: 0%"></div>
                    </div>
                    <div class="stage-time">--</div>
                </div>
            `;

            this.elements.stagesContainer.appendChild(card);
        });
    },

    /**
     * Start processing visualization
     * @param {string} taskId - Task ID for polling
     */
    start(taskId) {
        this.state.taskId = taskId;
        this.state.startTime = Date.now();
        this.state.isProcessing = true;
        this.state.completedStages = 0;
        this.state.currentProgress = 0;
        this.state.stageTimes = {};

        this.show();
        this.updateOverallProgress(0);
        this.startPolling();

        console.log('[ProcessingVisualizer] Started for task:', taskId);
    },

    /**
     * Start polling for status updates
     */
    startPolling() {
        if (this.state.pollingInterval) {
            clearInterval(this.state.pollingInterval);
        }

        this.state.pollingInterval = setInterval(() => {
            this.pollStatus();
        }, 2000); // Poll every 2 seconds

        // Initial poll
        this.pollStatus();
    },

    /**
     * Poll task status from API
     */
    async pollStatus() {
        if (!this.state.taskId) return;

        try {
            const status = await API.getStatus(this.state.taskId);
            this.updateFromStatus(status);

            if (status.status === 'completed' || status.status === 'failed') {
                this.stopPolling();
                if (status.status === 'completed') {
                    this.complete(status);
                } else {
                    this.fail(status.error || 'Unknown error');
                }
            }
        } catch (error) {
            console.error('[ProcessingVisualizer] Poll error:', error);
        }
    },

    /**
     * Update visualization from status object
     * @param {Object} status - Status object from API
     */
    updateFromStatus(status) {
        if (status.stage) {
            const stageIndex = this.stages.findIndex(s => s.id === status.stage);
            if (stageIndex !== -1) {
                this.setStage(status.stage);
            }
        }

        if (status.progress !== undefined) {
            this.updateOverallProgress(status.progress);
        }

        if (status.stage_times) {
            this.state.stageTimes = status.stage_times;
            this.updateStageTimings();
        }
    },

    /**
     * Set current processing stage
     * @param {string} stageId - Stage ID
     */
    setStage(stageId) {
        const stage = this.stages.find(s => s.id === stageId);
        if (!stage) return;

        // Complete previous stages
        if (this.state.currentStage) {
            this.completeStage(this.state.currentStage);
        }

        this.state.currentStage = stageId;
        this.state.stageStartTime = Date.now();

        // Update current stage display
        if (this.elements.currentStage) {
            this.elements.currentStage.textContent = stage.name;
        }
        if (this.elements.stageDescription) {
            this.elements.stageDescription.textContent = stage.description;
        }

        // Update stage card
        const card = document.getElementById(`stage-${stageId}`);
        if (card) {
            // Remove pending status from all cards
            document.querySelectorAll('.stage-card').forEach(c => {
                c.classList.remove('active');
            });

            card.classList.add('active');
            const statusIcon = card.querySelector('.status-icon');
            if (statusIcon) {
                statusIcon.className = 'status-icon processing';
                statusIcon.textContent = '⚙️';
            }
        }

        console.log('[ProcessingVisualizer] Stage:', stage.name);
    },

    /**
     * Complete a stage
     * @param {string} stageId - Stage ID
     */
    completeStage(stageId) {
        const duration = Date.now() - (this.state.stageStartTime || Date.now());
        this.state.stageTimes[stageId] = duration;

        const card = document.getElementById(`stage-${stageId}`);
        if (card) {
            card.classList.remove('active');
            card.classList.add('completed');

            const statusIcon = card.querySelector('.status-icon');
            if (statusIcon) {
                statusIcon.className = 'status-icon completed';
                statusIcon.textContent = '✅';
            }

            const progressFill = card.querySelector('.progress-fill');
            if (progressFill) {
                progressFill.style.width = '100%';
            }

            const timeDisplay = card.querySelector('.stage-time');
            if (timeDisplay) {
                timeDisplay.textContent = this.formatDuration(duration);
            }
        }

        this.state.completedStages++;
    },

    /**
     * Update overall progress
     * @param {number} progress - Progress percentage (0-100)
     */
    updateOverallProgress(progress) {
        this.state.currentProgress = progress;

        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = `${progress}%`;
        }

        if (this.elements.progressText) {
            this.elements.progressText.textContent = `${Math.round(progress)}%`;
        }

        // Update elapsed time
        if (this.state.startTime && this.elements.elapsedTime) {
            const elapsed = Date.now() - this.state.startTime;
            this.elements.elapsedTime.textContent = this.formatDuration(elapsed);
        }

        // Estimate remaining time
        if (this.state.startTime && this.elements.estimatedTime && progress > 0) {
            const elapsed = Date.now() - this.state.startTime;
            const estimated = (elapsed / progress) * (100 - progress);
            this.elements.estimatedTime.textContent = this.formatDuration(estimated);
        }
    },

    /**
     * Update stage timing displays
     */
    updateStageTimings() {
        Object.entries(this.state.stageTimes).forEach(([stageId, duration]) => {
            const card = document.getElementById(`stage-${stageId}`);
            if (card) {
                const timeDisplay = card.querySelector('.stage-time');
                if (timeDisplay) {
                    timeDisplay.textContent = this.formatDuration(duration);
                }
            }
        });
    },

    /**
     * Complete processing
     * @param {Object} result - Final result object
     */
    complete(result) {
        this.state.isProcessing = false;

        // Complete current stage
        if (this.state.currentStage) {
            this.completeStage(this.state.currentStage);
        }

        // Mark any remaining stages as completed
        this.stages.forEach(stage => {
            const card = document.getElementById(`stage-${stage.id}`);
            if (card && !card.classList.contains('completed')) {
                this.completeStage(stage.id);
            }
        });

        this.updateOverallProgress(100);

        // Show completion metrics
        this.showMetrics(result);

        console.log('[ProcessingVisualizer] Completed:', result);
    },

    /**
     * Show processing metrics
     * @param {Object} result - Result with metrics
     */
    showMetrics(result) {
        if (!this.elements.metricsContainer) return;

        const totalTime = Date.now() - this.state.startTime;
        const metrics = result.performance || this.state.stageTimes;

        let metricsHTML = `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-icon">⏱️</div>
                    <div class="metric-content">
                        <div class="metric-label">Total Time</div>
                        <div class="metric-value">${this.formatDuration(totalTime)}</div>
                    </div>
                </div>
        `;

        if (result.emotion) {
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-icon">😊</div>
                    <div class="metric-content">
                        <div class="metric-label">Detected Emotion</div>
                        <div class="metric-value">${result.emotion}</div>
                    </div>
                </div>
            `;
        }

        if (result.confidence) {
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-content">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">${(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `;
        }

        if (result.avatar_control) {
            const avgControl = Object.values(result.avatar_control).reduce((a, b) => a + b, 0) /
                Object.keys(result.avatar_control).length;
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-icon">🎮</div>
                    <div class="metric-content">
                        <div class="metric-label">Control Params</div>
                        <div class="metric-value">${Object.keys(result.avatar_control).length} params</div>
                    </div>
                </div>
            `;
        }

        metricsHTML += '</div>';
        this.elements.metricsContainer.innerHTML = metricsHTML;
    },

    /**
     * Handle processing failure
     * @param {string} error - Error message
     */
    fail(error) {
        this.state.isProcessing = false;

        const card = document.getElementById(`stage-${this.state.currentStage}`);
        if (card) {
            card.classList.add('failed');
            const statusIcon = card.querySelector('.status-icon');
            if (statusIcon) {
                statusIcon.className = 'status-icon failed';
                statusIcon.textContent = '❌';
            }
        }

        console.error('[ProcessingVisualizer] Failed:', error);
    },

    /**
     * Stop polling
     */
    stopPolling() {
        if (this.state.pollingInterval) {
            clearInterval(this.state.pollingInterval);
            this.state.pollingInterval = null;
        }
    },

    /**
     * Show visualizer
     */
    show() {
        if (this.elements.container) {
            this.elements.container.style.display = 'block';
        }
    },

    /**
     * Hide visualizer
     */
    hide() {
        if (this.elements.container) {
            this.elements.container.style.display = 'none';
        }
        this.stopPolling();
    },

    /**
     * Reset visualizer
     */
    reset() {
        this.state = {
            startTime: null,
            currentStage: null,
            stageStartTime: null,
            stageTimes: {},
            totalStages: 5,
            completedStages: 0,
            currentProgress: 0,
            isProcessing: false,
            taskId: null,
            pollingInterval: null
        };

        this.stopPolling();
        this.createStageCards();
        this.updateOverallProgress(0);

        if (this.elements.metricsContainer) {
            this.elements.metricsContainer.innerHTML = '';
        }

        console.log('[ProcessingVisualizer] Reset');
    },

    /**
     * Format duration in milliseconds to readable string
     * @param {number} ms - Duration in milliseconds
     * @returns {string}
     */
    formatDuration(ms) {
        if (ms < 1000) return `${Math.round(ms)}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        const minutes = Math.floor(ms / 60000);
        const seconds = Math.floor((ms % 60000) / 1000);
        return `${minutes}m ${seconds}s`;
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ProcessingVisualizer.init());
} else {
    ProcessingVisualizer.init();
}

// Export for use in other modules
window.ProcessingVisualizer = ProcessingVisualizer;
