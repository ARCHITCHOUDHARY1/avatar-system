/**
 * System Settings Panel
 * Real-time system monitoring and configuration
 */

const SystemSettings = {
    state: {
        isOpen: false,
        systemStatus: null,
        config: null,
        updateInterval: null,
        refreshRate: 3000 // 3 seconds
    },

    elements: {},

    /**
     * Initialize system settings panel
     */
    init() {
        console.log('[SystemSettings] Initializing...');

        this.createElements();
        this.bindEvents();
        this.loadSettings();

        console.log('[SystemSettings] Initialized successfully');
    },

    /**
     * Create UI elements
     */
    createElements() {
        // Create settings button in header
        const headerActions = document.querySelector('.header-actions');
        if (headerActions) {
            const settingsBtn = document.createElement('button');
            settingsBtn.className = 'btn-icon';
            settingsBtn.id = 'settingsBtn';
            settingsBtn.setAttribute('aria-label', 'System Settings');
            settingsBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M11.49 3.17c0-.38-.31-.69-.69-.69H9.2c-.38 0-.69.31-.69.69v1.09c0 .38.31.69.69.69h1.6c.38 0 .69-.31.69-.69V3.17zm-3.02 11.69c0 .38.31.69.69.69h1.6c.38 0 .69-.31.69-.69v-1.09c0-.38-.31-.69-.69-.69H9.2c-.38 0-.69.31-.69.69v1.09zm6.27-7.69c.27-.27.27-.71 0-.98l-1.13-1.13c-.27-.27-.71-.27-.98 0l-.77.77c-.27.27-.27.71 0 .98l1.13 1.13c.27.27.71.27.98 0l.77-.77zm-7.62 4.27c-.27.27-.27.71 0-.98l1.13 1.13c.27.27.71.27.98 0l.77-.77c.27-.27.27-.71 0-.98l-1.13-1.13c-.27-.27-.71-.27-.98 0l-.77.77zM17 9.2c-.38 0-.69.31-.69.69v1.6c0 .38.31.69.69.69h1.09c.38 0 .69-.31.69-.69V9.89c0-.38-.31-.69-.69-.69H17zm-14.91 0c-.38 0-.69.31-.69.69v1.6c0 .38.31.69.69.69H3.2c.38 0 .69-.31.69-.69V9.89c0-.38-.31-.69-.69-.69H2.09zm12.32-4.61c-.27-.27-.71-.27-.98 0l-.77.77c-.27.27-.27.71 0 .98l1.13 1.13c.27.27.71.27.98 0l.77-.77c.27-.27.27-.71 0-.98l-1.13-1.13zM6.27 14.41c-.27-.27-.71-.27-.98 0l-.77.77c-.27.27-.27.71 0 .98l1.13 1.13c.27.27.71.27.98 0l.77-.77c.27-.27.27-.71 0-.98l-1.13-1.13z"/>
                </svg>
            `;
            headerActions.insertBefore(settingsBtn, headerActions.firstChild);
            this.elements.settingsBtn = settingsBtn;
        }

        // Create settings modal
        const modal = document.createElement('div');
        modal.className = 'settings-modal';
        modal.id = 'settingsModal';
        modal.style.display = 'none';
        modal.innerHTML = `
            <div class="settings-overlay"></div>
            <div class="settings-panel glass-panel">
                <div class="settings-header">
                    <h2>System Settings</h2>
                    <button class="btn-close" id="closeSettings" aria-label="Close">×</button>
                </div>
                
                <div class="settings-content">
                    <!-- System Status Section -->
                    <section class="settings-section">
                        <h3>🖥️ System Status</h3>
                        <div class="status-grid">
                            <div class="status-item">
                                <span class="status-label">Backend Status</span>
                                <span class="status-value" id="backendStatus">
                                    <span class="status-dot connecting"></span>
                                    Connecting...
                                </span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">Device</span>
                                <span class="status-value" id="deviceStatus">--</span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">GPU Memory</span>
                                <span class="status-value" id="gpuMemory">-- / -- GB</span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">CPU Usage</span>
                                <span class="status-value">
                                    <span id="cpuUsage">--</span>%
                                    <meter id="cpuMeter" min="0" max="100" value="0"></meter>
                                </span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">RAM Usage</span>
                                <span class="status-value">
                                    <span id="ramUsage">--</span>%
                                    <meter id="ramMeter" min="0" max="100" value="0"></meter>
                                </span>
                            </div>
                        </div>
                    </section>

                    <!-- Models Status Section -->
                    <section class="settings-section">
                        <h3>🧠 Models Status</h3>
                        <div class="models-list" id="modelsList">
                            <div class="loading-indicator">Loading model status...</div>
                        </div>
                    </section>

                    <!-- Configuration Section -->
                    <section class="settings-section">
                        <h3>⚙️ Configuration</h3>
                        <div class="config-grid">
                            <div class="config-item">
                                <label for="deviceSelect">Processing Device</label>
                                <select id="deviceSelect" class="config-select">
                                    <option value="auto">Auto-detect</option>
                                    <option value="cuda">CUDA (GPU)</option>
                                    <option value="cpu">CPU</option>
                                </select>
                            </div>
                            <div class="config-item">
                                <label for="precisionSelect">Precision</label>
                                <select id="precisionSelect" class="config-select">
                                    <option value="fp16">FP16 (Fast)</option>
                                    <option value="fp32">FP32 (Accurate)</option>
                                </select>
                            </div>
                            <div class="config-item">
                                <label for="qualityPreset">Quality Preset</label>
                                <select id="qualityPreset" class="config-select">
                                    <option value="fast">Fast (Lower Quality)</option>
                                    <option value="balanced">Balanced</option>
                                    <option value="quality">Quality (Slower)</option>
                                </select>
                            </div>
                            <div class="config-item">
                                <label>
                                    <input type="checkbox" id="autoRefresh" checked>
                                    Auto-refresh status
                                </label>
                            </div>
                        </div>
                        <button class="btn-primary" id="saveSettings">Save Settings</button>
                    </section>

                    <!-- About Section -->
                    <section class="settings-section">
                        <h3>ℹ️ About</h3>
                        <p><strong>Avatar System Orchestrator</strong></p>
                        <p>Version: 1.0.0</p>
                        <p>LangGraph-based AI Avatar Generation</p>
                        <p style="margin-top: 1rem; font-size: 0.875rem; opacity: 0.7;">
                            Powered by SadTalker, Mistral AI, and GFPGAN
                        </p>
                    </section>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Cache element references
        this.elements.modal = modal;
        this.elements.closeBtn = modal.querySelector('#closeSettings');
        this.elements.overlay = modal.querySelector('.settings-overlay');
        this.elements.saveBtn = modal.querySelector('#saveSettings');
        this.elements.deviceSelect = modal.querySelector('#deviceSelect');
        this.elements.precisionSelect = modal.querySelector('#precisionSelect');
        this.elements.qualityPreset = modal.querySelector('#qualityPreset');
        this.elements.autoRefresh = modal.querySelector('#autoRefresh');
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Open settings
        if (this.elements.settingsBtn) {
            this.elements.settingsBtn.addEventListener('click', () => this.show());
        }

        // Close settings
        this.elements.closeBtn.addEventListener('click', () => this.hide());
        this.elements.overlay.addEventListener('click', () => this.hide());

        // Save settings
        this.elements.saveBtn.addEventListener('click', () => this.saveSettings());

        // Auto-refresh toggle
        this.elements.autoRefresh.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        });

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.state.isOpen) {
                this.hide();
            }
        });
    },

    /**
     * Show settings panel
     */
    show() {
        this.elements.modal.style.display = 'flex';
        // Force reflow
        this.elements.modal.offsetHeight;
        this.elements.modal.classList.add('active');
        this.state.isOpen = true;

        // Fetch current status
        this.fetchSystemStatus();

        // Start auto-refresh if enabled
        if (this.elements.autoRefresh.checked) {
            this.startAutoRefresh();
        }
    },

    /**
     * Hide settings panel
     */
    hide() {
        this.elements.modal.classList.remove('active');
        setTimeout(() => {
            if (!this.state.isOpen) {
                this.elements.modal.style.display = 'none';
            }
        }, 300); // Wait for transition
        this.state.isOpen = false;
        this.stopAutoRefresh();
    },

    /**
     * Fetch system status from backend
     */
    async fetchSystemStatus() {
        try {
            const response = await fetch('/api/v1/system/status');

            if (response.ok) {
                const status = await response.json();
                this.updateSystemStatus(status);
            } else {
                console.debug('[SystemSettings] Status endpoint currently check unavailable');
                // Could retry or show limited status
                this.updateSystemStatus(this.getFallbackStatus());
            }
        } catch (error) {
            console.debug('[SystemSettings] Backend connection retry...');
            this.updateSystemStatus(this.getFallbackStatus());
        }
    },

    /**
     * Update UI with system status
     */
    updateSystemStatus(status) {
        this.state.systemStatus = status;

        // Update backend status
        const backendStatus = document.getElementById('backendStatus');
        if (backendStatus) {
            const isReady = status.backend_status === 'ready';
            backendStatus.innerHTML = `
                <span class="status-dot ${isReady ? 'ready' : 'error'}"></span>
                ${isReady ? 'Ready' : 'Connecting/Error'}
            `;
        }

        // Update device
        const deviceStatus = document.getElementById('deviceStatus');
        if (deviceStatus) {
            deviceStatus.textContent = status.device_name || status.device?.toUpperCase() || 'CPU';
        }

        // Update GPU memory
        const gpuMemory = document.getElementById('gpuMemory');
        if (gpuMemory && status.gpu_memory) {
            gpuMemory.textContent = `${status.gpu_memory.used} / ${status.gpu_memory.total} GB`;
        }

        // Update CPU usage
        const cpuUsage = document.getElementById('cpuUsage');
        const cpuMeter = document.getElementById('cpuMeter');
        if (cpuUsage && status.cpu_usage !== undefined) {
            cpuUsage.textContent = status.cpu_usage.toFixed(1);
            if (cpuMeter) cpuMeter.value = status.cpu_usage;
        }

        // Update RAM usage
        const ramUsage = document.getElementById('ramUsage');
        const ramMeter = document.getElementById('ramMeter');
        if (ramUsage && status.ram_usage !== undefined) {
            ramUsage.textContent = status.ram_usage.toFixed(1);
            if (ramMeter) ramMeter.value = status.ram_usage;
        }

        // Update models list
        this.updateModelsList(status.models_loaded || []);
    },

    /**
     * Update models list
     */
    updateModelsList(models) {
        const modelsList = document.getElementById('modelsList');
        if (!modelsList) return;

        if (!models || models.length === 0) {
            modelsList.innerHTML = '<div class="no-models">No models status available</div>';
            return;
        }

        modelsList.innerHTML = models.map(model => `
            <div class="model-item">
                <span class="model-icon">✓</span>
                <span class="model-name">${this.formatModelName(model.name || model)}</span>
                <span class="model-status ${model.status === 'loaded' ? 'loaded' : ''}">${model.status || 'Unknown'}</span>
            </div>
        `).join('');
    },

    /**
     * Format model name for display
     */
    formatModelName(name) {
        const map = {
            'whisper': 'Whisper (Transcription)',
            'wavlm': 'WavLM (Audio Features)',
            'mistral': 'Mistral-7B (Controller)',
            'sadtalker': 'SadTalker (Video Gen)',
            'gfpgan': 'GFPGAN (Enhancement)'
        };
        const key = typeof name === 'string' ? name.toLowerCase() : '';
        return map[key] || name;
    },

    /**
     * Get fallback status data (used when backend unreachable)
     */
    getFallbackStatus() {
        return {
            backend_status: 'error',
            device: 'Unknown',
            gpu_memory: { used: 0, total: 0 },
            cpu_usage: 0,
            ram_usage: 0,
            models_loaded: []
        };
    },

    /**
     * Start auto-refresh
     */
    startAutoRefresh() {
        this.stopAutoRefresh();
        this.state.updateInterval = setInterval(() => {
            if (this.state.isOpen) {
                this.fetchSystemStatus();
            }
        }, this.state.refreshRate);
    },

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.state.updateInterval) {
            clearInterval(this.state.updateInterval);
            this.state.updateInterval = null;
        }
    },

    /**
     * Save settings
     */
    saveSettings() {
        const settings = {
            device: this.elements.deviceSelect.value,
            precision: this.elements.precisionSelect.value,
            qualityPreset: this.elements.qualityPreset.value,
            autoRefresh: this.elements.autoRefresh.checked
        };

        localStorage.setItem('systemSettings', JSON.stringify(settings));
        UI.showToast('Settings saved successfully', 'success');

        // Optionally send to backend
        // fetch('/api/v1/system/config', { method: 'POST', body: JSON.stringify(settings) });
    },

    /**
     * Load saved settings
     */
    loadSettings() {
        try {
            const saved = localStorage.getItem('systemSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                if (this.elements.deviceSelect) this.elements.deviceSelect.value = settings.device || 'auto';
                if (this.elements.precisionSelect) this.elements.precisionSelect.value = settings.precision || 'fp16';
                if (this.elements.qualityPreset) this.elements.qualityPreset.value = settings.qualityPreset || 'balanced';
                if (this.elements.autoRefresh) this.elements.autoRefresh.checked = settings.autoRefresh !== false;
            }
        } catch (error) {
            console.error('[SystemSettings] Failed to load settings:', error);
        }
    }
};

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => SystemSettings.init());
} else {
    SystemSettings.init();
}
