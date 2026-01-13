/**
 * Audio Visualization Handler
 * Handles frequency spectrum, waveform, and metrics visualization
 */

(function () {
    'use strict';

    // Canvas elements
    let frequencyCanvas, frequencyCtx;
    let waveformCanvas, waveformCtx;

    // UI elements
    let audioEnergy, audioPitch;
    let bassValue, bassFill;
    let midValue, midFill;
    let trebleValue, trebleFill;

    /**
     * Initialize visualizer
     */
    function init() {
        // Get canvas elements
        frequencyCanvas = document.getElementById('frequencyCanvas');
        waveformCanvas = document.getElementById('waveformCanvas');

        if (!frequencyCanvas || !waveformCanvas) {
            console.warn('[AudioVisualizer] Canvas elements not found');
            return;
        }

        frequencyCtx = frequencyCanvas.getContext('2d');
        waveformCtx = waveformCanvas.getContext('2d');

        // Get UI elements
        audioEnergy = document.getElementById('audioEnergy');
        audioPitch = document.getElementById('audioPitch');
        bassValue = document.getElementById('bassValue');
        bassFill = document.getElementById('bassFill');
        midValue = document.getElementById('midValue');
        midFill = document.getElementById('midFill');
        trebleValue = document.getElementById('trebleValue');
        trebleFill = document.getElementById('trebleFill');

        // Listen for audio level events
        window.addEventListener('microphoneAudioLevel', handleAudioMetrics);

        console.log('[AudioVisualizer] Initialized');
    }

    /**
     * Handle audio metrics update
     */
    function handleAudioMetrics(event) {
        const metrics = event.detail;

        // Update metrics display
        if (audioEnergy) {
            audioEnergy.textContent = `${metrics.energy.toFixed(1)} dB`;
        }

        if (audioPitch) {
            audioPitch.textContent = metrics.pitch > 0 ? `${metrics.pitch} Hz` : '-- Hz';
        }

        // Update frequency bands
        if (metrics.bands) {
            updateFrequencyBand(bassValue, bassFill, metrics.bands.bass);
            updateFrequencyBand(midValue, midFill, metrics.bands.mid);
            updateFrequencyBand(trebleValue, trebleFill, metrics.bands.treble);
        }

        // Draw frequency spectrum
        if (metrics.frequencyData && frequencyCtx) {
            drawFrequencySpectrum(metrics.frequencyData);
        }

        // Draw waveform
        if (metrics.waveformData && waveformCtx) {
            drawWaveform(metrics.waveformData);
        }
    }

    /**
     * Update frequencyband display
     */
    function updateFrequencyBand(valueElem, fillElem, value) {
        if (valueElem) {
            valueElem.textContent = `${value}%`;
        }
        if (fillElem) {
            fillElem.style.width = `${value}%`;
        }
    }

    /**
     * Draw frequency spectrum
     */
    function drawFrequencySpectrum(dataArray) {
        const width = frequencyCanvas.width;
        const height = frequencyCanvas.height;

        // Clear canvas
        frequencyCtx.clearRect(0, 0, width, height);

        // bar width
        const barWidth = (width / dataArray.length) * 2.5;
        let x = 0;

        // Draw frequency bars
        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * height;

            // Create gradient for each bar
            const gradient = frequencyCtx.createLinearGradient(0, height - barHeight, 0, height);

            // Color based on frequency range
            if (i < dataArray.length * 0.15) {
                // Bass - blue/purple
                gradient.addColorStop(0, '#667eea');
                gradient.addColorStop(1, '#764ba2');
            } else if (i < dataArray.length * 0.5) {
                // Mid - pink/red
                gradient.addColorStop(0, '#f093fb');
                gradient.addColorStop(1, '#f5576c');
            } else {
                // Treble - cyan/blue
                gradient.addColorStop(0, '#4facfe');
                gradient.addColorStop(1, '#00f2fe');
            }

            frequencyCtx.fillStyle = gradient;
            frequencyCtx.fillRect(x, height - barHeight, barWidth, barHeight);

            x += barWidth + 1;
        }
    }

    /**
     * Draw waveform
     */
    function drawWaveform(dataArray) {
        const width = waveformCanvas.width;
        const height = waveformCanvas.height;

        // Clear canvas
        waveformCtx.clearRect(0, 0, width, height);

        // Draw waveform line
        waveformCtx.lineWidth = 2;
        waveformCtx.strokeStyle = '#667eea';
        waveformCtx.beginPath();

        const sliceWidth = width / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * height) / 2;

            if (i === 0) {
                waveformCtx.moveTo(x, y);
            } else {
                waveformCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        waveformCtx.lineTo(width, height / 2);
        waveformCtx.stroke();

        // Add glow effect
        waveformCtx.shadowBlur = 10;
        waveformCtx.shadowColor = '#667eea';
        waveformCtx.stroke();
        waveformCtx.shadowBlur = 0;
    }

    /**
     * Clear visualizations
     */
    function clear() {
        if (frequencyCtx) {
            frequencyCtx.clearRect(0, 0, frequencyCanvas.width, frequencyCanvas.height);
        }
        if (waveformCtx) {
            waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        }

        // Reset metrics
        if (audioEnergy) audioEnergy.textContent = '-60 dB';
        if (audioPitch) audioPitch.textContent = '-- Hz';

        updateFrequencyBand(bassValue, bassFill, 0);
        updateFrequencyBand(midValue, midFill, 0);
        updateFrequencyBand(trebleValue, trebleFill, 0);
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Export for external use
    window.AudioVisualizer = {
        init,
        clear
    };

})();
