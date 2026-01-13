// ============================================================================
// Unified App - Hardware Detection & Troubleshooting
// ============================================================================

/**
 * Toggle troubleshooter modal
 */
function toggleTroubleshooter() {
    const modal = document.getElementById('troubleshooterModal');
    if (modal) {
        modal.style.display = modal.style.display === 'flex' ? 'none' : 'flex';
    }
}

/**
 * Copy diagnostic report to clipboard
 */
function copyDiagnostics() {
    const hardwareLabel = document.getElementById('hardwareLabel')?.innerText || 'Unknown';
    const modelLabel = document.getElementById('modelLabel')?.innerText || 'Unknown';

    const report = `Avatar System Diagnostic Report
Time: ${new Date().toISOString()}
User Agent: ${navigator.userAgent}
Platform: ${navigator.platform}
Hardware: ${hardwareLabel}
Model: ${modelLabel}
Resolution: ${window.screen.width}x${window.screen.height}
Language: ${navigator.language}
WebGL: ${detectWebGL() ? 'Enabled' : 'Disabled'}
`;

    navigator.clipboard.writeText(report).then(() => {
        alert('Diagnostic report copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy diagnostics:', err);
        alert('Failed to copy diagnostic report');
    });
}

/**
 * Detect WebGL support
 */
function detectWebGL() {
    try {
        const canvas = document.createElement('canvas');
        return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    } catch (e) {
        return false;
    }
}

/**
 * Check hardware status and update widget
 */
async function checkHardware() {
    const widget = document.getElementById('hardwareStatus');
    const label = document.getElementById('hardwareLabel');
    const model = document.getElementById('modelLabel');
    const estimate = document.getElementById('timeEstimate');
    const statDevice = document.getElementById('statDevice');

    if (!widget) return;

    try {
        // Call backend API to check hardware
        const response = await fetch('/api/v1/status');
        if (response.ok) {
            const status = await response.json();

            if (status.gpu_available) {
                widget.classList.add('gpu-mode');
                widget.classList.remove('cpu-mode');
                label.innerText = 'GPU ACCELERATED';
                model.innerText = status.model || 'SadTalker Active';
                if (estimate) estimate.innerText = '~30-60 Seconds';
                if (statDevice) statDevice.innerText = 'GPU';

                // Update GPU metric
                const gpuMetric = document.getElementById('gpuMetric');
                if (gpuMetric && status.gpu_memory) {
                    gpuMetric.innerText = status.gpu_memory;
                }
            } else {
                widget.classList.add('cpu-mode');
                widget.classList.remove('gpu-mode');
                label.innerText = 'CPU OPTIMIZED';
                model.innerText = 'Wav2Lip Active';
                if (estimate) estimate.innerText = '~3-5 Minutes';
                if (statDevice) statDevice.innerText = 'CPU';

                const gpuMetric = document.getElementById('gpuMetric');
                if (gpuMetric) gpuMetric.innerText = 'N/A';
            }

            // Update diagnostic backend status
            const diagBackend = document.getElementById('diagBackend');
            if (diagBackend) {
                diagBackend.innerText = 'Connected';
                diagBackend.style.color = '#00f2fe';
            }
        } else {
            throw new Error('Backend not responding');
        }
    } catch (error) {
        console.error('Hardware check failed:', error);

        // Default to CPU mode
        widget.classList.add('cpu-mode');
        label.innerText = 'CPU MODE (Backend Offline)';
        model.innerText = 'Connecting...';
        if (estimate) estimate.innerText = 'Unknown';

        const diagBackend = document.getElementById('diagBackend');
        if (diagBackend) {
            diagBackend.innerText = 'Disconnected';
            diagBackend.style.color = '#ff5555';
        }
    }
}

/**
 * Update system metrics (mock data for now, will connect to backend)
 */
function updateMockMetrics() {
    const cpu = Math.floor(Math.random() * 20) + 10; // 10-30%
    const ram = (Math.random() * 0.5 + 4).toFixed(1); // 4.0-4.5 GB

    const cpuMetric = document.getElementById('cpuMetric');
    const ramMetric = document.getElementById('ramMetric');
    const perfCPU = document.getElementById('perfCPU');
    const perfRAM = document.getElementById('perfRAM');

    if (cpuMetric) cpuMetric.innerText = `${cpu}%`;
    if (ramMetric) ramMetric.innerText = `${ram} GB`;
    if (perfCPU) perfCPU.innerText = `${cpu}%`;
    if (perfRAM) perfRAM.innerText = `${(ram / 8 * 100).toFixed(1)}%`;

    const widget = document.getElementById('hardwareStatus');
    const gpuMetric = document.getElementById('gpuMetric');

    if (widget && widget.classList.contains('gpu-mode') && gpuMetric) {
        gpuMetric.innerText = `${Math.floor(Math.random() * 40)}%`;
    }
}

/**
 * Update diagnostic information
 */
function updateDiagnosticInfo() {
    const diagOS = document.getElementById('diagOS');
    const diagBrowser = document.getElementById('diagBrowser');
    const diagWebGL = document.getElementById('diagWebGL');

    if (diagOS) {
        if (navigator.platform.includes('Win')) diagOS.innerText = 'Windows';
        else if (navigator.platform.includes('Mac')) diagOS.innerText = 'macOS';
        else if (navigator.platform.includes('Linux')) diagOS.innerText = 'Linux';
        else diagOS.innerText = navigator.platform;
    }

    if (diagBrowser) {
        const ua = navigator.userAgent;
        if (ua.includes('Firefox')) diagBrowser.innerText = 'Firefox';
        else if (ua.includes('Chrome')) diagBrowser.innerText = 'Chrome';
        else if (ua.includes('Safari')) diagBrowser.innerText = 'Safari';
        else if (ua.includes('Edge')) diagBrowser.innerText = 'Edge';
        else diagBrowser.innerText = 'Unknown';
    }

    if (diagWebGL) {
        diagWebGL.innerText = detectWebGL() ? 'Enabled' : 'Disabled';
        diagWebGL.style.color = detectWebGL() ? '#00f2fe' : '#ff5555';
    }
}

/**
 * Initialize unified app on page load
 */
function initUnifiedApp() {
    console.log('[UnifiedApp] Initializing...');

    // Check hardware status
    checkHardware();

    // Update diagnostic info
    updateDiagnosticInfo();

    // Start metrics update interval
    setInterval(updateMockMetrics, 2000);

    // Re-check hardware every 30 seconds
    setInterval(checkHardware, 30000);

    console.log('[UnifiedApp] Initialization complete');
}

// Browser compatibility: Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initUnifiedApp);
} else {
    initUnifiedApp();
}

// Export functions to global scope
window.toggleTroubleshooter = toggleTroubleshooter;
window.copyDiagnostics = copyDiagnostics;
window.checkHardware = checkHardware;
