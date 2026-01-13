// ============================================================================
// Dashboard Module - Fetch and display stats/metrics
// ============================================================================

const Dashboard = {
    refreshInterval: 5000, // 5 seconds
    intervalId: null,

    /**
     * Initialize dashboard
     */
    init() {
        this.loadDashboardStats();
        this.loadPerformanceMetrics();

        // Auto-refresh every 5 seconds
        this.intervalId = setInterval(() => {
            this.loadDashboardStats();
            this.loadPerformanceMetrics();
        }, this.refreshInterval);

        console.log('[Dashboard] Dashboard initialized');
    },

    /**
     * Load dashboard statistics
     */
    async loadDashboardStats() {
        try {
            const response = await fetch(`${API.baseURL}/api/v1/stats/dashboard`);
            if (!response.ok) {
                throw new Error('Failed to fetch dashboard stats');
            }

            const stats = await response.json();

            // Update stat cards
            document.getElementById('statTotalJobs').textContent = stats.total_jobs || 0;
            document.getElementById('statCompleted').textContent = stats.completed || 0;
            document.getElementById('statSuccessRate').textContent = stats.success_rate || 0;
            document.getElementById('statAvgFPS').textContent = stats.avg_fps || 0;
            document.getElementById('statQuality').textContent = stats.avg_quality_score || 0;

            console.log('[Dashboard] Stats updated:', stats);

        } catch (error) {
            console.error('[Dashboard] Failed to load stats:', error);
        }
    },

    /**
     * Load performance metrics
     */
    async loadPerformanceMetrics() {
        try {
            const response = await fetch(`${API.baseURL}/api/v1/stats/performance`);
            if (!response.ok) {
                throw new Error('Failed to fetch performance metrics');
            }

            const metrics = await response.json();

            // Update device info
            const deviceName = metrics.device?.name || 'CPU';
            const deviceType = metrics.device?.type || 'cpu';
            document.getElementById('statDevice').textContent = deviceType.toUpperCase();

            // Update performance values
            const gpuUsed = metrics.device?.memory_used_gb?.toFixed(2) || 0;
            const gpuTotal = metrics.device?.memory_total_gb?.toFixed(2) || 0;
            document.getElementById('perfGPUMem').textContent = `${gpuUsed} / ${gpuTotal} GB`;

            const ramPercent = metrics.system?.ram_percent?.toFixed(1) || 0;
            document.getElementById('perfRAM').textContent = `${ramPercent}%`;

            const cpuPercent = metrics.system?.cpu_percent?.toFixed(1) || 0;
            document.getElementById('perfCPU').textContent = `${cpuPercent}%`;

            console.log('[Dashboard] Performance metrics updated:', metrics);

        } catch (error) {
            console.error('[Dashboard] Failed to load performance metrics:', error);
        }
    },

    /**
     * Stop auto-refresh
     */
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
            console.log('[Dashboard] Auto-refresh stopped');
        }
    }
};

// Initialize dashboard when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => Dashboard.init());
} else {
    Dashboard.init();
}

// Export for debugging
window.Dashboard = Dashboard;
