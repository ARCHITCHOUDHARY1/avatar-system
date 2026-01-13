// ============================================================================
// UI Utilities Module - Toast Notifications, Modals, and Helpers
// ============================================================================

const UI = {
    // Toast notification configuration
    toastDuration: 5000,
    toastContainer: null,
    activeToasts: new Set(),

    /**
     * Initialize UI utilities
     */
    init() {
        this.toastContainer = document.getElementById('toastContainer');
        this.setupThemeToggle();
        console.log('[UI] UI utilities initialized');
    },

    /**
     * Show toast notification
     * @param {string} type - success, error, warning, info
     * @param {string} title - Toast title
     * @param {string} message - Toast message
     * @param {number} duration - Duration in milliseconds (0 for persistent)
     */
    showToast(type, title, message, duration = null) {
        if (!this.toastContainer) {
            console.error('[UI] Toast container not found');
            return;
        }

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const icons = {
            success: `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>`,
            error: `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>`,
            warning: `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
            </svg>`,
            info: `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
            </svg>`
        };

        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-content">
                <div class="toast-title">${this.escapeHtml(title)}</div>
                <div class="toast-message">${this.escapeHtml(message)}</div>
            </div>
        `;

        this.toastContainer.appendChild(toast);
        this.activeToasts.add(toast);

        const toastDuration = duration !== null ? duration : this.toastDuration;

        if (toastDuration > 0) {
            setTimeout(() => {
                this.removeToast(toast);
            }, toastDuration);
        }

        console.log(`[UI] Toast shown: ${type} - ${title}`);
        return toast;
    },

    /**
     * Remove toast notification
     * @param {HTMLElement} toast - Toast element to remove
     */
    removeToast(toast) {
        if (!toast || !this.activeToasts.has(toast)) return;

        toast.classList.add('remove');
        this.activeToasts.delete(toast);

        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 200);
    },

    /**
     * Clear all toast notifications
     */
    clearAllToasts() {
        this.activeToasts.forEach(toast => {
            this.removeToast(toast);
        });
    },

    /**
     * Show loading overlay
     * @param {string} text - Loading text
     */
    showLoading(text = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            const loadingText = overlay.querySelector('.loading-text');
            if (loadingText) {
                loadingText.textContent = text;
            }
            overlay.style.display = 'flex';
            console.log('[UI] Loading overlay shown');
        }
    },

    /**
     * Hide loading overlay
     */
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
            console.log('[UI] Loading overlay hidden');
        }
    },

    /**
     * Update progress bar
     * @param {number} percentage - Progress percentage (0-100)
     * @param {string} status - Status text
     */
    updateProgress(percentage, status = '') {
        const progressBar = document.getElementById('progressBar');
        const progressPercentage = document.getElementById('progressPercentage');
        const progressStatus = document.getElementById('progressStatus');

        if (progressBar) {
            progressBar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }

        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(percentage)}%`;
        }

        if (progressStatus && status) {
            progressStatus.textContent = status;
        }
    },

    /**
     * Show progress section
     */
    showProgress() {
        const section = document.getElementById('progressSection');
        if (section) {
            section.style.display = 'block';
            this.updateProgress(0, 'Initializing...');
            console.log('[UI] Progress section shown');
        }
    },

    /**
     * Hide progress section
     */
    hideProgress() {
        const section = document.getElementById('progressSection');
        if (section) {
            section.style.display = 'none';
            console.log('[UI] Progress section hidden');
        }
    },

    /**
     * Show result section
     */
    showResult() {
        const section = document.getElementById('resultSection');
        if (section) {
            section.style.display = 'block';
            console.log('[UI] Result section shown');
        }
    },

    /**
     * Hide result section
     */
    hideResult() {
        const section = document.getElementById('resultSection');
        if (section) {
            section.style.display = 'none';
            console.log('[UI] Result section hidden');
        }
    },

    /**
     * Format file size to human readable format
     * @param {number} bytes - File size in bytes
     * @returns {string} Formatted file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
    },

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Validate file type
     * @param {File} file - File to validate
     * @param {string[]} allowedTypes - Array of allowed MIME types
     * @returns {boolean} True if valid
     */
    validateFileType(file, allowedTypes) {
        return allowedTypes.some(type => {
            if (type.includes('*')) {
                const baseType = type.split('/')[0];
                return file.type.startsWith(baseType);
            }
            return file.type === type;
        });
    },

    /**
     * Validate file size
     * @param {File} file - File to validate
     * @param {number} maxSizeMB - Maximum size in MB
     * @returns {boolean} True if valid
     */
    validateFileSize(file, maxSizeMB) {
        const maxBytes = maxSizeMB * 1024 * 1024;
        return file.size <= maxBytes;
    },

    /**
     * Create image preview from file
     * @param {File} file - Image file
     * @returns {Promise<string>} Data URL of image
     */
    createImagePreview(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    /**
     * Update API status indicator
     * @param {boolean} connected - Connection status
     * @param {string} text - Status text
     */
    updateApiStatus(connected, text = '') {
        const indicator = document.getElementById('apiStatus');
        if (!indicator) return;

        const statusText = indicator.querySelector('.status-text');

        if (connected) {
            indicator.classList.remove('error');
            indicator.classList.add('connected');
            if (statusText) {
                statusText.textContent = text || 'Connected';
            }
        } else {
            indicator.classList.remove('connected');
            indicator.classList.add('error');
            if (statusText) {
                statusText.textContent = text || 'Disconnected';
            }
        }
    },

    /**
     * Setup theme toggle functionality
     */
    setupThemeToggle() {
        const toggle = document.getElementById('themeToggle');
        if (!toggle) return;

        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);

        toggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';

            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            console.log(`[UI] Theme changed to ${newTheme}`);
        });
    },

    /**
     * Smooth scroll to element
     * @param {string} elementId - Element ID to scroll to
     */
    scrollToElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    },

    /**
     * Debounce function to limit execution rate
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Copy text to clipboard
     * @param {string} text - Text to copy
     * @returns {Promise<void>}
     */
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showToast('success', 'Copied', 'Text copied to clipboard');
            console.log('[UI] Text copied to clipboard');
        } catch (err) {
            console.error('[UI] Failed to copy text:', err);
            this.showToast('error', 'Copy Failed', 'Failed to copy text to clipboard');
        }
    },

    /**
     * Check if browser supports required features
     * @returns {Object} Feature support object
     */
    checkBrowserSupport() {
        const support = {
            fileAPI: !!(window.File && window.FileReader && window.FileList && window.Blob),
            fetch: !!window.fetch,
            promises: !!window.Promise,
            localStorage: !!window.localStorage,
            clipboard: !!navigator.clipboard
        };

        const allSupported = Object.values(support).every(v => v);

        if (!allSupported) {
            console.warn('[UI] Browser feature support:', support);
        }

        return { ...support, allSupported };
    },

    /**
     * Log debug information
     * @param {string} category - Log category
     * @param {string} message - Log message
     * @param {*} data - Additional data
     */
    log(category, message, data = null) {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] [${category}] ${message}`;

        if (data) {
            console.log(logMessage, data);
        } else {
            console.log(logMessage);
        }
    }
};

// Initialize UI when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => UI.init());
} else {
    UI.init();
}

// Export for use in other modules
window.UI = UI;

// End of UI utilities
