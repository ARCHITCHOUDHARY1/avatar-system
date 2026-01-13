// ============================================================================
// API Client Module - RESTful API Communication
// ============================================================================

const API = {
    baseURL: window.location.origin,
    apiVersion: 'v1',
    timeout: 300000, // 5 minutes for generation
    maxRetries: 3,
    retryDelay: 1000,

    /**
     * Initialize API client
     */
    init() {
        this.checkHealth();
        console.log('[API] API client initialized');
    },

    /**
     * Build API endpoint URL
     * @param {string} path - API path
     * @returns {string} Full URL
     */
    buildURL(path) {
        const cleanPath = path.startsWith('/') ? path.slice(1) : path;
        return `${this.baseURL}/api/${this.apiVersion}/${cleanPath}`;
    },

    /**
     * Make HTTP request with error handling and retries
     * @param {string} url - Request URL
     * @param {Object} options - Fetch options
     * @param {number} retryCount - Current retry count
     * @returns {Promise<Response>}
     */
    async makeRequest(url, options = {}, retryCount = 0) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json',
                    ...options.headers
                }
            });

            clearTimeout(timeoutId);
            return response;

        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new Error('Request timeout. Please try again.');
            }

            if (retryCount < this.maxRetries && this.isRetryableError(error)) {
                console.warn(`[API] Retry ${retryCount + 1}/${this.maxRetries} for ${url}`);
                await this.delay(this.retryDelay * Math.pow(2, retryCount));
                return this.makeRequest(url, options, retryCount + 1);
            }

            throw error;
        }
    },

    /**
     * Check if error is retryable
     * @param {Error} error - Error object
     * @returns {boolean}
     */
    isRetryableError(error) {
        return error.name === 'TypeError' ||
            error.message.includes('network') ||
            error.message.includes('Failed to fetch');
    },

    /**
     * Delay helper for retries
     * @param {number} ms - Milliseconds to delay
     * @returns {Promise<void>}
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    /**
     * Handle API response
     * @param {Response} response - Fetch response
     * @returns {Promise<Object>}
     */
    async handleResponse(response) {
        const contentType = response.headers.get('content-type');

        if (!response.ok) {
            let errorMessage = `Request failed: ${response.status} ${response.statusText}`;

            try {
                if (contentType && contentType.includes('application/json')) {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorData.message || errorMessage;
                } else {
                    const errorText = await response.text();
                    if (errorText) {
                        errorMessage = errorText;
                    }
                }
            } catch (e) {
                console.error('[API] Error parsing error response:', e);
            }

            throw new Error(errorMessage);
        }

        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }

        return await response.text();
    },

    /**
     * Check API health status
     * @returns {Promise<Object>}
     */
    async checkHealth() {
        try {
            const response = await this.makeRequest(`${this.baseURL}/health`);
            const data = await this.handleResponse(response);

            const isHealthy = data.status === 'healthy';
            UI.updateApiStatus(isHealthy, isHealthy ? 'Connected' : 'Unhealthy');

            console.log('[API] Health check:', data);
            return data;

        } catch (error) {
            console.error('[API] Health check failed:', error);
            UI.updateApiStatus(false, 'Connection Error');
            throw error;
        }
    },

    /**
     * Upload image file
     * @param {File} file - Image file
     * @param {Function} onProgress - Progress callback
     * @returns {Promise<Object>}
     */
    async uploadImage(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        const url = this.buildURL('upload/image');

        try {
            console.log('[API] Uploading image:', file.name);

            const xhr = new XMLHttpRequest();

            return new Promise((resolve, reject) => {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentage = (e.loaded / e.total) * 100;
                        onProgress(percentage);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            console.log('[API] Image uploaded:', response);
                            resolve(response);
                        } catch (e) {
                            reject(new Error('Invalid response format'));
                        }
                    } else {
                        try {
                            const error = JSON.parse(xhr.responseText);
                            reject(new Error(error.detail || 'Upload failed'));
                        } catch (e) {
                            reject(new Error(`Upload failed: ${xhr.status}`));
                        }
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error during upload'));
                });

                xhr.addEventListener('timeout', () => {
                    reject(new Error('Upload timeout'));
                });

                xhr.open('POST', url);
                xhr.timeout = this.timeout;
                xhr.send(formData);
            });

        } catch (error) {
            console.error('[API] Image upload failed:', error);
            throw error;
        }
    },

    /**
     * Upload audio file
     * @param {File} file - Audio file
     * @param {Function} onProgress - Progress callback
     * @returns {Promise<Object>}
     */
    async uploadAudio(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        const url = this.buildURL('upload/audio');

        try {
            console.log('[API] Uploading audio:', file.name);

            const xhr = new XMLHttpRequest();

            return new Promise((resolve, reject) => {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentage = (e.loaded / e.total) * 100;
                        onProgress(percentage);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            console.log('[API] Audio uploaded:', response);
                            resolve(response);
                        } catch (e) {
                            reject(new Error('Invalid response format'));
                        }
                    } else {
                        try {
                            const error = JSON.parse(xhr.responseText);
                            reject(new Error(error.detail || 'Upload failed'));
                        } catch (e) {
                            reject(new Error(`Upload failed: ${xhr.status}`));
                        }
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error during upload'));
                });

                xhr.addEventListener('timeout', () => {
                    reject(new Error('Upload timeout'));
                });

                xhr.open('POST', url);
                xhr.timeout = this.timeout;
                xhr.send(formData);
            });

        } catch (error) {
            console.error('[API] Audio upload failed:', error);
            throw error;
        }
    },

    /**
     * Upload video file
     * @param {File} file - Video file
     * @param {Function} onProgress - Progress callback
     * @returns {Promise<Object>}
     */
    async uploadVideo(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        const url = this.buildURL('upload/video');

        try {
            console.log('[API] Uploading video:', file.name);

            const xhr = new XMLHttpRequest();

            return new Promise((resolve, reject) => {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentage = (e.loaded / e.total) * 100;
                        onProgress(percentage);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            console.log('[API] Video uploaded:', response);
                            resolve(response);
                        } catch (e) {
                            reject(new Error('Invalid response format'));
                        }
                    } else {
                        try {
                            const error = JSON.parse(xhr.responseText);
                            reject(new Error(error.detail || 'Upload failed'));
                        } catch (e) {
                            reject(new Error(`Upload failed: ${xhr.status}`));
                        }
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error during upload'));
                });

                xhr.addEventListener('timeout', () => {
                    reject(new Error('Upload timeout'));
                });

                xhr.open('POST', url);
                xhr.timeout = this.timeout;
                xhr.send(formData);
            });

        } catch (error) {
            console.error('[API] Video upload failed:', error);
            throw error;
        }
    },

    /**
     * Generate avatar
     * @param {Object} params - Generation parameters
     * @returns {Promise<Object>}
     */
    async generateAvatar(params) {
        const url = this.buildURL('generate');

        try {
            console.log('[API] Generating avatar with params:', params);

            const response = await this.makeRequest(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const data = await this.handleResponse(response);
            console.log('[API] Avatar generation response:', data);
            return data;

        } catch (error) {
            console.error('[API] Avatar generation failed:', error);
            throw error;
        }
    },

    /**
     * Get generation status
     * @param {string} taskId - Task ID
     * @returns {Promise<Object>}
     */
    async getStatus(taskId) {
        const url = this.buildURL(`status/${taskId}`);

        try {
            const response = await this.makeRequest(url);
            const data = await this.handleResponse(response);
            return data;

        } catch (error) {
            console.error('[API] Get status failed:', error);
            throw error;
        }
    },

    /**
     * Poll for status updates
     * @param {string} taskId - Task ID
     * @param {Function} onUpdate - Update callback
     * @param {number} interval - Poll interval in ms
     * @returns {Promise<Object>} Final result
     */
    async pollStatus(taskId, onUpdate, interval = 2000) {
        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    const status = await this.getStatus(taskId);

                    if (onUpdate) {
                        onUpdate(status);
                    }

                    if (status.status === 'completed') {
                        resolve(status);
                    } else if (status.status === 'failed' || status.status === 'error') {
                        reject(new Error(status.error || 'Generation failed'));
                    } else {
                        setTimeout(poll, interval);
                    }

                } catch (error) {
                    reject(error);
                }
            };

            poll();
        });
    },

    /**
     * Download file from URL
     * @param {string} url - File URL
     * @param {string} filename - Download filename
     */
    async downloadFile(url, filename) {
        try {
            console.log('[API] Downloading file:', url);

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error('Download failed');
            }

            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(downloadUrl);

            console.log('[API] File downloaded successfully');

        } catch (error) {
            console.error('[API] Download failed:', error);
            throw error;
        }
    },

    /**
     * Validate API response
     * @param {Object} data - Response data
     * @param {string[]} requiredFields - Required fields
     * @returns {boolean}
     */
    validateResponse(data, requiredFields = []) {
        if (!data || typeof data !== 'object') {
            return false;
        }

        return requiredFields.every(field => {
            return data.hasOwnProperty(field) && data[field] !== null && data[field] !== undefined;
        });
    },

    /**
     * Get error message from error object
     * @param {Error} error - Error object
     * @returns {string} User-friendly error message
     */
    getErrorMessage(error) {
        if (error.message) {
            return error.message;
        }

        if (typeof error === 'string') {
            return error;
        }

        return 'An unexpected error occurred. Please try again.';
    }
};

// Initialize API client when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => API.init());
} else {
    API.init();
}

// Export for use in other modules
window.API = API;
