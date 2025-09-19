/**
 * LocalVectorDB Inspector API Client
 *
 * JavaScript client for interacting with LocalVectorDB REST API and Inspector endpoints
 */

class LocalVectorDBAPIClient {
    constructor(config = {}) {
        this.apiBaseUrl = config.apiBaseUrl || '/api/v1';
        this.inspectorApiUrl = config.inspectorApiUrl || '/inspector/api';
        this.apiKey = config.apiKey || null;
        this.defaultTimeout = config.timeout || 30000;
        this.isAuthenticated = config.isAuthenticated || false;
        this.requireApiKey = config.requireApiKey || false;

        // Setup default headers
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };

        if (this.apiKey) {
            this.defaultHeaders['Authorization'] = `Bearer ${this.apiKey}`;
        }

        // Bind methods to preserve context
        this.get = this.get.bind(this);
        this.post = this.post.bind(this);
        this.put = this.put.bind(this);
        this.delete = this.delete.bind(this);
    }

    /**
     * Set API key for authenticated requests
     */
    setApiKey(apiKey) {
        this.apiKey = apiKey;
        if (apiKey) {
            this.defaultHeaders['Authorization'] = `Bearer ${apiKey}`;
        } else {
            delete this.defaultHeaders['Authorization'];
        }
    }

    /**
     * Make HTTP request with error handling
     */
    async request(url, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.defaultTimeout);

        try {
            const config = {
                ...options,
                signal: controller.signal,
                headers: {
                    ...this.defaultHeaders,
                    ...options.headers
                }
            };

            const response = await fetch(url, config);
            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    errorData = {error: 'Unknown error', message: response.statusText};
                }

                throw new APIError(
                    errorData.message || errorData.error || response.statusText,
                    response.status,
                    errorData
                );
            }

            // Handle empty responses
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }

        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new APIError('Request timeout', 408);
            }

            if (error instanceof APIError) {
                throw error;
            }

            throw new APIError(
                error.message || 'Network error',
                0,
                {originalError: error}
            );
        }
    }

    /**
     * HTTP GET request
     */
    async get(endpoint, params = {}) {
        const url = new URL(endpoint, window.location.origin);
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined) {
                url.searchParams.append(key, params[key]);
            }
        });

        return this.request(url.toString(), {method: 'GET'});
    }

    /**
     * HTTP POST request
     */
    async post(endpoint, data = null, options = {}) {
        const config = {
            method: 'POST',
            ...options
        };

        if (data !== null) {
            if (data instanceof FormData) {
                config.body = data;
                // Don't set Content-Type for FormData, let browser set it with boundary
                const headers = {...this.defaultHeaders, ...options.headers};
                delete headers['Content-Type'];
                config.headers = headers;
            } else {
                config.body = JSON.stringify(data);
            }
        }

        const url = new URL(endpoint, window.location.origin);
        return this.request(url.toString(), config);
    }

    /**
     * HTTP PUT request
     */
    async put(endpoint, data = null) {
        const config = {
            method: 'PUT'
        };

        if (data !== null) {
            config.body = JSON.stringify(data);
        }

        const url = new URL(endpoint, window.location.origin);
        return this.request(url.toString(), config);
    }

    /**
     * HTTP DELETE request
     */
    async delete(endpoint) {
        const url = new URL(endpoint, window.location.origin);
        return this.request(url.toString(), {method: 'DELETE'});
    }

    // ========== Database Management APIs ==========

    /**
     * List all databases
     */
    async listDatabases() {
        return this.get(`${this.apiBaseUrl}/databases`);
    }

    /**
     * Get database information
     */
    async getDatabaseInfo(dbName) {
        return this.get(`${this.apiBaseUrl}/${dbName}/info`);
    }

    /**
     * Create new database
     */
    async createDatabase(config) {
        return this.post(`${this.apiBaseUrl}/databases`, config);
    }

    /**
     * Delete database
     */
    async deleteDatabase(dbName) {
        return this.delete(`${this.apiBaseUrl}/${dbName}`);
    }

    /**
     * Get database statistics
     */
    async getDatabaseStats(dbName) {
        return this.get(`${this.apiBaseUrl}/${dbName}/info`);
    }

    // ========== Document Management APIs ==========

    /**
     * List documents with pagination
     */
    async listDocuments(dbName, params = {}) {
        const defaultParams = {
            page: 1,
            limit: 20
        };
        return this.get(`${this.apiBaseUrl}/${dbName}/documents`, {...defaultParams, ...params});
    }

    /**
     * Get document by ID
     */
    async getDocument(dbName, docId) {
        return this.get(`${this.apiBaseUrl}/${dbName}/documents/${docId}`);
    }

    /**
     * Add a single document to database
     */
    async addDocument(dbName, document) {
        const data = {
            content: document.content,
            id: document.id || undefined,
            metadata: document.metadata || undefined
        };
        return this.post(`${this.apiBaseUrl}/${dbName}/add`, data);
    }

    /**
     * Add documents to database
     */
    async addDocuments(dbName, documents, metadata = null, options = {}) {
        const data = {documents, ...options};
        if (metadata) {
            data.metadata = metadata;
        }
        return this.post(`${this.apiBaseUrl}/${dbName}/documents`, data);
    }

    /**
     * Update document
     */
    async updateDocument(dbName, docId, data) {
        return this.put(`${this.apiBaseUrl}/${dbName}/documents/${docId}`, data);
    }

    /**
     * Delete document
     */
    async deleteDocument(dbName, docId) {
        return this.delete(`${this.apiBaseUrl}/${dbName}/documents/${docId}`);
    }

    /**
     * Check if documents exist
     */
    async checkDocumentsExist(dbName, ids) {
        return this.post(`${this.apiBaseUrl}/${dbName}/documents/exists`, {ids});
    }

    /**
     * Filter documents with advanced query
     */
    async filterDocuments(dbName, filters = {}) {
        return this.post(`${this.apiBaseUrl}/${dbName}/filter`, filters);
    }

    // ========== Search APIs ==========

    /**
     * Unified query interface
     */
    async query(dbName, queryText, options = {}) {
        const data = {
            query: queryText,
            search_type: options.searchType || 'hybrid',
            return_type: options.returnType || 'documents',
            k: options.limit || 10,
            score_threshold: options.scoreThreshold || 0.0,
            ...options
        };

        return this.post(`${this.apiBaseUrl}/${dbName}/query`, data);
    }

    /**
     * Vector similarity search
     */
    async vectorSearch(dbName, queryText, options = {}) {
        return this.query(dbName, queryText, {...options, searchType: 'vector'});
    }

    /**
     * Keyword search
     */
    async keywordSearch(dbName, queryText, options = {}) {
        return this.query(dbName, queryText, {...options, searchType: 'keyword'});
    }

    /**
     * Hybrid search
     */
    async hybridSearch(dbName, queryText, options = {}) {
        return this.query(dbName, queryText, {...options, searchType: 'hybrid'});
    }

    /**
     * Multi-column search
     */
    async queryMultiColumn(dbName, queryText, columns = null, options = {}) {
        const data = {
            query: queryText,
            columns: columns,
            search_type: options.searchType || 'vector',
            return_type: options.returnType || 'documents',
            k: options.limit || 10,
            score_threshold: options.scoreThreshold || 0.0,
            ...options
        };

        return this.post(`${this.apiBaseUrl}/${dbName}/query-multi-column`, data);
    }

    /**
     * Global search across databases
     */
    async globalSearch(queryText, options = {}) {
        const data = {
            query: queryText,
            databases: options.databases || null,
            search_type: options.searchType || 'hybrid',
            return_type: options.returnType || 'documents',
            k: options.limit || 10,
            score_threshold: options.scoreThreshold || 0.0,
            ...options
        };

        return this.post(`${this.apiBaseUrl}/search`, data);
    }

    // ========== File Upload APIs ==========

    /**
     * Upload files with extraction
     */
    async uploadFiles(dbName, files, options = {}) {
        const formData = new FormData();

        // Add files
        if (files instanceof FileList) {
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }
        } else if (Array.isArray(files)) {
            files.forEach(file => formData.append('files', file));
        } else {
            formData.append('files', files);
        }

        // Add options
        if (options.metadata) {
            formData.append('metadata', JSON.stringify(options.metadata));
        }
        if (options.useFilenameAsId) {
            formData.append('use_filename_as_id', 'true');
        }
        if (options.batchSize) {
            formData.append('batch_size', options.batchSize.toString());
        }
        if (options.ids) {
            formData.append('ids', JSON.stringify(options.ids));
        }

        return this.post(`${this.apiBaseUrl}/${dbName}/upload`, formData);
    }

    /**
     * Get supported file formats
     */
    async getSupportedFormats() {
        return this.get(`${this.apiBaseUrl}/upload/supported-formats`);
    }

    /**
     * Preview file extraction
     */
    async previewExtraction(file) {
        const formData = new FormData();
        formData.append('file', file);

        return this.post(`${this.apiBaseUrl}/upload/extract-preview`, formData);
    }

    // ========== Embedding APIs ==========

    /**
     * Get embeddings for texts using database provider
     */
    async getDatabaseEmbeddings(dbName, texts = null, ids = null) {
        const data = {};
        if (texts) data.texts = Array.isArray(texts) ? texts : [texts];
        if (ids) data.ids = Array.isArray(ids) ? ids : [ids];

        return this.post(`${this.apiBaseUrl}/${dbName}/embeddings`, data);
    }

    /**
     * Get embeddings using specific provider
     */
    async getEmbeddings(provider, model, texts) {
        return this.post(`${this.apiBaseUrl}/embeddings`, {
            provider,
            model,
            texts: Array.isArray(texts) ? texts : [texts]
        });
    }

    // ========== Schema Management APIs ==========

    /**
     * Get metadata schema info
     */
    async getMetadataSchema(dbName) {
        return this.get(`${this.apiBaseUrl}/${dbName}/schema`);
    }

    /**
     * Update metadata schema
     */
    async updateMetadataSchema(dbName, schema, options = {}) {
        return this.put(`${this.apiBaseUrl}/${dbName}/schema`, {
            metadata_schema: schema,
            ...options
        });
    }

    // ========== System APIs ==========

    /**
     * Health check
     */
    async healthCheck() {
        return this.get(`${this.apiBaseUrl}/health`);
    }

    // ========== Inspector-specific APIs ==========

    /**
     * Get databases list for inspector
     */
    async getInspectorDatabases() {
        return this.get(`${this.inspectorApiUrl}/databases`);
    }

    /**
     * Get system statistics for inspector
     */
    async getSystemStats() {
        return this.get(`${this.inspectorApiUrl}/system/stats`);
    }
}

/**
 * API Error class for structured error handling
 */
class APIError extends Error {
    constructor(message, status = 0, details = {}) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.details = details;
    }

    toString() {
        return `APIError (${this.status}): ${this.message}`;
    }
}

/**
 * Global API client instance
 */
let InspectorAPI = null;

/**
 * Initialize API client with configuration
 */
function initializeAPIClient(config = {}) {
    InspectorAPI = new LocalVectorDBAPIClient(config);
    return InspectorAPI;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LocalVectorDBAPIClient,
        APIError,
        initializeAPIClient
    };
}

// Global reference
window.LocalVectorDBAPIClient = LocalVectorDBAPIClient;
window.APIError = APIError;
window.initializeAPIClient = initializeAPIClient;