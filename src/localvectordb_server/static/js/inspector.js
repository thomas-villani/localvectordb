/**
 * LocalVectorDB Inspector Main Application
 *
 * Main JavaScript application logic for the LocalVectorDB Inspector UI
 */

class InspectorApp {
    constructor(config = {}) {
        this.config = {
            apiBaseUrl: '/api/v1',
            inspectorApiUrl: '/inspector/api',
            isAuthenticated: false,
            requireApiKey: false,
            refreshInterval: 30000, // 30 seconds
            ...config
        };

        this.apiClient = null;
        this.refreshTimer = null;
        this.currentPage = null;
        this.eventListeners = new Map();

        // Bind methods
        this.init = this.init.bind(this);
        this.setupEventListeners = this.setupEventListeners.bind(this);
        this.handlePageNavigation = this.handlePageNavigation.bind(this);
        this.handleFormSubmission = this.handleFormSubmission.bind(this);
        this.loadDynamicContent = this.loadDynamicContent.bind(this);
    }

    /**
     * Show document detail in modal
     */
    static async showDocumentDetail(docId) {
        const app = window.InspectorApp;
        if (!app) return;

        const dbName = app.getCurrentDatabase();
        if (!dbName) return;

        try {
            showLoading();
            const doc = await app.apiClient.getDocument(dbName, docId);
            hideLoading();

            const modalContent = `
                <div class="document-detail">
                    <div class="detail-section">
                        <h4>Document Information</h4>
                        <div class="info-grid">
                            <div class="info-item">
                                <label>ID:</label>
                                <span>${doc.id}</span>
                            </div>
                            <div class="info-item">
                                <label>Content Hash:</label>
                                <span class="monospace">${doc.content_hash || 'N/A'}</span>
                            </div>
                            <div class="info-item">
                                <label>Created:</label>
                                <span>${doc.created_at ? new Date(doc.created_at).toLocaleString() : 'N/A'}</span>
                            </div>
                            <div class="info-item">
                                <label>Updated:</label>
                                <span>${doc.updated_at ? new Date(doc.updated_at).toLocaleString() : 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h4>Content</h4>
                        <div class="document-content">
                            <pre>${app.escapeHtml(doc.content)}</pre>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h4>Metadata</h4>
                        <div class="metadata-display">
                            ${app.formatMetadataDetailed(doc.metadata)}
                        </div>
                    </div>
                </div>
            `;

            showModal(`Document: ${docId}`, modalContent, `
                <button class="btn btn-secondary" onclick="closeModal()">Close</button>
                <button class="btn btn-primary" onclick="InspectorApp.editDocument('${docId}')">Edit Document</button>
            `);

        } catch (error) {
            hideLoading();
            app.showAlert(`Failed to load document: ${error.message}`, 'error');
        }
    }

    /**
     * Static method to load documents (for pagination buttons)
     */
    static loadDocuments(page = 1) {
        const app = window.InspectorApp;
        if (app && app.loadDocuments) {
            app.loadDocuments(page);
        }
    }

    /**
     * Edit document
     */
    static async editDocument(docId) {
        console.log('Edit document:', docId);
        // Edit document functionality
    }

    /**
     * Delete document
     */
    static async deleteDocument(docId) {
        if (confirm('Are you sure you want to delete this document?')) {
            console.log('Delete document:', docId);
            // Delete document functionality
        }
    }

    /**
     * Initialize the inspector application
     */
    init(config = {}) {
        // Merge configuration
        this.config = {...this.config, ...config};

        // Initialize API client
        this.apiClient = initializeAPIClient(this.config);
        window.InspectorAPI = this.apiClient;

        // Setup global event listeners
        this.setupEventListeners();

        // Detect current page and initialize page-specific functionality
        this.detectCurrentPage();

        // Start background refresh if on dashboard
        if (this.currentPage === 'dashboard') {
            this.startBackgroundRefresh();
        }

        console.log('LocalVectorDB Inspector initialized');
    }

    /**
     * Setup global event listeners
     */
    setupEventListeners() {
        // Handle navigation clicks
        document.addEventListener('click', (event) => {
            const link = event.target.closest('[data-nav-target]');
            if (link) {
                event.preventDefault();
                this.handlePageNavigation(link.dataset.navTarget);
            }
        });

        // Handle form submissions
        document.addEventListener('submit', (event) => {
            const form = event.target.closest('[data-inspector-form]');
            if (form) {
                event.preventDefault();
                this.handleFormSubmission(form);
            }
        });

        // Handle dynamic content loading
        document.addEventListener('click', (event) => {
            const trigger = event.target.closest('[data-load-content]');
            if (trigger) {
                event.preventDefault();
                this.loadDynamicContent(trigger);
            }
        });

        // Handle window resize for responsive charts
        window.addEventListener('resize', this.debounce(() => {
            this.handleWindowResize();
        }, 300));

        // Handle visibility change for pausing/resuming updates
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseBackgroundUpdates();
            } else {
                this.resumeBackgroundUpdates();
            }
        });
    }

    /**
     * Detect current page based on URL or page markers
     */
    detectCurrentPage() {
        const path = window.location.pathname;
        const body = document.body;

        if (path.includes('/inspector/query') || body.classList.contains('query-page')) {
            this.currentPage = 'query';
            this.initializeQueryPage();
        } else if (path.includes('/inspector/database/') || body.classList.contains('database-page')) {
            this.currentPage = 'database';
            this.initializeDatabasePage();
        } else if (path.includes('/inspector/embeddings') || body.classList.contains('embeddings-page')) {
            this.currentPage = 'embeddings';
            this.initializeEmbeddingsPage();
        } else if (path.includes('/inspector/admin') || body.classList.contains('admin-page')) {
            this.currentPage = 'admin';
            this.initializeAdminPage();
        } else if (path.includes('/inspector') || body.classList.contains('dashboard-page')) {
            this.currentPage = 'dashboard';
            this.initializeDashboardPage();
        }
    }

    /**
     * Initialize dashboard-specific functionality
     */
    initializeDashboardPage() {
        // Setup real-time statistics updates
        this.setupDashboardUpdates();

        // Initialize dashboard widgets
        this.initializeDashboardWidgets();
    }

    /**
     * Initialize query page functionality
     */
    initializeQueryPage() {
        // Setup query form handlers
        this.setupQueryHandlers();

        // Initialize query history
        this.loadQueryHistory();

        // Setup auto-complete for query input
        this.setupQueryAutoComplete();
    }

    /**
     * Initialize database detail page functionality
     */
    initializeDatabasePage() {
        // Setup document browser
        this.setupDocumentBrowser();

        // Initialize pagination
        this.setupPagination();

        // Setup document detail modals
        this.setupDocumentModals();
    }

    /**
     * Initialize embeddings visualization page
     */
    initializeEmbeddingsPage() {
        // Setup visualization controls
        this.setupVisualizationControls();

        // Initialize chart containers
        this.initializeCharts();
    }

    /**
     * Initialize admin page functionality
     */
    initializeAdminPage() {
        // Setup system monitoring
        this.setupSystemMonitoring();

        // Initialize admin controls
        this.setupAdminControls();
    }

    /**
     * Setup dashboard real-time updates
     */
    setupDashboardUpdates() {
        const updateDashboard = async () => {
            try {
                // Update system stats
                const systemStats = await this.apiClient.getSystemStats();
                this.updateSystemStatsDisplay(systemStats);

                // Update database list
                const databases = await this.apiClient.getInspectorDatabases();
                this.updateDatabaseGrid(databases.databases);

            } catch (error) {
                console.warn('Failed to update dashboard:', error);
            }
        };

        // Initial update
        updateDashboard();

        // Setup periodic updates
        this.dashboardUpdateInterval = setInterval(updateDashboard, this.config.refreshInterval);
    }

    /**
     * Update system statistics display
     */
    updateSystemStatsDisplay(stats) {
        // Update uptime
        const uptimeElement = document.getElementById('uptime-display');
        if (uptimeElement && stats.uptime_seconds) {
            const hours = (stats.uptime_seconds / 3600).toFixed(1);
            uptimeElement.textContent = `${hours}h`;
        }

        // Update active databases count
        const activeDbElement = document.querySelector('[data-stat="active-databases"]');
        if (activeDbElement && stats.active_databases !== undefined) {
            activeDbElement.textContent = stats.active_databases;
        }

        // Update background thread status indicators
        this.updateStatusIndicators(stats.background_threads || {});
    }

    /**
     * Update database grid with fresh data
     */
    updateDatabaseGrid(databases) {
        const gridContainer = document.querySelector('.database-grid');
        if (!gridContainer || !databases) return;

        databases.forEach(db => {
            const card = gridContainer.querySelector(`[data-db-name="${db.name}"]`);
            if (card) {
                // Update document count
                const docCountElement = card.querySelector('.db-stat-value[data-stat="documents"]');
                if (docCountElement) {
                    docCountElement.textContent = db.documents || 0;
                }

                // Update chunk count
                const chunkCountElement = card.querySelector('.db-stat-value[data-stat="chunks"]');
                if (chunkCountElement) {
                    chunkCountElement.textContent = db.chunks || 0;
                }
            }
        });
    }

    /**
     * Update status indicators
     */
    updateStatusIndicators(threadStatus) {
        const indicators = {
            'registry-sync': threadStatus.registry_sync_running,
            'cleanup': threadStatus.cleanup_running,
            'health-check': threadStatus.health_check_running
        };

        Object.entries(indicators).forEach(([name, isRunning]) => {
            const indicator = document.querySelector(`[data-status="${name}"]`);
            if (indicator) {
                indicator.className = `status-indicator ${isRunning ? '' : 'warning'}`;
            }
        });
    }

    /**
     * Setup query form handlers
     */
    setupQueryHandlers() {
        const queryForm = document.getElementById('query-form');
        if (queryForm) {
            queryForm.addEventListener('submit', async (event) => {
                event.preventDefault();
                await this.executeQuery(queryForm);
            });
        }

        // Setup real-time query validation
        const queryInput = document.getElementById('query-input');
        if (queryInput) {
            queryInput.addEventListener('input', this.debounce(() => {
                this.validateQuery(queryInput.value);
            }, 500));
        }
    }

    /**
     * Execute search query
     */
    async executeQuery(form) {
        const formData = new FormData(form);
        const dbName = formData.get('database');
        const queryText = formData.get('query');
        const searchType = formData.get('search_type') || 'hybrid';
        const limit = parseInt(formData.get('limit')) || 10;

        if (!dbName || !queryText.trim()) {
            this.showAlert('Please select a database and enter a query', 'warning');
            return;
        }

        try {
            showLoading();

            // Record query in history
            this.addToQueryHistory({dbName, queryText, searchType, limit});

            const startTime = Date.now();
            const results = await this.apiClient.query(dbName, queryText, {
                searchType,
                limit,
                scoreThreshold: parseFloat(formData.get('score_threshold')) || 0.0,
                vectorWeight: parseFloat(formData.get('vector_weight')) || 0.7
            });

            const queryTime = Date.now() - startTime;

            hideLoading();
            this.displayQueryResults(results, queryTime);

        } catch (error) {
            hideLoading();
            this.showAlert(`Query failed: ${error.message}`, 'error');
            console.error('Query error:', error);
        }
    }

    /**
     * Display query results
     */
    displayQueryResults(results, queryTime) {
        const resultsContainer = document.getElementById('query-results');
        if (!resultsContainer) return;

        const {results: documents, search_type, total_results} = results;

        let html = `
            <div class="results-header">
                <h3>Search Results</h3>
                <div class="results-meta">
                    <span class="result-count">${total_results} results</span>
                    <span class="search-type">Search: ${search_type}</span>
                    <span class="query-time">${queryTime}ms</span>
                </div>
            </div>
        `;

        if (documents && documents.length > 0) {
            html += '<div class="results-list">';

            documents.forEach((doc, index) => {
                html += `
                    <div class="result-item" data-doc-id="${doc.id}">
                        <div class="result-header">
                            <span class="result-rank">#${index + 1}</span>
                            <span class="result-score">Score: ${doc.score.toFixed(4)}</span>
                            <span class="result-id">ID: ${doc.id}</span>
                        </div>
                        <div class="result-content">
                            ${this.highlightSearchTerms(doc.content, 300)}
                        </div>
                        <div class="result-metadata">
                            ${this.formatMetadata(doc.metadata)}
                        </div>
                        <div class="result-actions">
                            <button class="btn btn-sm btn-primary" onclick="InspectorApp.showDocumentDetail('${doc.id}')">
                                View Document
                            </button>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
        } else {
            html += '<div class="no-results">No results found for your query.</div>';
        }

        resultsContainer.innerHTML = html;
        resultsContainer.scrollIntoView({behavior: 'smooth'});
    }

    /**
     * Setup document browser with pagination and filtering
     */
    setupDocumentBrowser() {
        const browserContainer = document.getElementById('document-browser');
        if (!browserContainer) return;

        // Initialize pagination
        this.currentPage = 1;
        this.pageSize = 20;
        this.totalDocuments = 0;

        // Load initial documents
        this.loadDocuments();

        // Setup filters
        this.setupDocumentFilters();
    }

    /**
     * Load documents with pagination
     */
    async loadDocuments(page = 1, filters = {}) {
        const dbName = this.getCurrentDatabase();
        if (!dbName) return;

        try {
            showLoading();

            const params = {
                page,
                limit: this.pageSize,
                ...filters
            };

            const response = await this.apiClient.listDocuments(dbName, params);

            hideLoading();
            this.displayDocuments(response.documents);
            this.updatePagination(response.pagination);

        } catch (error) {
            hideLoading();
            this.showAlert(`Failed to load documents: ${error.message}`, 'error');
        }
    }

    /**
     * Display documents in the browser
     */
    displayDocuments(documents) {
        const container = document.getElementById('documents-list');
        if (!container) return;

        if (!documents || documents.length === 0) {
            container.innerHTML = '<div class="no-documents">No documents found.</div>';
            return;
        }

        let html = '';
        documents.forEach(doc => {
            html += `
                <div class="document-item" data-doc-id="${doc.id}">
                    <div class="document-header">
                        <h4 class="document-id">${doc.id}</h4>
                        <div class="document-meta">
                            <span class="doc-length">${doc.content.length} chars</span>
                            ${doc.updated_at ? `<span class="doc-date">${new Date(doc.updated_at).toLocaleDateString()}</span>` : ''}
                        </div>
                    </div>
                    <div class="document-preview">
                        ${doc.content.substring(0, 200)}${doc.content.length > 200 ? '...' : ''}
                    </div>
                    <div class="document-metadata">
                        ${this.formatMetadata(doc.metadata)}
                    </div>
                    <div class="document-actions">
                        <button class="btn btn-sm btn-primary" onclick="InspectorApp.showDocumentDetail('${doc.id}')">
                            View Details
                        </button>
                        <button class="btn btn-sm btn-secondary" onclick="InspectorApp.editDocument('${doc.id}')">
                            Edit
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="InspectorApp.deleteDocument('${doc.id}')">
                            Delete
                        </button>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    }

    /**
     * Utility methods
     */
    getCurrentDatabase() {
        // Extract database name from URL or page context
        const path = window.location.pathname;
        const match = path.match(/\/inspector\/database\/([^\/]+)/);
        return match ? match[1] : null;
    }

    highlightSearchTerms(text, maxLength = 300) {
        // Basic text truncation and highlighting
        let truncated = text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
        return this.escapeHtml(truncated);
    }

    formatMetadata(metadata) {
        if (!metadata || Object.keys(metadata).length === 0) {
            return '<span class="no-metadata">No metadata</span>';
        }

        return Object.entries(metadata)
            .slice(0, 3) // Show first 3 metadata fields
            .map(([key, value]) => `<span class="metadata-tag">${key}: ${value}</span>`)
            .join(' ');
    }

    formatMetadataDetailed(metadata) {
        if (!metadata || Object.keys(metadata).length === 0) {
            return '<div class="no-metadata">No metadata available</div>';
        }

        let html = '<div class="metadata-grid">';
        Object.entries(metadata).forEach(([key, value]) => {
            html += `
                <div class="metadata-row">
                    <div class="metadata-key">${this.escapeHtml(key)}</div>
                    <div class="metadata-value">${this.escapeHtml(JSON.stringify(value))}</div>
                </div>
            `;
        });
        html += '</div>';
        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showAlert(message, type = 'info') {
        if (typeof showAlert === 'function') {
            showAlert(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

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
    }

    /**
     * Query history management
     */
    addToQueryHistory(query) {
        const history = this.getQueryHistory();
        history.unshift({
            ...query,
            timestamp: Date.now()
        });

        // Keep only last 50 queries
        const limitedHistory = history.slice(0, 50);
        localStorage.setItem('inspector_query_history', JSON.stringify(limitedHistory));
    }

    getQueryHistory() {
        try {
            const history = localStorage.getItem('inspector_query_history');
            return history ? JSON.parse(history) : [];
        } catch (error) {
            console.warn('Failed to load query history:', error);
            return [];
        }
    }

    /**
     * Background update management
     */
    startBackgroundRefresh() {
        this.stopBackgroundRefresh(); // Clear any existing timer

        if (this.currentPage === 'dashboard') {
            this.refreshTimer = setInterval(() => {
                if (!document.hidden) {
                    this.setupDashboardUpdates();
                }
            }, this.config.refreshInterval);
        }
    }

    stopBackgroundRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }

        if (this.dashboardUpdateInterval) {
            clearInterval(this.dashboardUpdateInterval);
            this.dashboardUpdateInterval = null;
        }
    }

    pauseBackgroundUpdates() {
        this.stopBackgroundRefresh();
    }

    resumeBackgroundUpdates() {
        if (this.currentPage === 'dashboard') {
            this.startBackgroundRefresh();
        }
    }

    /**
     * Window resize handler
     */
    handleWindowResize() {
        // Refresh charts if on embeddings page
        if (this.currentPage === 'embeddings' && this.charts) {
            Object.values(this.charts).forEach(chart => {
                if (chart && typeof chart.resize === 'function') {
                    chart.resize();
                }
            });
        }
    }

    /**
     * Handle page navigation
     */
    handlePageNavigation(target) {
        window.location.href = target;
    }

    /**
     * Handle form submissions
     */
    handleFormSubmission(form) {
        const formType = form.dataset.inspectorForm;

        switch (formType) {
            case 'query':
                this.executeQuery(form);
                break;
            case 'document-edit':
                this.handleDocumentEdit(form);
                break;
            default:
                console.warn('Unknown form type:', formType);
        }
    }

    /**
     * Load dynamic content
     */
    loadDynamicContent(trigger) {
        const contentType = trigger.dataset.loadContent;
        const target = trigger.dataset.target;

        switch (contentType) {
            case 'documents':
                this.loadDocuments();
                break;
            case 'stats':
                this.loadSystemStats();
                break;
            default:
                console.warn('Unknown content type:', contentType);
        }
    }

    /**
     * Setup document filters
     */
    setupDocumentFilters() {
        const filterForm = document.getElementById('document-filters');
        if (filterForm) {
            filterForm.addEventListener('submit', (event) => {
                event.preventDefault();
                const formData = new FormData(filterForm);
                const filters = {
                    search: formData.get('search'),
                    metadata: formData.get('metadata_filter')
                };
                this.loadDocuments(1, filters);
            });
        }
    }

    /**
     * Update pagination display
     */
    updatePagination(pagination) {
        const paginationContainer = document.getElementById('pagination');
        if (!paginationContainer || !pagination) return;

        const {current_page, total_pages, total_items} = pagination;

        let html = '<div class="pagination-info">';
        html += `<span>Page ${current_page} of ${total_pages} (${total_items} total)</span>`;
        html += '</div>';

        if (total_pages > 1) {
            html += '<div class="pagination-nav">';

            // Previous button
            if (current_page > 1) {
                html += `<button class="btn btn-sm btn-secondary" onclick="InspectorApp.loadDocuments(${current_page - 1})">&laquo; Previous</button>`;
            }

            // Page numbers
            const startPage = Math.max(1, current_page - 2);
            const endPage = Math.min(total_pages, current_page + 2);

            for (let i = startPage; i <= endPage; i++) {
                const isActive = i === current_page ? ' active' : '';
                html += `<button class="btn btn-sm btn-outline-primary${isActive}" onclick="InspectorApp.loadDocuments(${i})">${i}</button>`;
            }

            // Next button
            if (current_page < total_pages) {
                html += `<button class="btn btn-sm btn-secondary" onclick="InspectorApp.loadDocuments(${current_page + 1})">Next &raquo;</button>`;
            }

            html += '</div>';
        }

        paginationContainer.innerHTML = html;
    }

    /**
     * Setup pagination handlers
     */
    setupPagination() {
        // Pagination is handled dynamically in updatePagination
    }

    /**
     * Setup document modals
     */
    setupDocumentModals() {
        // Document modals are handled by the showDocumentDetail static method
    }

    /**
     * Initialize dashboard widgets
     */
    initializeDashboardWidgets() {
        // Dashboard widgets are initialized automatically
    }

    /**
     * Load query history
     */
    loadQueryHistory() {
        const history = this.getQueryHistory();
        const historyContainer = document.getElementById('query-history');

        if (historyContainer && history.length > 0) {
            let html = '<h5>Recent Queries</h5><div class="query-history-list">';

            history.slice(0, 10).forEach((query, index) => {
                html += `
                    <div class="history-item" onclick="InspectorApp.loadHistoryQuery(${index})">
                        <div class="history-query">${query.queryText}</div>
                        <div class="history-meta">
                            <span class="history-db">${query.dbName}</span>
                            <span class="history-time">${new Date(query.timestamp).toLocaleString()}</span>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
            historyContainer.innerHTML = html;
        }
    }

    /**
     * Setup query auto-complete
     */
    setupQueryAutoComplete() {
        // Auto-complete can be implemented later
    }

    /**
     * Setup visualization controls
     */
    setupVisualizationControls() {
        // Visualization controls setup
    }

    /**
     * Initialize charts
     */
    initializeCharts() {
        // Charts initialization
    }

    /**
     * Setup system monitoring
     */
    setupSystemMonitoring() {
        // System monitoring setup
    }

    /**
     * Setup admin controls
     */
    setupAdminControls() {
        // Admin controls setup
    }

    /**
     * Validate query input
     */
    validateQuery(query) {
        // Query validation logic
    }

    /**
     * Cleanup method
     */
    destroy() {
        this.stopBackgroundRefresh();

        // Remove event listeners
        this.eventListeners.forEach((listener, element) => {
            element.removeEventListener(listener.event, listener.handler);
        });
        this.eventListeners.clear();

        console.log('LocalVectorDB Inspector destroyed');
    }
}

// Global instance - will be initialized from base.html
window.InspectorApp = null;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Initialization will be called from base.html with config
    });
} else {
    // DOM already loaded
    // Initialization will be called from base.html with config
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InspectorApp;
}