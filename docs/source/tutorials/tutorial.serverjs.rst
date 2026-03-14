========================================================================
JavaScript Search App Tutorial: Building a Simple Document Search Engine
========================================================================

This tutorial will guide you through building a simple web-based document search engine using the LocalVectorDB server API. You'll create a single-page application that can search through documents using vector similarity, keyword search, or hybrid search - all without writing any Python code!

What We'll Build
================

We'll create a clean, responsive web application that:

* Connects to a LocalVectorDB server via REST API
* Provides a search interface with different search types
* Displays search results with relevance scores
* Shows document metadata and content previews
* Includes basic error handling and loading states

No Python knowledge required - just HTML, CSS, and JavaScript!

Prerequisites
=============

Before starting, ensure you have:

* Python 3.12 or higher
* A web browser (Chrome, Firefox, Safari, etc.)
* Basic knowledge of HTML, CSS, and JavaScript
* A text editor or IDE

Setting Up the Server
=====================

First, let's set up and start the LocalVectorDB server using the command-line interface.

Install and Configure LocalVectorDB
------------------------------------

.. code-block:: bash

   # Install LocalVectorDB (requires Python 3.12+)
   pip install localvectordb

   # Create a basic configuration file
   lvdb config init --interactive

During the interactive setup, choose these options:

* **Configuration file path**: `./.lvdb-config.toml` (default)
* **Server host**: `127.0.0.1`
* **Server port**: `5000`
* **Database directory**: `./search-demo-db`
* **Enable CORS**: `Yes` (important for web applications!)
* **CORS origins**: `localhost` or `all` (for development **only**)
* **API Authentication**: `No` (to keep it simple)

Create Sample Documents and Database
------------------------------------

Let's create some sample documents to search through. First, create a folder for our documents:

.. code-block:: bash

   # Create a folder for sample documents
   mkdir server-docs

   # Copy some files that you'd like to search
   cp /path/to/file server-docs/file
   # ...etc.

Now create the database and add all the documents:

.. code-block:: bash

   # Create a new database for our demo
   lvdb create search_demo

   # Add all documents from the server-docs folder
   # NOTE: only plaintext files can be added from the cli this way. You must convert PDFs or DOCX files first.
   lvdb db search_demo add "server-docs/*.txt"

Start the Server
----------------

.. code-block:: bash

   # Start the LocalVectorDB server
   lvdb serve

You should see output indicating the server is running on `http://127.0.0.1:5000`. Keep this terminal window open while developing the web application.

Creating the Web Application
============================

Now let's build our single-page search application. We'll create three files: HTML for structure, CSS for styling, and JavaScript for functionality.

HTML Structure (index.html)
----------------------------

.. code-block:: html

   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>LocalVectorDB Search Demo</title>
       <link rel="stylesheet" href="styles.css">
   </head>
   <body>
       <div class="container">
           <!-- Header -->
           <header class="header">
               <h1>Document Search Engine</h1>
               <p>Powered by LocalVectorDB</p>
           </header>

           <!-- Search Form -->
           <section class="search-section">
               <form id="searchForm" class="search-form">
                   <div class="search-input-container">
                       <input
                           type="text"
                           id="searchInput"
                           placeholder="Enter your search query..."
                           required
                           autocomplete="off"
                       >
                       <button type="submit" id="searchButton">
                           <span class="button-text">Search</span>
                           <span class="spinner" id="spinner">⟳</span>
                       </button>
                   </div>

                   <div class="search-options">
                       <label class="option-group">
                           <span>Search Type:</span>
                           <select id="searchType">
                               <option value="hybrid">Hybrid (Best Results)</option>
                               <option value="vector">Vector Similarity</option>
                               <option value="keyword">Keyword Search</option>
                           </select>
                       </label>

                       <label class="option-group">
                           <span>Max Results:</span>
                           <select id="maxResults">
                               <option value="5">5</option>
                               <option value="10" selected>10</option>
                               <option value="20">20</option>
                           </select>
                       </label>
                   </div>
               </form>
           </section>

           <!-- Results Section -->
           <section class="results-section">
               <div id="resultsHeader" class="results-header hidden">
                   <h2>Search Results</h2>
                   <span id="resultsCount" class="results-count"></span>
               </div>

               <div id="resultsContainer" class="results-container">
                   <!-- Results will be inserted here -->
               </div>

               <div id="noResults" class="no-results hidden">
                   <h3>No results found</h3>
                   <p>Try adjusting your search terms or using a different search type.</p>
               </div>

               <div id="errorMessage" class="error-message hidden">
                   <h3>Something went wrong</h3>
                   <p id="errorText">Please check if the LocalVectorDB server is running.</p>
               </div>
           </section>

           <!-- Footer -->
           <footer class="footer">
               <p>
                   Built with
                   <a href="https://github.com/thomas-villani/localvectordb" target="_blank">LocalVectorDB</a>
                   • Search powered by AI embeddings
               </p>
           </footer>
       </div>

       <script src="script.js"></script>
   </body>
   </html>

CSS Styling (styles.css)
-------------------------

.. code-block:: css

   /* Reset and base styles */
   * {
       margin: 0;
       padding: 0;
       box-sizing: border-box;
   }

   body {
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       line-height: 1.6;
       color: #333;
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       min-height: 100vh;
   }

   .container {
       max-width: 800px;
       margin: 0 auto;
       padding: 20px;
       min-height: 100vh;
       display: flex;
       flex-direction: column;
   }

   /* Header */
   .header {
       text-align: center;
       margin-bottom: 40px;
       color: white;
   }

   .header h1 {
       font-size: 2.5rem;
       margin-bottom: 10px;
       text-shadow: 0 2px 4px rgba(0,0,0,0.3);
   }

   .header p {
       font-size: 1.1rem;
       opacity: 0.9;
   }

   /* Search Section */
   .search-section {
       background: white;
       border-radius: 12px;
       padding: 30px;
       box-shadow: 0 8px 32px rgba(0,0,0,0.1);
       margin-bottom: 30px;
   }

   .search-form {
       display: flex;
       flex-direction: column;
       gap: 20px;
   }

   .search-input-container {
       display: flex;
       gap: 12px;
       align-items: center;
   }

   #searchInput {
       flex: 1;
       padding: 16px 20px;
       border: 2px solid #e1e5e9;
       border-radius: 8px;
       font-size: 16px;
       transition: border-color 0.3s ease;
   }

   #searchInput:focus {
       outline: none;
       border-color: #667eea;
       box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
   }

   #searchButton {
       padding: 16px 24px;
       background: #667eea;
       color: white;
       border: none;
       border-radius: 8px;
       font-size: 16px;
       font-weight: 600;
       cursor: pointer;
       transition: all 0.3s ease;
       position: relative;
       min-width: 100px;
   }

   #searchButton:hover {
       background: #5a6fd8;
       transform: translateY(-1px);
   }

   #searchButton:disabled {
       background: #94a3b8;
       cursor: not-allowed;
       transform: none;
   }

   .spinner {
       display: none;
       animation: spin 1s linear infinite;
   }

   .searching .button-text {
       display: none;
   }

   .searching .spinner {
       display: inline;
   }

   @keyframes spin {
       0% { transform: rotate(0deg); }
       100% { transform: rotate(360deg); }
   }

   .search-options {
       display: flex;
       gap: 20px;
       flex-wrap: wrap;
   }

   .option-group {
       display: flex;
       flex-direction: column;
       gap: 8px;
       font-weight: 500;
   }

   .option-group select {
       padding: 8px 12px;
       border: 2px solid #e1e5e9;
       border-radius: 6px;
       font-size: 14px;
       background: white;
   }

   /* Results Section */
   .results-section {
       flex: 1;
   }

   .results-header {
       display: flex;
       justify-content: space-between;
       align-items: center;
       margin-bottom: 20px;
       color: white;
   }

   .results-header h2 {
       font-size: 1.5rem;
   }

   .results-count {
       background: rgba(255,255,255,0.2);
       padding: 8px 16px;
       border-radius: 20px;
       font-size: 14px;
       font-weight: 500;
   }

   .results-container {
       display: flex;
       flex-direction: column;
       gap: 16px;
   }

   .result-card {
       background: white;
       border-radius: 8px;
       padding: 24px;
       box-shadow: 0 4px 16px rgba(0,0,0,0.1);
       transition: transform 0.2s ease, box-shadow 0.2s ease;
   }

   .result-card:hover {
       transform: translateY(-2px);
       box-shadow: 0 8px 24px rgba(0,0,0,0.15);
   }

   .result-header {
       display: flex;
       justify-content: space-between;
       align-items: flex-start;
       margin-bottom: 12px;
   }

   .result-title {
       font-size: 1.1rem;
       font-weight: 600;
       color: #1e293b;
   }

   .result-score {
       background: #667eea;
       color: white;
       padding: 4px 12px;
       border-radius: 12px;
       font-size: 12px;
       font-weight: 600;
   }

   .result-content {
       color: #64748b;
       line-height: 1.7;
       margin-bottom: 12px;
   }

   .result-metadata {
       display: flex;
       gap: 16px;
       font-size: 12px;
       color: #94a3b8;
   }

   .metadata-item {
       display: flex;
       align-items: center;
       gap: 4px;
   }

   /* Empty states */
   .no-results, .error-message {
       text-align: center;
       padding: 60px 20px;
       color: white;
   }

   .no-results-icon, .error-icon {
       font-size: 4rem;
       margin-bottom: 20px;
   }

   .no-results h3, .error-message h3 {
       font-size: 1.5rem;
       margin-bottom: 12px;
   }

   .no-results p, .error-message p {
       opacity: 0.8;
       font-size: 1.1rem;
   }

   /* Footer */
   .footer {
       text-align: center;
       margin-top: 40px;
       padding: 20px;
       color: rgba(255,255,255,0.8);
   }

   .footer a {
       color: white;
       text-decoration: none;
       font-weight: 500;
   }

   .footer a:hover {
       text-decoration: underline;
   }

   /* Utility classes */
   .hidden {
       display: none !important;
   }

   /* Responsive design */
   @media (max-width: 600px) {
       .container {
           padding: 16px;
       }

       .header h1 {
           font-size: 2rem;
       }

       .search-section {
           padding: 20px;
       }

       .search-input-container {
           flex-direction: column;
       }

       #searchButton {
           width: 100%;
       }

       .search-options {
           flex-direction: column;
       }

       .results-header {
           flex-direction: column;
           gap: 12px;
           align-items: flex-start;
       }

       .result-header {
           flex-direction: column;
           gap: 8px;
           align-items: flex-start;
       }

       .result-metadata {
           flex-direction: column;
           gap: 8px;
       }
   }

JavaScript Functionality (script.js)
------------------------------------

.. code-block:: javascript

   // Configuration
   const API_BASE_URL = 'http://127.0.0.1:5000/api/v1';
   const DATABASE_NAME = 'search_demo';

   // DOM Elements
   const searchForm = document.getElementById('searchForm');
   const searchInput = document.getElementById('searchInput');
   const searchButton = document.getElementById('searchButton');
   const searchType = document.getElementById('searchType');
   const maxResults = document.getElementById('maxResults');
   const resultsHeader = document.getElementById('resultsHeader');
   const resultsCount = document.getElementById('resultsCount');
   const resultsContainer = document.getElementById('resultsContainer');
   const noResults = document.getElementById('noResults');
   const errorMessage = document.getElementById('errorMessage');
   const errorText = document.getElementById('errorText');

   // State
   let isSearching = false;

   // Initialize the application
   document.addEventListener('DOMContentLoaded', function() {
       console.log('LocalVectorDB Search Demo initialized');

       // Check server connection on startup
       checkServerConnection();

       // Set up event listeners
       searchForm.addEventListener('submit', handleSearch);
       searchInput.addEventListener('keyup', handleSearchInputChange);

       // Focus on search input
       searchInput.focus();
   });

   /**
    * Check if the LocalVectorDB server is accessible
    */
   async function checkServerConnection() {
       try {
           const response = await fetch(`${API_BASE_URL}/health`);
           if (response.ok) {
               console.log('✅ Server connection successful');
           } else {
               throw new Error(`Server responded with status: ${response.status}`);
           }
       } catch (error) {
           console.error('❌ Server connection failed:', error);
           showError('Cannot connect to LocalVectorDB server. Please ensure it is running on http://127.0.0.1:5000');
       }
   }

   /**
    * Handle search form submission
    */
   async function handleSearch(event) {
       event.preventDefault();

       if (isSearching) return;

       const query = searchInput.value.trim();
       if (!query) {
           searchInput.focus();
           return;
       }

       await performSearch(query);
   }

   /**
    * Handle search input changes
    */
   function handleSearchInputChange(event) {
       // Enable search on Enter key
       if (event.key === 'Enter') {
           handleSearch(event);
       }
   }

   /**
    * Perform the actual search
    */
   async function performSearch(query) {
       setSearchingState(true);
       hideAllStates();

       try {
           console.log(`Searching for: "${query}"`);

           const searchParams = {
               query: query,
               search_type: searchType.value,
               return_type: 'documents',
               k: parseInt(maxResults.value),
               score_threshold: 0.0
           };

           const response = await fetch(`${API_BASE_URL}/${DATABASE_NAME}/query`, {
               method: 'POST',
               headers: {
                   'Content-Type': 'application/json',
               },
               body: JSON.stringify(searchParams)
           });

           if (!response.ok) {
               const errorData = await response.json().catch(() => ({}));
               throw new Error(errorData.error?.message || `HTTP ${response.status}: ${response.statusText}`);
           }

           const data = await response.json();
           console.log('Search results:', data);

           displayResults(data.results, query);

       } catch (error) {
           console.error('Search failed:', error);
           showError(`Search failed: ${error.message}`);
       } finally {
           setSearchingState(false);
       }
   }

   /**
    * Display search results
    */
   function displayResults(results, query) {
       hideAllStates();

       if (!results || results.length === 0) {
           showNoResults();
           return;
       }

       // Show results header
       resultsHeader.classList.remove('hidden');
       resultsCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''} for "${query}"`;

       // Clear previous results
       resultsContainer.innerHTML = '';

       // Create result cards
       results.forEach((result, index) => {
           const resultCard = createResultCard(result, index + 1);
           resultsContainer.appendChild(resultCard);
       });

       // Scroll to results
       resultsHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
   }

   /**
    * Create a result card element
    */
   function createResultCard(result, position) {
       const card = document.createElement('div');
       card.className = 'result-card';

       // Format score as percentage
       const scorePercent = Math.round(result.score * 100);

       // Truncate content if too long
       const maxContentLength = 300;
       let content = result.content;
       if (content.length > maxContentLength) {
           content = content.substring(0, maxContentLength) + '...';
       }

       card.innerHTML = `
           <div class="result-header">
               <div class="result-title">Document #${position}</div>
               <div class="result-score">${scorePercent}% match</div>
           </div>
           <div class="result-content">${escapeHtml(content)}</div>
           <div class="result-metadata">
               <div class="metadata-item">
                   <span>ID: ${escapeHtml(result.id)}</span>
               </div>
               <div class="metadata-item">
                   <span>Type: ${escapeHtml(result.type)}</span>
               </div>
               <div class="metadata-item">
                   <span>Length: ${result.content.length} chars</span>
               </div>
           </div>
       `;

       return card;
   }

   /**
    * Set searching state
    */
   function setSearchingState(searching) {
       isSearching = searching;

       if (searching) {
           searchButton.classList.add('searching');
           searchButton.disabled = true;
           searchInput.disabled = true;
       } else {
           searchButton.classList.remove('searching');
           searchButton.disabled = false;
           searchInput.disabled = false;
       }
   }

   /**
    * Hide all result states
    */
   function hideAllStates() {
       resultsHeader.classList.add('hidden');
       noResults.classList.add('hidden');
       errorMessage.classList.add('hidden');
       resultsContainer.innerHTML = '';
   }

   /**
    * Show no results state
    */
   function showNoResults() {
       hideAllStates();
       noResults.classList.remove('hidden');
   }

   /**
    * Show error state
    */
   function showError(message) {
       hideAllStates();
       errorText.textContent = message;
       errorMessage.classList.remove('hidden');
   }

   /**
    * Escape HTML to prevent XSS
    */
   function escapeHtml(text) {
       const div = document.createElement('div');
       div.textContent = text;
       return div.innerHTML;
   }

   /**
    * Utility function to format numbers
    */
   function formatNumber(num) {
       return new Intl.NumberFormat().format(num);
   }

   /**
    * Add some helpful keyboard shortcuts
    */
   document.addEventListener('keydown', function(event) {
       // Focus search input when pressing '/' key
       if (event.key === '/' && event.target !== searchInput) {
           event.preventDefault();
           searchInput.focus();
           searchInput.select();
       }

       // Clear search when pressing Escape
       if (event.key === 'Escape') {
           if (document.activeElement === searchInput) {
               searchInput.value = '';
               hideAllStates();
           } else {
               searchInput.focus();
           }
       }
   });

   // Add some helpful console commands for development
   if (typeof window !== 'undefined') {
       window.searchDemo = {
           search: performSearch,
           checkConnection: checkServerConnection,
           config: {
               apiUrl: API_BASE_URL,
               database: DATABASE_NAME
           }
       };

       console.log('Development tools available at window.searchDemo');
   }

Running the Application
=======================

Now let's put it all together and run our search application:

Create the Project Structure
----------------------------

Create your web application files in the same directory where you have your configuration:

.. code-block:: bash

   # You should already be in your main project directory
   # Create the web application files
   touch index.html styles.css script.js

Add the code from the sections above to each respective file.

Start the LocalVectorDB Server
------------------------------

The LocalVectorDB server includes a built-in Flask web server that handles both the API and serves static files:

.. code-block:: bash

   # Start the LocalVectorDB server
   lvdb serve

You should see output indicating the server is running on `http://127.0.0.1:5000`.

Access Your Search Engine
-------------------------

1. Make sure your LocalVectorDB server is running on port 5000
2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Place your `index.html`, `styles.css`, and `script.js` files in the same directory as your config file
4. Navigate to `http://127.0.0.1:5000/index.html`

You should see your search engine! Try searching for terms like:

* "programming language"
* "machine learning"
* "web development"
* "artificial intelligence"

Testing Different Search Types
===============================

Your search engine supports three different search modes:

**Vector Similarity Search**
   Uses AI embeddings to find semantically similar content. Great for finding documents with similar meaning even if they use different words.

**Keyword Search**
   Traditional text search that looks for exact word matches. Fast and precise for finding specific terms.

**Hybrid Search**
   Combines both vector and keyword search for the best of both worlds. Usually provides the most relevant results.

Try the same query with different search types to see how the results differ!

Adding More Documents
=====================

You can easily add more documents to your search engine using the CLI:

.. code-block:: bash

   # Add a single document file
   lvdb db search_demo add /path/to/your/document.txt

   # Add multiple documents from a folder
   lvdb db search_demo add "/path/to/documents/*.txt"

   # Add documents with custom metadata
   lvdb db search_demo add server-docs/new-doc.txt --metadata '{"category":"tutorial","author":"you"}'

   # Add all files from your server-docs folder again (if you add more)
   lvdb db search_demo add "server-docs/*.txt"

Troubleshooting
===============

**CORS Errors**
   Make sure you enabled CORS when configuring the server. You can also add CORS headers manually:

   .. code-block:: bash

      lvdb config set server.cors_enabled true
      lvdb config set server.cors_allowed_origins '["http://localhost:8080"]'

**Server Connection Failed**
   Verify the LocalVectorDB server is running:

   .. code-block:: bash

      # Check if server is responding
      curl http://127.0.0.1:5000/api/v1/health

**No Search Results**
   Make sure you have documents in your database:

   .. code-block:: bash

      lvdb db search_demo list

**JavaScript Errors**
   Open your browser's developer console (F12) to see detailed error messages.

Enhancing the Application
=========================

Here are some ideas to extend your search engine:

**Add Document Upload**

.. code-block:: javascript

   // Add a file input to your HTML
   function handleFileUpload(file) {
       const formData = new FormData();
       formData.append('documents', [file.content]);
       formData.append('metadata', JSON.stringify({
           filename: file.name,
           uploaded_at: new Date().toISOString()
       }));

       fetch(`${API_BASE_URL}/${DATABASE_NAME}/documents`, {
           method: 'POST',
           body: formData
       });
   }

**Add Result Filtering**

.. code-block:: javascript

   // Add metadata filters to your search
   const searchParams = {
       query: query,
       search_type: searchType.value,
       filters: {
           category: selectedCategory,
           author: selectedAuthor
       }
   };

**Add Search History**

.. code-block:: javascript

   // Store searches in localStorage
   function saveSearchHistory(query, resultCount) {
       const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
       history.unshift({ query, resultCount, timestamp: Date.now() });
       localStorage.setItem('searchHistory', JSON.stringify(history.slice(0, 10)));
   }

**Add Real-time Search**

.. code-block:: javascript

   // Debounced search as user types
   let searchTimeout;
   searchInput.addEventListener('input', function() {
       clearTimeout(searchTimeout);
       searchTimeout = setTimeout(() => {
           if (searchInput.value.length > 2) {
               performSearch(searchInput.value);
           }
       }, 500);
   });

Deployment Options
==================

**Static Hosting**
   Deploy your search engine to GitHub Pages, Netlify, or Vercel. Just make sure to update the API URL to point to your deployed server.

**Docker Container**
   Package both the server and web app in a Docker container for easy deployment.

**Cloud Deployment**
   Deploy the LocalVectorDB server to cloud platforms like Railway, Render, or DigitalOcean.

Conclusion
==========

Congratulations! You've built a complete document search engine using LocalVectorDB and vanilla JavaScript. Your application demonstrates:

**Core Features**
- Vector similarity search with AI embeddings
- Multiple search types (vector, keyword, hybrid)
- Clean, responsive user interface
- Real-time search capabilities
- Error handling and loading states

**Technical Skills**
- REST API integration
- Modern JavaScript (async/await, fetch)
- Responsive CSS design
- DOM manipulation
- User experience best practices

**LocalVectorDB Integration**
- Server configuration and management
- Document ingestion via CLI
- RESTful API usage
- Search result processing

This foundation can be extended into more sophisticated applications like document management systems, knowledge bases,
or AI-powered search engines. The modular design makes it easy to add features like authentication, file uploads,
advanced filtering, and more.

Happy searching!
