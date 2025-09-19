# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/inspector.py
"""
LocalVectorDB Inspector UI

Web-based interface for inspecting LocalVectorDB databases, testing queries,
visualizing embeddings, and managing the system.
"""
import functools
import logging
from datetime import datetime

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, session, url_for

from localvectordb_server._auth import validate_api_key
from localvectordb_server._error_handlers import handle_errors
from localvectordb_server._logcfg import log_performance
from localvectordb_server.keymanager import PermissionLevel

logger = logging.getLogger(__name__)

# Create inspector blueprint
inspector_bp = Blueprint(
    'inspector',
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/inspector/static'
)


def inspector_enabled():
    """Check if inspector is enabled in configuration"""
    config = getattr(current_app, 'config_obj', None)
    if config and hasattr(config.server, 'inspector_enabled'):
        return config.server.inspector_enabled
    # Default to disabled if not configured
    return False


def require_inspector_auth(required_permission=PermissionLevel.READ_ONLY):
    """Decorator to require authentication for inspector routes with permission checking

    Parameters
    ----------
    required_permission : PermissionLevel
        The minimum permission level required (READ_ONLY or READ_WRITE)
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not inspector_enabled():
                return render_template('inspector_disabled.html'), 503

            # Check if API key authentication is required
            config = getattr(current_app, 'config_obj', None)
            if config and config.server.security.require_api_key:
                # For web interface, check session or require API key parameter
                api_key = request.args.get('api_key')
                if not api_key:
                    return redirect(url_for('inspector.login'))

                # Validate API key and check permissions
                try:
                    key_manager = getattr(current_app, 'key_manager', None)
                    if key_manager:
                        is_valid, permission_level, key_id = key_manager.validate_key_with_permissions(api_key)
                        if not is_valid:
                            flash('Invalid API key', 'error')
                            return redirect(url_for('inspector.login'))

                        # Check if user has required permission
                        if required_permission == PermissionLevel.READ_WRITE and permission_level == PermissionLevel.READ_ONLY:
                            flash('Insufficient permissions. This action requires write access.', 'error')
                            return redirect(url_for('inspector.dashboard'))

                        # Store key_id and permission level in session (NOT the raw key)
                        session['inspector_key_id'] = key_id
                        session['inspector_permission_level'] = permission_level.value
                    else:
                        # Fall back to simple validation if new method not available
                        if not validate_api_key(api_key):
                            flash('Invalid API key', 'error')
                            return redirect(url_for('inspector.login'))
                        session['inspector_api_key'] = api_key
                except Exception as e:
                    logger.error(f"API key validation error: {e}")
                    flash('Authentication error', 'error')
                    return redirect(url_for('inspector.login'))

            return f(*args, **kwargs)

        return decorated_function

    return decorator


@inspector_bp.route('/')
@require_inspector_auth(PermissionLevel.READ_ONLY)
@log_performance("inspector_dashboard")
def dashboard():
    """Main dashboard showing system overview"""
    try:
        # Get system statistics
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            flash('Database manager not available', 'error')
            return render_template('error.html', error="Database manager not available")

        # Get list of databases
        databases = db_manager.list_databases()

        # Get database statistics
        db_stats = []
        for db_name in databases:
            try:
                db = db_manager.get_db(db_name)
                stats = db.get_stats()
                stats['name'] = db_name
                db_stats.append(stats)
            except Exception as e:
                logger.error(f"Error getting stats for database {db_name}: {e}")
                db_stats.append({
                    'name': db_name,
                    'error': str(e),
                    'documents': 0,
                    'chunks': 0
                })

        # Get system information
        manager_stats = db_manager.get_manager_stats()

        return render_template('dashboard.html',
                               databases=db_stats,
                               system_stats=manager_stats,
                               total_databases=len(databases))

    except Exception as e:
        logger.error(f"Error in dashboard: {e}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('error.html', error=str(e))


@inspector_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page for API key authentication"""
    if not inspector_enabled():
        return render_template('inspector_disabled.html'), 503

    if request.method == 'POST':
        api_key = request.form.get('api_key')
        if not api_key:
            flash('API key is required', 'error')
            return render_template('login.html')

        # Validate API key and get permissions
        try:
            key_manager = getattr(current_app, 'key_manager', None)
            if key_manager:
                is_valid, permission_level, key_id = key_manager.validate_key_with_permissions(api_key)
                if is_valid:
                    # Store key_id and permission level in session (NOT the raw key)
                    session['inspector_key_id'] = key_id
                    session['inspector_permission_level'] = permission_level.value if permission_level else 'read_write'
                    # Note: NOT storing the raw API key in session anymore
                    return redirect(url_for('inspector.dashboard'))
                else:
                    flash('Invalid API key', 'error')
            else:
                flash('Key manager not available', 'error')
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Authentication error', 'error')

    return render_template('login.html')


@inspector_bp.route('/logout')
def logout():
    """Logout and clear session"""
    session.pop('inspector_api_key', None)  # Remove if still present (backward compat)
    session.pop('inspector_key_id', None)
    session.pop('inspector_permission_level', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('inspector.login'))


@inspector_bp.route('/database/<db_name>')
@require_inspector_auth(PermissionLevel.READ_ONLY)
@log_performance("inspector_database")
def database_detail(db_name):
    """Database detail view with document browser"""
    try:
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            flash('Database manager not available', 'error')
            return redirect(url_for('inspector.dashboard'))

        # Get database instance
        db = db_manager.get_db(db_name)

        # Get database statistics and metadata
        stats = db.get_stats()
        schema_info = db.get_metadata_schema_info()

        # Get configuration
        config_info = {
            'name': db.name,
            'embedding_provider': db.embedding_provider.provider_name,
            'embedding_model': db.embedding_provider.model,
            'embedding_dimension': db.embedding_dimension,
            'chunking_method': db.chunking_method,
            'chunk_size': db.chunk_size,
            'chunk_overlap': db.chunk_overlap,
            'fts_enabled': db.fts_enabled
        }

        return render_template('database.html',
                               db_name=db_name,
                               stats=stats,
                               schema_info=schema_info,
                               config=config_info)

    except Exception as e:
        logger.error(f"Error in database detail for {db_name}: {e}")
        flash(f'Error loading database {db_name}: {str(e)}', 'error')
        return redirect(url_for('inspector.dashboard'))


@inspector_bp.route('/query')
@require_inspector_auth(PermissionLevel.READ_ONLY)
@log_performance("inspector_query")
def query_interface():
    """Interactive query testing interface"""
    try:
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            flash('Database manager not available', 'error')
            return redirect(url_for('inspector.dashboard'))

        # Get list of databases for query testing
        databases = db_manager.list_databases()

        return render_template('query.html', databases=databases)

    except Exception as e:
        logger.error(f"Error in query interface: {e}")
        flash(f'Error loading query interface: {str(e)}', 'error')
        return redirect(url_for('inspector.dashboard'))


@inspector_bp.route('/embeddings')
@require_inspector_auth(PermissionLevel.READ_ONLY)
@log_performance("inspector_embeddings")
def embeddings_view():
    """Embedding visualization interface"""
    try:
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            flash('Database manager not available', 'error')
            return redirect(url_for('inspector.dashboard'))

        # Get list of databases for embedding analysis
        databases = db_manager.list_databases()

        return render_template('embeddings.html', databases=databases)

    except Exception as e:
        logger.error(f"Error in embeddings view: {e}")
        flash(f'Error loading embeddings view: {str(e)}', 'error')
        return redirect(url_for('inspector.dashboard'))


@inspector_bp.route('/admin')
@require_inspector_auth(PermissionLevel.READ_ONLY)
@log_performance("inspector_admin")
def admin_interface():
    """System administration interface"""
    try:
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            flash('Database manager not available', 'error')
            return redirect(url_for('inspector.dashboard'))

        # Get system statistics and configuration
        manager_stats = db_manager.get_manager_stats()
        config = getattr(current_app, 'config_obj', None)

        # Get API keys information (if available)
        api_keys_info = []
        key_manager = getattr(current_app, 'key_manager', None)
        if key_manager:
            try:
                api_keys_info = key_manager.list_keys()
            except Exception as e:
                logger.error(f"Error getting API keys: {e}")

        return render_template('admin.html',
                               manager_stats=manager_stats,
                               config=config,
                               api_keys=api_keys_info)

    except Exception as e:
        logger.error(f"Error in admin interface: {e}")
        flash(f'Error loading admin interface: {str(e)}', 'error')
        return redirect(url_for('inspector.dashboard'))


@inspector_bp.route('/api/databases')
@require_inspector_auth(PermissionLevel.READ_ONLY)
@handle_errors
def api_databases():
    """API endpoint to get database list with stats"""
    try:
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            return jsonify({'error': 'Database manager not available'}), 500

        databases = db_manager.list_databases()
        db_stats = []

        for db_name in databases:
            try:
                db = db_manager.get_db(db_name)
                stats = db.get_stats()
                stats['name'] = db_name
                db_stats.append(stats)
            except Exception as e:
                logger.error(f"Error getting stats for database {db_name}: {e}")
                db_stats.append({
                    'name': db_name,
                    'error': str(e),
                    'documents': 0,
                    'chunks': 0
                })

        return jsonify({'databases': db_stats})

    except Exception as e:
        logger.error(f"Error in api_databases: {e}")
        return jsonify({'error': str(e)}), 500


@inspector_bp.route('/api/system/stats')
@require_inspector_auth(PermissionLevel.READ_ONLY)
@handle_errors
def api_system_stats():
    """API endpoint to get system statistics"""
    try:
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            return jsonify({'error': 'Database manager not available'}), 500

        stats = db_manager.get_manager_stats()
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error in api_system_stats: {e}")
        return jsonify({'error': str(e)}), 500


@inspector_bp.route('/api/database/<db_name>/upload', methods=['POST'])
@require_inspector_auth(PermissionLevel.READ_WRITE)
@handle_errors
def api_upload_document(db_name):
    """API endpoint to upload a document file to a database"""
    try:
        db_manager = getattr(current_app, 'db_manager', None)
        if not db_manager:
            return jsonify({'error': 'Database manager not available'}), 500

        # Check if database exists
        if db_name not in db_manager.list_databases():
            return jsonify({'error': f'Database {db_name} not found'}), 404

        # Get the database instance
        db = db_manager.get_db(db_name)

        # Check for file in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read file content
        file_content = file.read()

        # Try to decode as text
        try:
            content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            # For binary files, we might need special handling
            # For now, return an error
            return jsonify({'error': 'File must be text-based'}), 400

        # Get optional document ID from form
        doc_id = request.form.get('doc_id', None)

        # Get metadata from form (if any)
        metadata = {}
        for key in request.form:
            if key.startswith('metadata_'):
                field_name = key[9:]  # Remove 'metadata_' prefix
                metadata[field_name] = request.form[key]

        # Add file-related metadata
        metadata['source_file'] = file.filename
        metadata['uploaded_via'] = 'inspector'
        metadata['upload_timestamp'] = datetime.now().isoformat()

        # Add document to database
        if doc_id:
            db.add(content, id=doc_id, metadata=metadata if metadata else None)
        else:
            doc_id = db.add(content, metadata=metadata if metadata else None)

        # Get the added document to return info
        added_doc = db.get(doc_id)

        return jsonify({
            'success': True,
            'document_id': doc_id,
            'chunks_created': len(added_doc.chunks) if hasattr(added_doc, 'chunks') else 1,
            'content_length': len(content),
            'filename': file.filename
        })

    except Exception as e:
        logger.error(f"Error uploading document to {db_name}: {e}")
        return jsonify({'error': str(e)}), 500


# Error handlers for inspector blueprint
@inspector_bp.errorhandler(404)
def inspector_not_found(e):
    """Custom 404 handler for inspector"""
    return render_template('error.html',
                           error="Page not found",
                           message="The requested inspector page was not found."), 404


@inspector_bp.errorhandler(500)
def inspector_server_error(e):
    """Custom 500 handler for inspector"""
    logger.error(f"Inspector server error: {e}")
    return render_template('error.html',
                           error="Server error",
                           message="An internal server error occurred in the inspector."), 500


# Template context processor to inject common variables
@inspector_bp.context_processor
def inject_inspector_context():
    """Inject common context variables into all inspector templates"""
    config = getattr(current_app, 'config_obj', None)
    # Check for either new key_id or old api_key for backward compat
    is_logged_in = 'inspector_key_id' in session or 'inspector_api_key' in session
    # Display key_id if available, otherwise show truncated old api_key for backward compat
    current_user = None
    if 'inspector_key_id' in session:
        current_user = session.get('inspector_key_id', '')
    elif 'inspector_api_key' in session:
        current_user = session.get('inspector_api_key', '')[:8] + '...'

    return {
        'inspector_enabled': inspector_enabled(),
        'require_api_key': config.server.security.require_api_key if config else False,
        'logged_in': is_logged_in,
        'current_user': current_user,
        'permission_level': session.get('inspector_permission_level', 'unknown')
    }
