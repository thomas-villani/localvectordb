import type { ErrorCode, ServerErrorPayload } from "./types.js";

/**
 * Base error class for all LocalVectorDB SDK errors.
 */
export class LocalVectorDBError extends Error {
  public readonly code: string;
  public readonly statusCode: number;
  public readonly details: Record<string, unknown>;
  public readonly recoverable: boolean;
  public readonly requestId?: string;
  public readonly timestamp?: string;

  constructor(
    message: string,
    options: {
      code?: string;
      statusCode?: number;
      details?: Record<string, unknown>;
      recoverable?: boolean;
      requestId?: string;
      timestamp?: string;
    } = {},
  ) {
    super(message);
    this.name = "LocalVectorDBError";
    this.code = options.code ?? "UNKNOWN_ERROR";
    this.statusCode = options.statusCode ?? 0;
    this.details = options.details ?? {};
    this.recoverable = options.recoverable ?? false;
    this.requestId = options.requestId;
    this.timestamp = options.timestamp;
  }
}

// ---------------------------------------------------------------------------
// 4xx errors
// ---------------------------------------------------------------------------

export class ValidationError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "VALIDATION_ERROR", statusCode: 400, ...options });
    this.name = "ValidationError";
  }
}

export class AuthenticationError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, {
      code: "AUTHENTICATION_ERROR",
      statusCode: 401,
      ...options,
    });
    this.name = "AuthenticationError";
  }
}

export class PermissionError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "PERMISSION_ERROR", statusCode: 403, ...options });
    this.name = "PermissionError";
  }
}

export class NotFoundError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { statusCode: 404, ...options });
    this.name = "NotFoundError";
  }
}

export class DatabaseNotFoundError extends NotFoundError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "DATABASE_NOT_FOUND", ...options });
    this.name = "DatabaseNotFoundError";
  }
}

export class DocumentNotFoundError extends NotFoundError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "DOCUMENT_NOT_FOUND", ...options });
    this.name = "DocumentNotFoundError";
  }
}

export class ConflictError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { statusCode: 409, ...options });
    this.name = "ConflictError";
  }
}

export class DuplicateDocumentError extends ConflictError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "DUPLICATE_DOCUMENT_ID", ...options });
    this.name = "DuplicateDocumentError";
  }
}

export class DatabaseAlreadyExistsError extends ConflictError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "DATABASE_ALREADY_EXISTS", ...options });
    this.name = "DatabaseAlreadyExistsError";
  }
}

/** A patch's `expectHash` precondition did not match the stored document (409). */
export class PatchConflictError extends ConflictError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "HASH_CONFLICT", ...options });
    this.name = "PatchConflictError";
  }
}

/** A patch op was unmatched, ambiguous, overlapping, or out of range (422). */
export class PatchFailedError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "PATCH_FAILED", statusCode: 422, ...options });
    this.name = "PatchFailedError";
  }
}

// ---------------------------------------------------------------------------
// 5xx errors
// ---------------------------------------------------------------------------

export class ServerError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { statusCode: 500, ...options });
    this.name = "ServerError";
  }
}

export class ConfigurationError extends ServerError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "CONFIGURATION_ERROR", ...options });
    this.name = "ConfigurationError";
  }
}

export class ServiceUnavailableError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { statusCode: 503, ...options });
    this.name = "ServiceUnavailableError";
  }
}

export class EmbeddingError extends ServiceUnavailableError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "EMBEDDING_ERROR", ...options });
    this.name = "EmbeddingError";
  }
}

export class OllamaNotAvailableError extends ServiceUnavailableError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "OLLAMA_NOT_AVAILABLE", ...options });
    this.name = "OllamaNotAvailableError";
  }
}

export class DatabaseConnectionError extends ServiceUnavailableError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "DATABASE_CONNECTION_ERROR", ...options });
    this.name = "DatabaseConnectionError";
  }
}

// ---------------------------------------------------------------------------
// Network errors (not HTTP-level)
// ---------------------------------------------------------------------------

export class ConnectionError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "CONNECTION_ERROR", ...options });
    this.name = "ConnectionError";
  }
}

export class TimeoutError extends LocalVectorDBError {
  constructor(
    message: string,
    options: ConstructorParameters<typeof LocalVectorDBError>[1] = {},
  ) {
    super(message, { code: "TIMEOUT_ERROR", ...options });
    this.name = "TimeoutError";
  }
}

// ---------------------------------------------------------------------------
// Factory: map server error response → typed exception
// ---------------------------------------------------------------------------

const CODE_TO_CLASS: Record<
  string,
  new (
    message: string,
    options?: ConstructorParameters<typeof LocalVectorDBError>[1],
  ) => LocalVectorDBError
> = {
  VALIDATION_ERROR: ValidationError,
  // Bad filter specs (unknown fields, unsupported operators) are 400s.
  INVALID_FILTER: ValidationError,
  DATABASE_NOT_FOUND: DatabaseNotFoundError,
  DOCUMENT_NOT_FOUND: DocumentNotFoundError,
  DUPLICATE_DOCUMENT_ID: DuplicateDocumentError,
  HASH_CONFLICT: PatchConflictError,
  PATCH_FAILED: PatchFailedError,
  DATABASE_ALREADY_EXISTS: DatabaseAlreadyExistsError,
  // SSRF guard on POST /databases: a request-supplied embedding provider URL was
  // rejected (not opted in, or host not in the allowlist). Both are 403s.
  EMBEDDING_URL_NOT_ALLOWED: PermissionError,
  EMBEDDING_URL_HOST_NOT_ALLOWED: PermissionError,
  EMBEDDING_ERROR: EmbeddingError,
  OLLAMA_NOT_AVAILABLE: OllamaNotAvailableError,
  DATABASE_CONNECTION_ERROR: DatabaseConnectionError,
  CONFIGURATION_ERROR: ConfigurationError,
  DATABASE_ERROR: ServerError,
  INTERNAL_ERROR: ServerError,
};

const STATUS_TO_CLASS: Record<
  number,
  new (
    message: string,
    options?: ConstructorParameters<typeof LocalVectorDBError>[1],
  ) => LocalVectorDBError
> = {
  400: ValidationError,
  401: AuthenticationError,
  403: PermissionError,
  404: NotFoundError,
  409: ConflictError,
  503: ServiceUnavailableError,
};

/**
 * Create a typed error from an HTTP status code and response body.
 */
export function errorFromResponse(
  statusCode: number,
  body: ServerErrorPayload | string,
): LocalVectorDBError {
  // Try to extract structured error info
  let message: string;
  let code: ErrorCode | undefined;
  let details: Record<string, unknown> = {};
  let recoverable = false;
  let requestId: string | undefined;
  let timestamp: string | undefined;

  if (typeof body === "string") {
    message = body || `HTTP ${statusCode}`;
  } else if (body?.error) {
    const err = body.error;
    message = err.message || `HTTP ${statusCode}`;
    code = err.code;
    details = err.details ?? {};
    recoverable = err.recoverable ?? false;
    requestId = err.request_id;
    timestamp = err.timestamp;
  } else {
    message = `HTTP ${statusCode}`;
  }

  const opts = { code, statusCode, details, recoverable, requestId, timestamp };

  // Match by error code first, then fall back to HTTP status
  if (code && code in CODE_TO_CLASS) {
    return new CODE_TO_CLASS[code](message, opts);
  }

  const StatusClass =
    STATUS_TO_CLASS[statusCode] ??
    (statusCode >= 500 ? ServerError : LocalVectorDBError);

  return new StatusClass(message, opts);
}
