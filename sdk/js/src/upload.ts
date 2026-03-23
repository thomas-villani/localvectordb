import type { UploadableFile, UploadOptions, UploadResponse } from "./types.js";
import type { HttpClient } from "./http.js";

/**
 * Upload files to a database with automatic text extraction.
 *
 * @internal Consumed via {@link DatabaseHandle.upload}.
 */
export async function uploadFiles(
  httpClient: HttpClient,
  dbName: string,
  files: UploadableFile[],
  options?: UploadOptions,
): Promise<UploadResponse> {
  const formData = new FormData();

  for (const file of files) {
    if (file instanceof Blob) {
      // File extends Blob and carries .name; plain Blob gets a default name.
      const name =
        "name" in file && typeof file.name === "string"
          ? file.name
          : "upload";
      formData.append("files", file, name);
    } else {
      // { name, data, type } convenience form for Node.js
      const blob =
        file.data instanceof Blob
          ? file.data
          : new Blob([file.data as BlobPart], { type: file.type });
      formData.append("files", blob, file.name);
    }
  }

  // Append optional form fields (matching server Form() parameters).
  if (options?.metadata !== undefined) {
    formData.append("metadata", JSON.stringify(options.metadata));
  }
  if (options?.batch_size !== undefined) {
    formData.append("batch_size", String(options.batch_size));
  }
  if (options?.ids !== undefined) {
    formData.append(
      "ids",
      JSON.stringify(
        Array.isArray(options.ids) ? options.ids : [options.ids],
      ),
    );
  }
  if (options?.mode !== undefined) {
    formData.append("mode", options.mode);
  }
  if (options?.errors !== undefined) {
    formData.append("errors", options.errors);
  }
  if (options?.similarity_threshold !== undefined) {
    formData.append(
      "similarity_threshold",
      String(options.similarity_threshold),
    );
  }
  if (options?.use_filename_as_id !== undefined) {
    formData.append(
      "use_filename_as_id",
      options.use_filename_as_id ? "true" : "false",
    );
  }
  if (options?.extractor_kwargs !== undefined) {
    formData.append("extractor_kwargs", JSON.stringify(options.extractor_kwargs));
  }

  const response = await httpClient.postRaw(
    `/api/v1/${encodeURIComponent(dbName)}/upload`,
    formData,
  );

  return (await response.json()) as UploadResponse;
}
