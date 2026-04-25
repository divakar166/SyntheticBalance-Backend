# Backend Setup

This backend supports three storage modes:

- `InMemoryBackend`: local-only development, not suitable for Modal.
- `SupabaseMinioBackend`: Supabase metadata plus MinIO object storage.
- `SupabaseS3Backend`: Supabase metadata plus AWS S3 object storage.

For Modal-compatible training, use Supabase + AWS S3.

## Environment Variables

Set these for the S3-backed path:

```env
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...

AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_DATASET_BUCKET=your-dataset-bucket
AWS_S3_MODEL_BUCKET=your-model-bucket
```

Optional:

```env
AWS_SESSION_TOKEN=
AWS_S3_ENDPOINT_URL=
```

`AWS_S3_ENDPOINT_URL` is only needed for S3-compatible services such as LocalStack or R2-style endpoints. Leave it unset for normal AWS S3.

## Recommended AWS Layout

Create two buckets:

- one for uploaded and generated datasets
- one for trained model artifacts

Example names:

- `synthetic-data-poc-datasets-<unique-suffix>`
- `synthetic-data-poc-models-<unique-suffix>`

## Supabase Schema

Apply the SQL in [schema.sql](D:\Projects\synthetic-data-poc\backend\supabase\schema.sql) before running the backend.

## Notes

- The backend auto-selects S3 when the AWS variables above are present.
- If S3 is not configured but MinIO is, it will use the MinIO backend.
- If neither shared storage backend is configured, it falls back to in-memory storage.
- In-memory storage will not work with Modal workers because the data is not shared across processes.
