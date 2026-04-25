create table if not exists public.datasets (
    id text primary key,
    filename text,
    dataset_type text not null default 'real',
    object_key text not null,
    target text,
    n_rows integer not null,
    n_features integer not null,
    class_dist jsonb not null default '{}'::jsonb,
    schema jsonb not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.training_jobs (
    job_id text primary key,
    dataset_id text not null references public.datasets(id) on delete cascade,
    status text not null,
    current_epoch integer not null default 0,
    total_epochs integer not null,
    current_batch integer not null default 0,
    total_batches integer not null default 0,
    loss_history jsonb not null default '[]'::jsonb,
    batch_history jsonb not null default '[]'::jsonb,
    training_time_seconds double precision,
    final_loss double precision,
    error text,
    model_id text,
    model_path text,
    last_heartbeat timestamptz,
    started_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.trained_models (
    id text primary key,
    dataset_id text not null references public.datasets(id) on delete cascade,
    object_key text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz default now()
);

create index if not exists idx_training_jobs_dataset_id
    on public.training_jobs (dataset_id, created_at desc);

create index if not exists idx_trained_models_dataset_id
    on public.trained_models (dataset_id);
