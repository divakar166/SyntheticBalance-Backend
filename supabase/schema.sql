create table if not exists public.datasets (
    id text primary key,
    user_id uuid references auth.users(id) on delete cascade,
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
    loss_history jsonb not null default '[]'::jsonb,
    training_time_seconds double precision,
    final_loss double precision,
    error text,
    model_id text,
    model_path text,
    modal_call_id text,
    last_heartbeat timestamptz,
    started_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.training_jobs
    add column if not exists modal_call_id text;

alter table public.datasets
    add column if not exists user_id uuid references auth.users(id) on delete cascade;

alter table public.training_jobs
    add column if not exists user_id uuid references auth.users(id) on delete cascade;

create table if not exists public.generation_jobs (
    job_id text primary key,
    user_id uuid references auth.users(id) on delete cascade,
    dataset_id text not null references public.datasets(id) on delete cascade,
    status text not null,
    n_samples integer not null,
    synthetic_id text references public.datasets(id) on delete set null,
    synthetic_path text,
    preview jsonb not null default '[]'::jsonb,
    generation_time_seconds double precision,
    error text,
    modal_call_id text,
    last_heartbeat timestamptz,
    started_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.trained_models (
    id text primary key,
    user_id uuid references auth.users(id) on delete cascade,
    dataset_id text not null references public.datasets(id) on delete cascade,
    object_key text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz default now()
);

alter table public.generation_jobs
    add column if not exists user_id uuid references auth.users(id) on delete cascade;

alter table public.trained_models
    add column if not exists user_id uuid references auth.users(id) on delete cascade;

create index if not exists idx_training_jobs_dataset_id
    on public.training_jobs (dataset_id, created_at desc);

create index if not exists idx_generation_jobs_dataset_id
    on public.generation_jobs (dataset_id, created_at desc);

create index if not exists idx_trained_models_dataset_id
    on public.trained_models (dataset_id);

create index if not exists idx_datasets_user_id
    on public.datasets (user_id, created_at desc);

create index if not exists idx_training_jobs_user_id
    on public.training_jobs (user_id, created_at desc);

create index if not exists idx_generation_jobs_user_id
    on public.generation_jobs (user_id, created_at desc);

create index if not exists idx_trained_models_user_id
    on public.trained_models (user_id, created_at desc);

alter table public.datasets enable row level security;
alter table public.training_jobs enable row level security;
alter table public.generation_jobs enable row level security;
alter table public.trained_models enable row level security;

drop policy if exists "Users can read their datasets" on public.datasets;
create policy "Users can read their datasets"
    on public.datasets for select
    using (auth.uid() = user_id);

drop policy if exists "Users can read their training jobs" on public.training_jobs;
create policy "Users can read their training jobs"
    on public.training_jobs for select
    using (auth.uid() = user_id);

drop policy if exists "Users can read their generation jobs" on public.generation_jobs;
create policy "Users can read their generation jobs"
    on public.generation_jobs for select
    using (auth.uid() = user_id);

drop policy if exists "Users can read their trained models" on public.trained_models;
create policy "Users can read their trained models"
    on public.trained_models for select
    using (auth.uid() = user_id);
