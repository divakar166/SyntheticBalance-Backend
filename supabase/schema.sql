begin;

-- Core dataset records. Real uploads and synthetic outputs both live here.
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

-- One row per training attempt. A dataset can have many jobs/configs.
create table if not exists public.training_jobs (
    job_id text primary key,
    user_id uuid references auth.users(id) on delete cascade,
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
    updated_at timestamptz not null default now(),
    config jsonb not null default '{}'::jsonb,
    epochs_trained integer,
    early_stopped boolean,
    convergence_epoch integer,
    avg_epoch_time_seconds double precision,
    steps_per_epoch integer,
    n_training_rows integer,
    avg_samples_per_second double precision,
    final_generator_loss double precision,
    final_discriminator_loss double precision,
    final_loss_ratio double precision,
    final_mode_collapse_score double precision,
    best_generator_loss double precision,
    best_epoch integer,
    loss_stability_std double precision,
    sdmetrics jsonb default '{}'::jsonb,
    source text,
    gpu text
);

-- One row per trained model artifact. Multiple models can point at one source dataset.
create table if not exists public.trained_models (
    id text primary key,
    user_id uuid references auth.users(id) on delete cascade,
    dataset_id text not null references public.datasets(id) on delete cascade,
    training_job_id text,
    object_key text not null,
    metadata jsonb not null default '{}'::jsonb,
    config jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

-- One row per synthetic generation attempt. One trained model can create many synthetic datasets.
create table if not exists public.generation_jobs (
    job_id text primary key,
    user_id uuid references auth.users(id) on delete cascade,
    dataset_id text not null references public.datasets(id) on delete cascade,
    model_id text,
    status text not null,
    n_samples integer not null,
    run_sdmetrics boolean not null default true,
    synthetic_id text references public.datasets(id) on delete set null,
    synthetic_path text,
    preview jsonb not null default '[]'::jsonb,
    generation_time_seconds double precision,
    error text,
    modal_call_id text,
    last_heartbeat timestamptz,
    started_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    sdmetrics jsonb default '{}'::jsonb
);

-- Idempotent upgrades for existing Supabase projects.
alter table public.datasets
    add column if not exists user_id uuid references auth.users(id) on delete cascade,
    add column if not exists filename text,
    add column if not exists dataset_type text not null default 'real',
    add column if not exists object_key text,
    add column if not exists target text,
    add column if not exists n_rows integer,
    add column if not exists n_features integer,
    add column if not exists class_dist jsonb not null default '{}'::jsonb,
    add column if not exists schema jsonb,
    add column if not exists metadata jsonb not null default '{}'::jsonb,
    add column if not exists created_at timestamptz not null default now();

alter table public.training_jobs
    add column if not exists user_id uuid references auth.users(id) on delete cascade,
    add column if not exists model_id text,
    add column if not exists model_path text,
    add column if not exists modal_call_id text,
    add column if not exists last_heartbeat timestamptz,
    add column if not exists started_at timestamptz,
    add column if not exists created_at timestamptz not null default now(),
    add column if not exists updated_at timestamptz not null default now(),
    add column if not exists config jsonb not null default '{}'::jsonb,
    add column if not exists epochs_trained integer,
    add column if not exists early_stopped boolean,
    add column if not exists convergence_epoch integer,
    add column if not exists avg_epoch_time_seconds double precision,
    add column if not exists steps_per_epoch integer,
    add column if not exists n_training_rows integer,
    add column if not exists avg_samples_per_second double precision,
    add column if not exists final_generator_loss double precision,
    add column if not exists final_discriminator_loss double precision,
    add column if not exists final_loss_ratio double precision,
    add column if not exists final_mode_collapse_score double precision,
    add column if not exists best_generator_loss double precision,
    add column if not exists best_epoch integer,
    add column if not exists loss_stability_std double precision,
    add column if not exists sdmetrics jsonb default '{}'::jsonb,
    add column if not exists source text,
    add column if not exists gpu text;

alter table public.trained_models
    add column if not exists user_id uuid references auth.users(id) on delete cascade,
    add column if not exists training_job_id text,
    add column if not exists metadata jsonb not null default '{}'::jsonb,
    add column if not exists config jsonb not null default '{}'::jsonb,
    add column if not exists created_at timestamptz not null default now();

alter table public.generation_jobs
    add column if not exists user_id uuid references auth.users(id) on delete cascade,
    add column if not exists model_id text,
    add column if not exists run_sdmetrics boolean not null default true,
    add column if not exists synthetic_id text references public.datasets(id) on delete set null,
    add column if not exists synthetic_path text,
    add column if not exists preview jsonb not null default '[]'::jsonb,
    add column if not exists generation_time_seconds double precision,
    add column if not exists error text,
    add column if not exists modal_call_id text,
    add column if not exists last_heartbeat timestamptz,
    add column if not exists started_at timestamptz,
    add column if not exists created_at timestamptz not null default now(),
    add column if not exists updated_at timestamptz not null default now(),
    add column if not exists sdmetrics jsonb default '{}'::jsonb;

create index if not exists idx_datasets_user_id
    on public.datasets (user_id, created_at desc);
create index if not exists idx_datasets_type
    on public.datasets (dataset_type, created_at desc);

create index if not exists idx_training_jobs_dataset_id
    on public.training_jobs (dataset_id, created_at desc);
create index if not exists idx_training_jobs_user_id
    on public.training_jobs (user_id, created_at desc);
create index if not exists idx_training_jobs_status
    on public.training_jobs (status);
create index if not exists idx_training_jobs_model_id
    on public.training_jobs (model_id);
create index if not exists idx_training_jobs_source
    on public.training_jobs (source);

create index if not exists idx_trained_models_dataset_id
    on public.trained_models (dataset_id, created_at desc);
create index if not exists idx_trained_models_training_job_id
    on public.trained_models (training_job_id);
create index if not exists idx_trained_models_user_id
    on public.trained_models (user_id, created_at desc);

create index if not exists idx_generation_jobs_dataset_id
    on public.generation_jobs (dataset_id, created_at desc);
create index if not exists idx_generation_jobs_model_id
    on public.generation_jobs (model_id, created_at desc);
create index if not exists idx_generation_jobs_user_id
    on public.generation_jobs (user_id, created_at desc);
create index if not exists idx_generation_jobs_status
    on public.generation_jobs (status);
create index if not exists idx_generation_jobs_synthetic_id
    on public.generation_jobs (synthetic_id);

alter table public.datasets enable row level security;
alter table public.training_jobs enable row level security;
alter table public.trained_models enable row level security;
alter table public.generation_jobs enable row level security;

drop policy if exists "Users can read their datasets" on public.datasets;
create policy "Users can read their datasets"
    on public.datasets for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists "Users can insert their datasets" on public.datasets;
create policy "Users can insert their datasets"
    on public.datasets for insert
    to authenticated
    with check (auth.uid() = user_id);

drop policy if exists "Users can delete their datasets" on public.datasets;
create policy "Users can delete their datasets"
    on public.datasets for delete
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists "Service role full access on datasets" on public.datasets;
create policy "Service role full access on datasets"
    on public.datasets
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "Users can read their training jobs" on public.training_jobs;
create policy "Users can read their training jobs"
    on public.training_jobs for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists "Users can insert their training jobs" on public.training_jobs;
create policy "Users can insert their training jobs"
    on public.training_jobs for insert
    to authenticated
    with check (auth.uid() = user_id);

drop policy if exists "Users can update their training jobs" on public.training_jobs;
create policy "Users can update their training jobs"
    on public.training_jobs for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "Service role full access on training_jobs" on public.training_jobs;
create policy "Service role full access on training_jobs"
    on public.training_jobs
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "Users can read their trained models" on public.trained_models;
create policy "Users can read their trained models"
    on public.trained_models for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists "Service role full access on trained_models" on public.trained_models;
create policy "Service role full access on trained_models"
    on public.trained_models
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "Users can read their generation jobs" on public.generation_jobs;
create policy "Users can read their generation jobs"
    on public.generation_jobs for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists "Users can insert their generation jobs" on public.generation_jobs;
create policy "Users can insert their generation jobs"
    on public.generation_jobs for insert
    to authenticated
    with check (auth.uid() = user_id);

drop policy if exists "Users can update their generation jobs" on public.generation_jobs;
create policy "Users can update their generation jobs"
    on public.generation_jobs for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "Service role full access on generation_jobs" on public.generation_jobs;
create policy "Service role full access on generation_jobs"
    on public.generation_jobs
    for all
    to service_role
    using (true)
    with check (true);

drop view if exists public.trained_model_summary;
drop view if exists public.training_job_summary;

create view public.training_job_summary as
select
    tj.*,
    (tj.sdmetrics ->> 'quality_score')::double precision as sdmetrics_quality_score,
    (tj.sdmetrics ->> 'diagnostic_score')::double precision as sdmetrics_diagnostic_score,
    (tj.sdmetrics -> 'ml_efficacy' ->> 'train_on_synthetic_test_on_real_f1')::double precision as sdmetrics_tstr_f1,
    (tj.config ->> 'embedding_dim')::integer as embedding_dim,
    (tj.config ->> 'generator_lr')::double precision as generator_lr,
    (tj.config ->> 'discriminator_lr')::double precision as discriminator_lr,
    (tj.config ->> 'batch_size')::integer as batch_size
from public.training_jobs tj;

create view public.trained_model_summary as
select
    tm.id,
    tm.user_id,
    tm.dataset_id,
    coalesce(tm.training_job_id, tj.job_id) as training_job_id,
    tm.object_key,
    tm.metadata,
    tm.config,
    tm.created_at,
    tj.status,
    tj.training_time_seconds,
    tj.epochs_trained,
    tj.early_stopped,
    tj.final_loss,
    tj.final_generator_loss,
    tj.final_discriminator_loss,
    tj.final_loss_ratio,
    tj.best_generator_loss,
    tj.best_epoch,
    tj.sdmetrics,
    tj.config as training_config,
    (tj.sdmetrics ->> 'quality_score')::double precision as sdmetrics_quality_score,
    (tj.sdmetrics ->> 'diagnostic_score')::double precision as sdmetrics_diagnostic_score,
    (
        select count(*)
        from public.generation_jobs gj
        where gj.model_id = tm.id and gj.status = 'completed'
    ) as synthetic_dataset_count
from public.trained_models tm
left join public.training_jobs tj
    on tj.model_id = tm.id
    or tj.job_id = tm.training_job_id;

alter view public.training_job_summary set (security_invoker = on);
alter view public.trained_model_summary set (security_invoker = on);

notify pgrst, 'reload schema';

commit;
