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

-- Hyperparameter config (full dict: lr, dims, early-stopping, etc.)
alter table public.training_jobs
    add column if not exists config jsonb not null default '{}'::jsonb;

-- Convergence
alter table public.training_jobs
    add column if not exists epochs_trained integer;

alter table public.training_jobs
    add column if not exists early_stopped boolean;

alter table public.training_jobs
    add column if not exists convergence_epoch integer;

-- Timing / throughput
alter table public.training_jobs
    add column if not exists avg_epoch_time_seconds double precision;

alter table public.training_jobs
    add column if not exists steps_per_epoch integer;

alter table public.training_jobs
    add column if not exists n_training_rows integer;

alter table public.training_jobs
    add column if not exists avg_samples_per_second double precision;

-- Loss detail (the existing final_loss stays; add the richer breakdown)
alter table public.training_jobs
    add column if not exists final_generator_loss double precision;

alter table public.training_jobs
    add column if not exists final_discriminator_loss double precision;

alter table public.training_jobs
    add column if not exists final_loss_ratio double precision;

alter table public.training_jobs
    add column if not exists final_mode_collapse_score double precision;

-- Best-epoch tracking
alter table public.training_jobs
    add column if not exists best_generator_loss double precision;

alter table public.training_jobs
    add column if not exists best_epoch integer;

-- Overall stability metric (std of G-loss across all epochs)
alter table public.training_jobs
    add column if not exists loss_stability_std double precision;

-- SDMetrics quality report (QualityReport + DiagnosticReport + ML-efficacy)
alter table public.training_jobs
    add column if not exists sdmetrics jsonb not null default '{}'::jsonb;

-- Infrastructure provenance
alter table public.training_jobs
    add column if not exists source text;           -- "modal" | "local"

alter table public.training_jobs
    add column if not exists gpu text;              -- "T4", null for local


-- -----------------------------------------------------------------------------
-- generation_jobs
-- Add SDMetrics result written after generate_ctgan_modal completes.
-- -----------------------------------------------------------------------------

alter table public.generation_jobs
    add column if not exists sdmetrics jsonb not null default '{}'::jsonb;


-- -----------------------------------------------------------------------------
-- trained_models
-- The metadata column already holds a jsonb blob.  No structural change needed
-- because sdmetrics, config, gpu, and source are stored inside metadata.
-- If you want to query model quality scores directly, promote them to columns:
-- -----------------------------------------------------------------------------

alter table public.trained_models
    add column if not exists config jsonb not null default '{}'::jsonb;

-- Computed quality score extracted from sdmetrics report for fast filtering
alter table public.trained_models
    add column if not exists sdmetrics_quality_score double precision
        generated always as ((metadata -> 'sdmetrics' ->> 'quality_score')::double precision) stored;

alter table public.trained_models
    add column if not exists sdmetrics_diagnostic_score double precision
        generated always as ((metadata -> 'sdmetrics' ->> 'diagnostic_score')::double precision) stored;


-- -----------------------------------------------------------------------------
-- Indexes
-- Support fast queries on job status, quality scores, and provenance.
-- -----------------------------------------------------------------------------

-- Filter training jobs by status (e.g. all "running" jobs for heartbeat checks)
create index if not exists idx_training_jobs_status
    on public.training_jobs (status);

-- Filter by source to separate Modal vs local runs in analytics
create index if not exists idx_training_jobs_source
    on public.training_jobs (source);

-- Find best-quality models across a user's datasets
create index if not exists idx_trained_models_quality_score
    on public.trained_models (user_id, sdmetrics_quality_score desc nulls last);

-- Filter generation jobs by status
create index if not exists idx_generation_jobs_status
    on public.generation_jobs (status);


-- -----------------------------------------------------------------------------
-- Helper view: training_job_summary
-- Joins training_jobs with the sdmetrics scores for dashboards / reporting.
-- Avoids having to parse the full jsonb blob on the client.
-- -----------------------------------------------------------------------------

create or replace view public.training_job_summary as
select
    tj.job_id,
    tj.dataset_id,
    tj.user_id,
    tj.status,
    tj.source,
    tj.gpu,
    tj.epochs_trained,
    tj.total_epochs,
    tj.early_stopped,
    tj.convergence_epoch,
    tj.training_time_seconds,
    tj.avg_epoch_time_seconds,
    tj.n_training_rows,
    tj.avg_samples_per_second,
    -- Loss summary
    tj.final_generator_loss,
    tj.final_discriminator_loss,
    tj.final_loss_ratio,
    tj.final_mode_collapse_score,
    tj.best_generator_loss,
    tj.best_epoch,
    tj.loss_stability_std,
    -- SDMetrics scores promoted to top-level columns for easy sorting
    (tj.sdmetrics ->> 'quality_score')::double precision        as sdmetrics_quality_score,
    (tj.sdmetrics ->> 'diagnostic_score')::double precision     as sdmetrics_diagnostic_score,
    (tj.sdmetrics -> 'ml_efficacy' ->> 'train_on_synthetic_test_on_real_f1')::double precision
                                                                as sdmetrics_tstr_f1,
    -- Config fields promoted for easy filtering / grouping
    (tj.config ->> 'embedding_dim')::integer                    as embedding_dim,
    (tj.config ->> 'generator_lr')::double precision            as generator_lr,
    (tj.config ->> 'discriminator_lr')::double precision        as discriminator_lr,
    (tj.config ->> 'batch_size')::integer                       as batch_size,
    tj.error,
    tj.last_heartbeat,
    tj.created_at,
    tj.updated_at
from public.training_jobs tj;


-- Service role: full access (Modal runner + FastAPI backend write via service key)
drop policy if exists "Service role full access on training_jobs" on public.training_jobs;
create policy "Service role full access on training_jobs"
    on public.training_jobs
    for all
    to service_role
    using (true)
    with check (true);

-- Authenticated users: insert their own jobs
drop policy if exists "Users can insert their training jobs" on public.training_jobs;
create policy "Users can insert their training jobs"
    on public.training_jobs for insert
    to authenticated
    with check (auth.uid() = user_id);

-- Authenticated users: update only their own jobs
-- (also covers the Modal callback path that updates status / loss_history /
--  sdmetrics etc. – those writes go through the service role, not this policy)
drop policy if exists "Users can update their training jobs" on public.training_jobs;
create policy "Users can update their training jobs"
    on public.training_jobs for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);


-- -----------------------------------------------------------------------------
-- generation_jobs  (existing table, missing write + service-role policies)
-- -----------------------------------------------------------------------------

drop policy if exists "Service role full access on generation_jobs" on public.generation_jobs;
create policy "Service role full access on generation_jobs"
    on public.generation_jobs
    for all
    to service_role
    using (true)
    with check (true);

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


-- -----------------------------------------------------------------------------
-- datasets  (existing table, missing write + service-role policies)
-- -----------------------------------------------------------------------------

drop policy if exists "Service role full access on datasets" on public.datasets;
create policy "Service role full access on datasets"
    on public.datasets
    for all
    to service_role
    using (true)
    with check (true);

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


-- -----------------------------------------------------------------------------
-- trained_models  (existing table, missing write + service-role policies)
-- -----------------------------------------------------------------------------

drop policy if exists "Service role full access on trained_models" on public.trained_models;
create policy "Service role full access on trained_models"
    on public.trained_models
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "Users can read their trained models" on public.trained_models;
create policy "Users can read their trained models"
    on public.trained_models for select
    to authenticated
    using (auth.uid() = user_id);


-- -----------------------------------------------------------------------------
-- training_job_summary view
-- Views inherit the RLS of their underlying tables when security_invoker = on.
-- Setting this means the view runs as the calling user, so the training_jobs
-- row-level policy (auth.uid() = user_id) is enforced automatically.
-- -----------------------------------------------------------------------------

alter view public.training_job_summary
    set (security_invoker = on);


-- -----------------------------------------------------------------------------
-- Verification: list all active policies (useful after running this script)
-- -----------------------------------------------------------------------------
-- select schemaname, tablename, policyname, roles, cmd, qual
-- from pg_policies
-- where schemaname = 'public'
-- order by tablename, policyname;