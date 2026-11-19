-- ─────────────────────────────────────────────────────────────────────────────
-- Accounts, roles and activity for APOLLO-M (the web view and the analyst
-- console share one identity: Supabase Auth).
--
-- Run once in the Supabase SQL editor, after schema.sql. Safe to re-run.
--
-- What it enforces, in the database rather than in application code:
--   * every signed-up user gets a profile (trigger on auth.users);
--   * a user reads only their own profile and their own activity;
--   * an admin reads everyone's — "admin" is a role stored here, granted with
--     SQL by someone who already has database access, never a password in code;
--   * users can change their display name, not their role;
--   * the activity log is append-only: no update or delete policy exists.
--
-- Make yourself admin after signing up once:
--   update public.profiles set role = 'admin' where email = 'you@example.com';
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.profiles (
  id           uuid primary key references auth.users (id) on delete cascade,
  email        text not null,
  full_name    text,
  role         text not null default 'analyst' check (role in ('analyst', 'admin')),
  created_at   timestamptz not null default now(),
  last_seen_at timestamptz
);
alter table public.profiles enable row level security;

-- security definer so the policy below can ask "is the caller an admin?"
-- without the profiles policy recursively evaluating itself.
create or replace function public.is_admin()
returns boolean
language sql stable security definer set search_path = public
as $$
  select exists (select 1 from public.profiles where id = auth.uid() and role = 'admin');
$$;
revoke all on function public.is_admin() from public;
grant execute on function public.is_admin() to authenticated;

drop policy if exists "read own profile, admins read all" on public.profiles;
create policy "read own profile, admins read all" on public.profiles
  for select using (auth.uid() = id or public.is_admin());

drop policy if exists "update own profile" on public.profiles;
create policy "update own profile" on public.profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);

-- Column-level: a user may touch only their name and last-seen time. The role
-- column is not grantable, so no client can promote itself.
revoke update on public.profiles from authenticated, anon;
grant update (full_name, last_seen_at) on public.profiles to authenticated;
grant select on public.profiles to authenticated;

create or replace function public.handle_new_user()
returns trigger
language plpgsql security definer set search_path = public
as $$
begin
  insert into public.profiles (id, email, full_name)
  values (new.id, new.email,
          coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name'))
  on conflict (id) do nothing;
  return new;
end
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- Users who signed up before this migration.
insert into public.profiles (id, email, full_name)
select id, email, coalesce(raw_user_meta_data ->> 'full_name', raw_user_meta_data ->> 'name')
from auth.users
on conflict (id) do nothing;

create table if not exists public.activity (
  id       bigint generated always as identity primary key,
  user_id  uuid not null default auth.uid() references public.profiles (id) on delete cascade,
  at       timestamptz not null default now(),
  surface  text not null check (surface in ('console', 'web')),
  action   text not null check (char_length(action) between 1 and 40),
  detail   jsonb not null default '{}'::jsonb check (pg_column_size(detail) < 2000)
);
create index if not exists activity_user_at on public.activity (user_id, at desc);
create index if not exists activity_at on public.activity (at desc);
alter table public.activity enable row level security;

drop policy if exists "log own activity" on public.activity;
create policy "log own activity" on public.activity
  for insert with check (auth.uid() = user_id);

drop policy if exists "read own activity, admins read all" on public.activity;
create policy "read own activity, admins read all" on public.activity
  for select using (auth.uid() = user_id or public.is_admin());

grant select, insert on public.activity to authenticated;
revoke update, delete on public.activity from authenticated, anon;
