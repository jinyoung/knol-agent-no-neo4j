-- Create nodes table
create table if not exists nodes (
    id text primary key,
    title text not null,
    content text not null,
    node_type text not null,
    relationships jsonb not null default '{}',
    metadata jsonb not null default '{}',
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now()
);

-- Enable Row Level Security (RLS)
alter table nodes enable row level security;

-- Create policy to allow all operations for authenticated users
create policy "Enable all operations for authenticated users"
on nodes
for all
to authenticated
using (true)
with check (true);

-- Create index on node_type for faster filtering
create index if not exists idx_nodes_node_type on nodes(node_type);

-- Create index on created_at for chronological queries
create index if not exists idx_nodes_created_at on nodes(created_at);

-- Add function to automatically update updated_at timestamp
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

-- Create trigger to update updated_at on row update
create trigger update_nodes_updated_at
    before update on nodes
    for each row
    execute function update_updated_at_column(); 