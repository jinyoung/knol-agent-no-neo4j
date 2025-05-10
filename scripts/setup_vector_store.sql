-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Truncate the nodes table to start fresh
truncate table nodes;

-- Recreate the nodes table with all necessary columns
drop table if exists nodes;
create table nodes (
    id text primary key,
    title text,
    content text,
    node_type text,
    relationships jsonb default '{}',
    metadata jsonb default '{}',
    embedding vector(1536),
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now(),
    summary jsonb default '{}'
);

-- Create a function to match similar nodes
create or replace function match_documents (
  query_embedding vector(1536),
  filter jsonb default '{}'::jsonb
) returns table (
  id text,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    nodes.id,
    nodes.content,
    jsonb_build_object(
      'id', nodes.id,
      'title', nodes.title,
      'content', nodes.content,
      'node_type', nodes.node_type,
      'relationships', nodes.relationships,
      'metadata', nodes.metadata
    ) as metadata,
    1 - (nodes.embedding <=> query_embedding) as similarity
  from nodes
  where nodes.embedding is not null
  order by nodes.embedding <=> query_embedding
  limit 10;
end;
$$; 