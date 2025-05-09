-- Drop trigger first
DROP TRIGGER IF EXISTS update_nodes_updated_at ON nodes;

-- Drop the trigger function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop indexes
DROP INDEX IF EXISTS idx_nodes_created_at;
DROP INDEX IF EXISTS idx_nodes_node_type;

-- Drop RLS policy
DROP POLICY IF EXISTS "Enable all operations for authenticated users" ON nodes;

-- Disable RLS
ALTER TABLE nodes DISABLE ROW LEVEL SECURITY; 