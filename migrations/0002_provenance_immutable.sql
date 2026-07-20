-- Provenance is immutable after ingest · LOOPHOLES.md §8.
--
-- Enforced in the DATABASE, not in the engine, so no future code path — engine,
-- migration, or an operator's ad-hoc UPDATE — can rewrite a citation out from
-- under a human who is verifying it. The NOT NULL constraints in 0001 stop a
-- citation from being absent; this stops one from being *changed*.
--
-- ! Hand-written: pipeline_data.schema_generate renders tables and indexes only,
-- it has no way to express a trigger or function.

CREATE OR REPLACE FUNCTION chunks_provenance_is_immutable()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.source_url      IS DISTINCT FROM OLD.source_url
    OR NEW.source_title    IS DISTINCT FROM OLD.source_title
    OR NEW.locator_page    IS DISTINCT FROM OLD.locator_page
    OR NEW.locator_section IS DISTINCT FROM OLD.locator_section THEN
        RAISE EXCEPTION
            'provenance is immutable after ingest (chunk %)', OLD.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Partitioned tables propagate row triggers to every partition, so one
-- statement covers all 64.
CREATE TRIGGER chunks_provenance_guard
    BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_provenance_is_immutable();
