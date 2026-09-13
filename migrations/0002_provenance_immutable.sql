-- Provenance is immutable after ingest · docs/FAILURE_MODES.md §7.
--
-- Enforced in the DATABASE, ✗ in the engine, so no future code path — engine,
-- migration, or an operator's ad-hoc UPDATE — can rewrite a citation out from
-- under a human who is verifying it.
--
-- ! Guards the columns this corpus actually has. An earlier version of this
-- file named `locator_page` and `locator_section`, which do not exist: the
-- locator is built at query time from `chapter` and `article`
-- (`pipeline.rs::locator_of`). It could never have been applied, so the
-- immutability FAILURE_MODES claims was never actually enforced.
--
--   psql "$DATABASE_URL" -f migrations/0002_provenance_immutable.sql

CREATE OR REPLACE FUNCTION chunks_provenance_is_immutable()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.source_url   IS DISTINCT FROM OLD.source_url
    OR NEW.source_title IS DISTINCT FROM OLD.source_title
    OR NEW.chapter      IS DISTINCT FROM OLD.chapter
    OR NEW.article      IS DISTINCT FROM OLD.article THEN
        RAISE EXCEPTION
            'provenance is immutable after ingest (chunk %)', OLD.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ! UPDATE only, and only the provenance columns. Ingest INSERTs freely, and
-- `IS DISTINCT FROM` lets an UPDATE that leaves provenance alone through — so
-- re-embedding (`UPDATE chunks SET dense_v2 = ...`) and cluster reassignment
-- are unaffected. Verified against the live corpus: a tampered source_title is
-- rejected, a cluster_id write is not.
DROP TRIGGER IF EXISTS chunks_provenance_immutable ON chunks;
CREATE TRIGGER chunks_provenance_immutable
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_provenance_is_immutable();
