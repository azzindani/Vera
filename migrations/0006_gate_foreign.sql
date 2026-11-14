-- The jurisdictions an Indonesian-regulation corpus does not cover (ROADMAP.md Stage 1).
--
-- The engine refuses a query that names one of these unless the retrieved text names it
-- too (pipeline.rs::foreign_unanswered). The check against the text is what keeps
-- Indonesia's own treaties answerable: "ekstradisi Indonesia dengan Australia" retrieves
-- the treaty, and the treaty names Australia.
--
-- ! A fact about the CORPUS, so it lives in corpus_meta.gate, beside the floors that
-- `ravel db calibrate` writes. Applied only where a gate exists: the engine refuses to
-- start on a gate object it cannot read, and a bare {"foreign": [...]} is one.
-- ! Ravel's calibrate does not write this list yet; re-apply after a re-calibration.
--
-- Measured on the 1.7M corpus, 604 cases: off-topic refused 62 -> 74 of 106, real
-- questions refused 0 of 496 before and after.
UPDATE corpus_meta
SET gate = jsonb_set(gate::jsonb, '{foreign}', '["malaysia", "singapura", "singapore", "thailand", "filipina", "philippine", "philippines", "vietnam", "brunei", "myanmar", "kamboja", "laos", "timor leste", "australia", "selandia baru", "new zealand", "jepang", "japan", "japanese", "korea", "tiongkok", "china", "cina", "taiwan", "hong kong", "india", "pakistan", "bangladesh", "arab saudi", "saudi", "uni emirat arab", "emirat", "qatar", "mesir", "egypt", "turki", "turkey", "iran", "amerika serikat", "amerika", "united states", "usa", "kanada", "canada", "meksiko", "brasil", "brazil", "inggris", "united kingdom", "england", "britain", "british", "belanda", "netherlands", "dutch", "jerman", "germany", "prancis", "perancis", "france", "italia", "spanyol", "swedia", "sweden", "norwegia", "denmark", "finlandia", "swiss", "rusia", "russia", "uni eropa", "european union", "eu", "gdpr", "pdpa"]'::jsonb)
WHERE gate IS NOT NULL;
