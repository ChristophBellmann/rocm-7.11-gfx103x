# AGENTS.md (TheRock_gfx1031)

Diese Datei ist bewusst knapp: Sie definiert nur Kontext + Dokument-Priorität.
Details stehen ausschließlich in den jeweiligen Fachdokumenten.

## Zweck
- Dieses Repository ist das zentrale Custom-ROCm-Repo für `gfx1031`.
- `validation/` gehört zu diesem Repo (kein separates Projekt) und enthält den post-build Validierungsworkflow.

## Startpunkt
- Standard-Workdir: Repo-Root  
  `.`

## Dokument-Hierarchie (Quelle der Wahrheit)
1. `README.md` (aktueller Nutzungs-/Ablauffluss)
2. `AI_WORKFLOW_THEROCK_GFX1031.md` (Repo-Arbeitsregeln)
3. `BUILD_EXPERIENCE_NOTES.md` (konkrete Build-Erfahrungen/Fixes)
4. `validation/AI_WORKFLOW_VALIDATION.md` (operativer Validation-Workflow)
5. `validation/ANWEISUNG_STRUKTUR.md` (Struktur-/Namenskonventionen)

## Regel gegen Redundanz
- Keine operativen Details in `AGENTS.md` duplizieren.
- Wenn sich Abläufe ändern, nur die zuständige README/Workflow-Datei aktualisieren.
- `AGENTS.md` bleibt ein Navigations- und Prioritätsdokument.

## Sicherheitsgrenzen
- Keine systemweiten Änderungen und keine destruktiven Git-Operationen ohne explizite Freigabe.
