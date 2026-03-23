# AGENTS.md (TheRock_gfx1031)

Diese Datei bleibt bewusst knapp. Sie definiert Kontext, Rollen und Dokument-Prioritaet.
Operative Details gehoeren in die zustaendigen Fachdateien.

## Zweck
- `/media/christoph/some_space/Compute/TheRock_gfx1031` ist das zentrale Custom-ROCm-Repo fuer `gfx1031`.
- `validation/` gehoert zu diesem Repo und ist die Validierungs-/Consumer-Schicht fuer den Build.
- Custom Python-Wheels fuer PyTorch, ONNX Runtime und TensorFlow werden in den jeweiligen Framework-Forks gebaut und von TheRock validiert und konsumiert.

## Startpunkt
- Standard-Workdir: Repo-Root
  `/media/christoph/some_space/Compute/TheRock_gfx1031`

## Dokument-Hierarchie (Quelle der Wahrheit)
1. `README.md`
2. `CUSTOM_ROCM_ARTIFACTS.md`
3. `AI_WORKFLOW_THEROCK_GFX1031.md`
4. `BUILD_EXPERIENCE_NOTES.md`
5. `validation/AI_WORKFLOW_VALIDATION.md`
6. `validation/ANWEISUNG_STRUKTUR.md`

## Architektur-Kurzform
- TheRock baut den Custom-ROCm-Stack.
- `validation/` prueft in-tree und promoted Systemzustand.
- Framework-Forks sind die Source of Truth fuer `tools/rocm_release/` und damit fuer Wheel-Build/Promote.
- Consumer-Repos wie `wakeword` sollen diese Artefakte nur noch nutzen, nicht selbst custom bauen.

## Arbeitsprinzip fuer Agents

Agents sollen:

- zuerst vorhandene Dokumentation lesen
- bestehende Experimente respektieren
- neue Hypothesen nur minimal testen
- bekannte Sackgassen nicht wiederholen

Teure Operationen vermeiden:

- grosse Builds
- parallele Benchmarks
- mehrfach gleiche Experimente

## Regel gegen Redundanz

- Keine operativen Befehle in `AGENTS.md` duplizieren.
- Bei Ablaufaenderungen nur die zustaendige README/Workflow-Datei aktualisieren.
- `AGENTS.md` bleibt ein Navigations- und Prioritaetsdokument.

## Sicherheitsgrenzen

- Keine systemweiten Aenderungen ohne explizite Freigabe.
- Keine destruktiven Git-Operationen ohne explizite Freigabe.

Workflow-Dateiaenderungen in den Framework-Forks sind ein Sonderfall fuer Pushes:

- HTTPS braucht dafuer ein Credential mit `workflow`-Scope.
- SSH ist dafuer weiterhin ein valider manueller Ausweg.

## CI-Hinweis

- In den persoenlichen Framework-Forks sind automatische GitHub-Actions bewusst deaktiviert.
- Workflows sollen dort nur manuell laufen, ausser eine wiederverwendbare `workflow_call`-Datei wird bewusst als Baustein gebraucht.
