# ANWEISUNG_STRUKTUR.md (validation)

Dieses Dokument ist **verbindlich** fuer die Struktur unter `validation/`.
Ziel: klare Pfade, eine oeffentliche CLI, keine verdeckten Parallelstrukturen.

## 1. Geltungsbereich
- `validation/` ist Teil von `TheRock_gfx1031` und kein separates Projekt.
- `validation/` ist die Consumer-/Validator-Schicht fuer den in-tree und den explizit promoteten ROCm-Zustand.

## 2. Verbindliche Regeln
- Es gibt **genau einen** Python-Codepfad fuer Validation-Logik: `validation/src/`.
- User-Entrypoints liegen nur direkt unter `validation/`.
- `validation/scripts/` ist interne Launcher-/Bootstrap-Implementierung, keine User-Oberflaeche.
- Laufzeitdaten liegen nur in `validation/workspace/`.
- Caches liegen nur in `validation/workspace/cache/`.
- Framework-Fork-Checkouts fuer Build/Promote liegen nur in `validation/workspace/cache/git/`.
- Build-Artefakte externer Workloads liegen nur in `validation/workspace/builds/`.
- Reports und Logs liegen nur in `validation/workspace/runs/<run_id>/`.
- Externe Modell-Fixtures fuer reale Framework-Diagnostik liegen nur in `validation/workspace/cache/models/`.
- Keine neuen workload-spezifischen User-Wrapper unter `validation/scripts/` anlegen.
- Keine Framework-Build-/Promote-Logik unter `validation/scripts/` neu einfuehren; diese gehoert in die zustaendigen Framework-Forks.
- `validation/_cache/` ist nicht erlaubt.

## 3. Oeffentliche Zielstruktur
```text
validation/
├─ README.md
├─ AI_WORKFLOW_VALIDATION.md
├─ ANWEISUNG_STRUKTUR.md
├─ _launcher.py
├─ validate.py
├─ doctor.py
├─ cache_gc.py
├─ report_open.py
├─ pyproject.toml
├─ requirements-lock.txt
├─ .env.example
├─ .gitignore
├─ config/
├─ scripts/
├─ src/
├─ workspace/
└─ tests/
```

## 4. Dokumentgrenzen
- `validation/README.md`: Nutzerbefehle, Profile, praktische Bedienung.
- `AI_WORKFLOW_VALIDATION.md`: operativer Arbeitsablauf fuer Validation.
- `ANWEISUNG_STRUKTUR.md`: Struktur- und Drift-Regeln.
- `CUSTOM_ROCM_ARTIFACTS.md`: Trennung zwischen Systemartefakten und git-only Fixes.
- Keine operative Kommando-Duplikation in mehreren Dokumenten parallel pflegen.

## 5. ROCm-Kontext
- Standardfall: Validierung laeuft gegen in-tree ROCm (`<build>/dist/rocm`).
- Systemvalidierung gegen `/opt/rocm` ist nur ueber explizite `*_promoted`-Profile erlaubt.
- `doctor` bleibt bewusst no-download und in-tree orientiert.
- Wenn promoted Validation hinzugefuegt oder geaendert wird, muss klar dokumentiert sein, welches Prefix zur Laufzeit erwartet wird.
- Reale Modell-Diagnostik (z. B. Piper-ONNX auf ORT) darf als explizites Profil existieren, wenn der kleine Standard-Smoke die relevante Fehlerklasse nicht abdeckt.
- Solche Realmodell-Profile sollen auch Packaging- und Runtime-Regressionsklassen abdecken, die in Mini-Smokes unsichtbar bleiben, z. B. stale Wheel-Libraries oder MIOpen-Workspace-Fehlverhalten.
- Solche Realmodell-Profile sind nur dann gruen, wenn die GPU-Ausgabe semantisch
  gegen eine CPU-Referenz geprueft wurde; blosses "Session lief ohne Exception"
  ist fuer Validation nicht ausreichend.
- Wenn fuer Realmodell-Triage temporäre ORT-Debug-Overrides noetig sind, muessen sie als explizite Umgebungsvariablen dokumentiert werden und duerfen nicht stillschweigend in den Standard-Smoke wandern.
- Der aktuelle Piper/ORT-Diagnosepfad auf gfx1031 ist deterministisch zu halten:
  - explizites Modell
  - explizite Real-Case-Fixture
  - fester Seed
- Wenn mehrere ROCm-Bugfamilien parallel sichtbar werden, muss die Doku den
  aktuellen Trennstand explizit festhalten, damit spaetere Fixes nicht
  faelschlich als "gesamtes Problem geloest" dokumentiert werden.
- Wenn ein verdaechtiger Operator im isolierten Mini-Repro korrekt laeuft, im
  Vollgraph aber nicht, ist das als Topologie-/Lifetime-/Execution-Order-Befund
  zu dokumentieren und nicht vorschnell als nackter Kernel-Bug zu verbuchen.
- Wenn ein Consumer-Repo denselben Fehler zeigt, bleibt `validation/` trotzdem
  der bevorzugte Untersuchungsort; Consumer-Repros sind nur Sekundaernachweis.

## 6. Drift-Regeln
- Wenn eine neue oeffentliche Funktion noetig ist, zuerst pruefen, ob sie in `validation/validate.py` als Profil oder Option abbildbar ist.
- Wenn sowohl `validation/src/` als auch ein weiterer Paketpfad fuer dieselbe Logik entstehen, sofort auf `validation/src/` konsolidieren.
- Keine neuen oeffentlichen Entrypoints unterhalb von `validation/scripts/` oder in Unterordnern anlegen.
- Keine Repo-tracked Artefakte ausserhalb der vorgesehenen Root-Dateien, `config/`, `scripts/`, `src/` und `tests/` verteilen.
