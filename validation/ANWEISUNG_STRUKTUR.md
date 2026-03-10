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
- Build-Artefakte externer Workloads liegen nur in `validation/workspace/builds/`.
- Reports und Logs liegen nur in `validation/workspace/runs/<run_id>/`.
- Keine neuen workload-spezifischen User-Wrapper unter `validation/scripts/` anlegen.
- Keine Framework-Build-/Promote-Logik unter `validation/scripts/` neu einfuehren; diese gehoert in die zustaendigen Framework-Forks.
- `validation/_cache/` ist nicht erlaubt, ausser als klar dokumentierter Legacy-Symlink auf `workspace/cache`.

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

## 6. Drift-Regeln
- Wenn eine neue oeffentliche Funktion noetig ist, zuerst pruefen, ob sie in `validation/validate.py` als Profil oder Option abbildbar ist.
- Wenn sowohl `validation/src/` als auch ein weiterer Paketpfad fuer dieselbe Logik entstehen, sofort auf `validation/src/` konsolidieren.
- Keine neuen oeffentlichen Entrypoints unterhalb von `validation/scripts/` oder in Unterordnern anlegen.
- Keine Repo-tracked Artefakte ausserhalb der vorgesehenen Root-Dateien, `config/`, `scripts/`, `src/` und `tests/` verteilen.
