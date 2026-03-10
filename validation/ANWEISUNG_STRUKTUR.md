# ANWEISUNG_STRUKTUR.md (validation)

Dieses Dokument ist **verbindlich** fuer die Struktur unter `validation/`.
Ziel: klare Pfade, keine Duplikate, keine Interpretationsspielraeume.

## 1. Geltungsbereich
- `validation/` ist Teil von `TheRock_gfx1031` und **kein** separates Projekt.
- `validation/` validiert den in-tree Build (`<build>/dist/rocm`) und optional reale Workloads.

## 2. Verbindliche Regeln
- Es gibt **genau einen** Python-Codepfad: `validation/src/`.
- User-Entrypoints liegen nur direkt unter `validation/`.
- `validation/scripts/` ist interne Launcher-/Bootstrap-Implementierung, keine User-Oberflaeche.
- Laufzeitdaten liegen nur in `validation/workspace/` (gitignored).
- Caches liegen nur in `validation/workspace/cache/`.
- Build-Artefakte externer Workloads liegen nur in `validation/workspace/builds/`.
- Reports/Logs liegen nur in `validation/workspace/runs/<run_id>/`.
- `validation/_cache/` ist nicht erlaubt (nur als Legacy-Symlink auf `workspace/cache`, falls zwingend noetig).
- Keine Artefakte im Repo-Root, nicht unter `validation/` verteilen.

## 3. Zielstruktur (Soll-Zustand)
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
│  ├─ defaults.yaml
│  ├─ profiles/
│  │  ├─ full.yaml
│  │  ├─ quick.yaml
│  │  └─ airgapped.yaml
│  └─ layout/
│     ├─ gfx_targets.yaml
│     ├─ install_layouts.yaml
│     └─ env_exports.yaml
├─ scripts/
│  ├─ _bootstrap.py
│  ├─ doctor.py
│  ├─ validate.py
│  ├─ cache_gc.py
│  └─ report_open.py
├─ src/
│  ├─ cli/
│  ├─ core/
│  ├─ steps/
│  ├─ assets/
│  └─ data/
├─ workspace/
│  ├─ cache/
│  │  ├─ downloads/
│  │  ├─ git/
│  │  └─ docker/
│  ├─ builds/
│  └─ runs/
│     └─ <run_id>/
│        ├─ logs/
│        ├─ artifacts/
│        ├─ report.json
│        └─ report.html
└─ tests/
   ├─ unit/
   └─ integration/
```

## 4. Abgrenzung der Dokumente (keine Redundanz)
- `ANWEISUNG_STRUKTUR.md`: nur Struktur-/Pfadregeln.
- `AI_WORKFLOW_VALIDATION.md`: operativer Ablauf (wie arbeiten).
- `README.md`: Nutzung/Commands fuer Anwender.
- Details duerfen nicht in mehreren Dateien parallel gepflegt werden.

## 5. ROCm-Kontext (verbindlich)
- Standardfall: Validierung laeuft gegen in-tree ROCm (`<build>/dist/rocm`).
- Systemvalidierung gegen `/opt/rocm` ist nur ueber explizite `*_promoted`-Profile erlaubt.
- Verantwortlich dafuer sind `src/core/tree.py` und `src/core/rocm_env.py`.
- `doctor` bleibt no-download.

## 6. Migration/Drift-Regel
- Wenn sowohl `validation/src/` als auch ein weiterer Paketpfad existieren:
  - sofort auf `validation/src/` konsolidieren.
  - keine neuen Module ausserhalb `src/` anlegen.
