# MIOpen Workspace Report

## 1. Fundstelle

- ORT-Session-Erzeugung in [`validation/src/steps/plan.py`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/src/steps/plan.py) nutzt `providers=['ROCMExecutionProvider', 'CPUExecutionProvider']` ohne explizite `gpu_mem_limit`-, `default_memory_arena_cfg`- oder External-Allocator-Konfiguration.
- Der Debug-Helper in [`validation/src/steps/workloads/onnxruntime/piper_tts_debug.py`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/src/steps/workloads/onnxruntime/piper_tts_debug.py) nutzt denselben EP-Pfad und kann ROCm-Provider-Optionen injizieren.
- Die ROCm-EP-Defaults im ORT-Fork:
  - [`onnxruntime_c_api.h`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/cache/git/onnxruntime_rocm711/include/onnxruntime/core/session/onnxruntime_c_api.h): `gpu_mem_limit=SIZE_MAX`, `arena_extend_strategy=0`, `default_memory_arena_cfg=nullptr`
  - [`rocm_execution_provider_info.h`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/rocm_execution_provider_info.h): `miopen_conv_use_max_workspace=true`
- Die harte 32-MiB-Quelle im ORT-ROCm-Code:
  - [`conv.h`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv.h): `AlgoSearchWorkspaceSize = 32 * 1024 * 1024`
  - [`conv.cc`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv.cc): Conv-Search startet bei diesem 32-MiB-Floor
  - [`conv_transpose.cc`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv_transpose.cc): ConvTranspose-Search benutzt direkt `AlgoSearchWorkspaceSize`

## 2. Ursache

Die Warning hängt nicht an einem kleinen ORT-Arena-Limit. Sie entsteht, wenn der Piper-TTS-Graph Conv-/ConvTranspose-Search-Pfade trifft, in denen ORT-ROCm MIOpen nur einen 32-MiB-Search-Workspace übergibt, während MIOpen für einzelne Solver deutlich mehr verlangt.

Wesentliche Aussage:

- Nicht Ursache A: `gpu_mem_limit` oder `arena_extend_strategy`
  - explizit großer `gpu_mem_limit` und geänderte Arena-Strategie ändern die `provided size` nicht
  - `provided size` bleibt `33554432`
- Nicht Ursache B allein: MIOpen-Find-Modus
  - `FAST` unterdrückt in diesem Repro die Warning, aber der Lauf bleibt numerisch/semantisch falsch
  - `NORMAL` und `HYBRID` zeigen weiter die gleiche 32-MiB-Signatur
- Ursache C: konkreter Modell-/Operator-Pfad plus ORT-ROCm-Search-Budget
  - sobald der Piper-Pfad in den betroffenen Conv-/ConvTranspose-Search geht, liegt die Warning an `required workspace > 32 MiB provided`

Kurzform:

> Bestimmte MIOpen-Solver verlangen in diesem Piper-TTS-Graph deutlich mehr als 32 MiB. ORT-ROCm gibt im Search-Pfad trotzdem nur 32 MiB weiter. Das ist die Ursache der `IsEnoughWorkspace`-Warning.

## 3. Testbeleg

Repro-Skript:

- [`validation/scripts/repro_miopen_workspace.py`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/scripts/repro_miopen_workspace.py)

Matrix-Artefakte:

- JSON: [`miopen_workspace_matrix.json`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/debug/miopen_workspace_matrix_run2/miopen_workspace_matrix.json)
- Auto-Report: [`miopen_workspace_report.md`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/debug/miopen_workspace_matrix_run2/miopen_workspace_report.md)

Pflichtmatrix mit `en_US-lessac-low.onnx`, Case `mogli`:

| Case | Änderung | Warning | Provided | Max required | Solver | ROCm-Output |
| --- | --- | --- | --- | --- | --- | --- |
| A | Baseline | 34 | 33554432 | 686292992 | GemmFwdRest, GemmBwdRest | `[1,1,1,765952]` |
| B | `gpu_mem_limit=1<<40` | 34 | 33554432 | 686292992 | GemmFwdRest, GemmBwdRest | `[1,1,1,765952]` |
| C | `arena_extend_strategy=kSameAsRequested` | 34 | 33554432 | 686292992 | GemmFwdRest, GemmBwdRest | `[1,1,1,765952]` |
| D | `MIOPEN_FIND_MODE=NORMAL` | 17 | 33554432 | 686063616 | GemmFwdRest, GemmBwdRest | `[1,1,1,765696]` |
| E | `MIOPEN_FIND_MODE=FAST` | 0 | - | - | - | `[1,1,1,5120]` |
| F | `MIOPEN_FIND_MODE=HYBRID` | 34 | 33554432 | 686292992 | GemmFwdRest, GemmBwdRest | `[1,1,1,765952]` |
| H | `miopen_conv_use_max_workspace=0` | 35 | 33554432 | 686292992 | GemmFwdRest, GemmBwdRest | `[1,1,1,765952]` |
| I | `miopen_conv_use_max_workspace=1` | 34 | 33554432 | 686063616 | GemmFwdRest, GemmBwdRest | `[1,1,1,765696]` |

Abgeleitete Befunde:

- ORT-Memory-Limit-Hypothese widerlegt:
  - A/B/C sind praktisch identisch
  - die Session meldet weiter `gpu_mem_limit=SIZE_MAX` im Baseline-Fall
  - keine External-Allocator-Hooks aktiv
- MIOpen-Find-Modus beeinflusst das Verhalten, aber erklärt die Warning nicht allein:
  - `FAST` vermeidet den Warning-Pfad, ändert aber das Ergebnis drastisch und behebt nichts
  - `NORMAL` reduziert nur die Warning-Anzahl; `provided size` bleibt 32 MiB
- `miopen_conv_use_max_workspace` behebt die Warning ebenfalls nicht:
  - H und I bleiben beide bei `provided size = 33554432`

Damit ist die Warning nicht einfach "mehr Arena geben und gut" und auch nicht "nur ein anderer Find-Modus". Sie ist an den konkreten Search-Pfad des Modells gekoppelt.

## 4. Fix mit Begründung

Der lokale ORT-Fork ist jetzt so gepatcht, dass der Search-Workspace im ROCm-Conv-Pfad nicht mehr stumpf auf `32 MiB` zurückfällt, wenn MIOpen für die konkrete Konfiguration fälschlich `0` meldet.

Geänderte Stellen:

- [`conv.cc`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv.cc)
  - zuerst weiter `miopenConvolutionForwardGetWorkSpaceSize()`
  - wenn MIOpen dort `0` oder Fehler liefert und `miopen_conv_use_max_workspace=1` aktiv ist:
    - nicht mehr `32 MiB`
    - sondern ein aus freiem GPU-Speicher abgeleitetes Search-Budget (`~90%` frei, mindestens `32 MiB`)
- [`conv_transpose.cc`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/cache/git/onnxruntime_rocm711/onnxruntime/core/providers/rocm/nn/conv_transpose.cc)
  - derselbe Fallback-Gedanke für Backward-Data-Search

Warum das der richtige Fix ist:

- Der Debug-Lauf zeigt, dass MIOpen im problematischen Forward-Pfad oft `0` für `miopenConvolutionForwardGetWorkSpaceSize()` und `perf.memory=0` meldet, obwohl dieselbe Solverfamilie kurz darauf real `85 MiB` bis `686 MiB` verlangt.
- Das ist der konkrete Fehlmodus:
  - MIOpen unterberichtet Search-Workspace
  - ORT fiel deshalb bisher auf den historischen `32 MiB`-Floor zurück
  - genau daraus kam `IsEnoughWorkspace`

Post-Fix-Beleg:

- Mit gepatchtem ORT und Default `miopen_conv_use_max_workspace=1`:
  - [`post_fallback_patch_debug_child.log`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/debug/post_fallback_patch_debug_child.log)
  - keine `IsEnoughWorkspace`-Zeilen mehr
  - Debug zeigt stattdessen Search-Budgets um ca. `950 MiB` bis `1.0 GiB` für die bisher problematischen Shapes
- Gegenprobe mit derselben gepatchten Library, aber `miopen_conv_use_max_workspace=0`:
  - [`post_fallback_patch_ws_off_child.log`](/media/christoph/some_space/Compute/TheRock_gfx1031/validation/workspace/debug/post_fallback_patch_ws_off_child.log)
  - wieder `30` Warnings
  - weiter `provided size: 33554432`

Damit ist die Ursache jetzt belastbar isoliert und lokal behoben:

> Die Warning hing nicht an ORT-Arena-Limits und auch nicht nur am MIOpen-Find-Modus. Die eigentliche Ursache war: MIOpen meldet im betroffenen Conv-/ConvTranspose-Search für diese Piper-TTS-Shapes teils keinen nutzbaren Search-Bedarf, und ORT fiel deshalb auf einen festen 32-MiB-Fallback zurück. Der lokale Fix ersetzt diesen starren Fallback im `miopen_conv_use_max_workspace`-Pfad durch ein Search-Budget aus verfügbarem GPU-Speicher.

Abgrenzung:

> Der Workspace-Fix beseitigt die `IsEnoughWorkspace`-Warning, aber nicht automatisch den separaten TTS-Korrektheitsfehler. Dieser bleibt weiterhin unabhängig zu untersuchen.
