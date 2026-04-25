# Versione Base

This folder contains a detached base variant of the project for academic presentation.

Run it with:

```powershell
.\run_versione_base.ps1
```

Or directly:

```powershell
python -m streamlit run versione_base/app/main.py --server.port 8510
```

Notes:
- This variant uses its own `versione_base/cache`, `versione_base/data`, and `versione_base/models` folders.
- It only exposes the main page workflow.
- It removes advanced models that are not part of the base academic scope.
- It filters adult content and blocked genres/tags from search, training data, and recommendations.
