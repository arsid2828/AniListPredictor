# Versione base

Variante autonoma per la presentazione accademica. Include la dashboard principale con analisi predittiva e raccomandazioni per anime e manga, un insieme ridotto di modelli e i flussi Cold Start.

Installa le dipendenze indicate nel [README principale](../README.md). Da PowerShell puoi avviare la variante con lo script nella radice del repository.

```powershell
.\run_versione_base.ps1
```

Per avviarla manualmente, esegui questi comandi dalla radice del progetto.

```bash
cd versione_base
python -m streamlit run app/main.py --server.port 8510 --server.address 127.0.0.1
```

La variante usa le proprie cartelle `cache/`, `data/` e `models/` dentro `versione_base/`.
