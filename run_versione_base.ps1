Push-Location (Join-Path $PSScriptRoot 'versione_base')
try {
    python -m streamlit run app/main.py --server.port 8510 --server.address 127.0.0.1
} finally {
    Pop-Location
}
