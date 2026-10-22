# Instala o Escritório de Recrutamento neste computador (Windows 10 ou 11).
# Não rode diretamente: dê dois cliques em instalar.bat, na mesma pasta.
#
# Antes: coloque o arquivo escritorio.env (gerado uma vez com "python -m app.preparar --local") na pasta do
# projeto, ao lado das pastas backend e windows. Pode rodar de novo para atualizar: nada se perde.
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $raiz "backend"
$venv = Join-Path $backend ".venv"
function Passo($texto) { Write-Host ""; Write-Host "==> $texto" -ForegroundColor Cyan }
function Falha($texto) { Write-Host ""; Write-Host $texto -ForegroundColor Red; exit 1 }

# 1. Python 3.11 ou mais novo
Passo "Procurando o Python"
function Achar-Python {
    $opcoes = @(
        @("py", "-3"),
        @("python"),
        @("$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"),
        @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")
    )
    foreach ($o in $opcoes) {
        try {
            $args_ = @()
            if ($o.Count -gt 1) { $args_ = $o[1..($o.Count - 1)] }
            $v = & $o[0] @args_ -c "import sys; print(sys.version_info >= (3, 11))" 2>$null
            if ($LASTEXITCODE -eq 0 -and "$v".Trim() -eq "True") { return ,$o }
        } catch { }
    }
    return $null
}
$py = Achar-Python
if (-not $py) {
    Write-Host "Python não encontrado: instalando o Python 3.12 (só para este usuário)..."
    try {
        winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    } catch { }
    $py = Achar-Python
    if (-not $py) { Falha "Não consegui instalar o Python sozinho. Instale o Python 3.12 em https://www.python.org/downloads/ (marque 'Add python.exe to PATH') e rode o instalador de novo." }
}
$pyArgs = @()
if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }

# 2. Ambiente do programa (fica dentro da pasta backend\.venv)
Passo "Preparando o programa (alguns minutos na primeira vez)"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $py[0] @pyArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { Falha "Não consegui criar o ambiente do programa." }
}
$pyVenv = Join-Path $venv "Scripts\python.exe"
$pywVenv = Join-Path $venv "Scripts\pythonw.exe"
& $pyVenv -m pip install --upgrade pip --quiet --disable-pip-version-check
& $pyVenv -m pip install -r (Join-Path $backend "requirements.txt") --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Falha "Não consegui instalar as dependências (confira a internet e rode de novo)." }

# 3. Configuração (a mesma em todos os computadores do RH)
Passo "Configuração"
$env_ = Join-Path $backend ".env"
$distribuido = Join-Path $raiz "escritorio.env"
if (Test-Path $distribuido) {
    Copy-Item $distribuido $env_ -Force
    Write-Host "Configuração copiada de escritorio.env"
} elseif (-not (Test-Path $env_)) {
    Falha "Falta o arquivo escritorio.env na pasta $raiz. Peça ao responsável pelo sistema (ele é gerado uma vez com: python -m app.preparar --local)."
}
if (-not (Select-String -Path $env_ -Pattern "^ACESSO=local" -Quiet)) {
    Falha "A configuração não é para os computadores do RH (falta ACESSO=local)."
}
# só o usuário deste computador lê a configuração (tem a chave dos currículos)
icacls $env_ /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null

# 4. Atalhos
Passo "Criando os atalhos"
$shell = New-Object -ComObject WScript.Shell
function Atalho($pasta, $nome, $argumentos) {
    $a = $shell.CreateShortcut((Join-Path $pasta "$nome.lnk"))
    $a.TargetPath = $pywVenv
    $a.Arguments = $argumentos
    $a.WorkingDirectory = $backend
    $a.Description = "Escritório de Recrutamento"
    $a.Save()
}
Atalho ([Environment]::GetFolderPath("Desktop")) "Escritório de Recrutamento" "-m app.local"
# liga junto com o Windows (sem janela): a equipe de agentes trabalha mesmo com o escritório fechado
Atalho ([Environment]::GetFolderPath("Startup")) "Escritório de Recrutamento (segundo plano)" "-m app.local --sem-janela"

# 5. Teste rápido e abertura
Passo "Conferindo a conexão com o banco"
& $pyVenv -c "import sys; sys.path.insert(0, r'$backend'); from app.db import engine; from sqlalchemy import text; engine.connect().execute(text('select 1')); print('Banco respondendo.')"
if ($LASTEXITCODE -ne 0) { Falha "O banco não respondeu. Confira a internet e a DATABASE_URL do escritorio.env." }

Passo "Pronto! Abrindo o escritório"
Start-Process -FilePath $pywVenv -ArgumentList "-m app.local" -WorkingDirectory $backend
Write-Host "Da próxima vez, use o atalho 'Escritório de Recrutamento' na área de trabalho."
