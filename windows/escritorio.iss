; Instalador do Escritório de Recrutamento para Windows (Inno Setup 6).
; Um instalador só para todas as empresas: o que muda é a configuração, escolhida numa página do instalador e
; guardada em %APPDATA%\EscritorioRecrutamento. Nenhum segredo vai dentro do programa.
; A empresa recebe a configuração lacrada (empresa.escritorio, de "python -m app.preparar --lacrar") e um código de
; ativação: o Ativador confere o código e guarda a configuração cifrada pelo Windows (escritorio.cfg), sem nenhum
; arquivo legível no computador. Um escritorio.env legível ainda é aceito (computador do responsável, TI).
;
;   pyinstaller windows\escritorio.spec          (gera dist\EscritorioRecrutamento)
;   iscc windows\escritorio.iss                  (gera dist-instalador\EscritorioRecrutamento-Setup.exe)
;
; Instalação sem perguntas (ex.: TI da empresa):
;   EscritorioRecrutamento-Setup.exe /VERYSILENT /CONFIG="C:\caminho\empresa.escritorio" /CODIGO=XXXX-XXXX-XXXX-XXXX-XXXX

#define Nome "Escritório de Recrutamento"
#define Exe "EscritorioRecrutamento.exe"
#ifndef Versao
  #define Versao "1.0.0"
#endif

[Setup]
AppId={{6E5A2C1B-3D47-4E8F-9A21-5B7C0D9E8F14}
AppName={#Nome}
AppVersion={#Versao}
AppVerName={#Nome} {#Versao}
DefaultDirName={localappdata}\Programs\EscritorioRecrutamento
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist-instalador
OutputBaseFilename=EscritorioRecrutamento-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#Nome}
UninstallDisplayIcon={app}\{#Exe}
SetupIconFile=icone.ico
WizardSmallImageFile=icone-instalador-55.bmp,icone-instalador-110.bmp
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "pt"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Files]
Source: "..\dist\EscritorioRecrutamento\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; só usado durante a instalação (página do código de ativação); não fica no computador
Source: "..\dist\Ativador.exe"; Flags: dontcopy

[Icons]
Name: "{userdesktop}\{#Nome}"; Filename: "{app}\{#Exe}"
Name: "{userprograms}\{#Nome}"; Filename: "{app}\{#Exe}"
; liga junto com o Windows, sem janela: a equipe de agentes trabalha mesmo com o escritório fechado
Name: "{userstartup}\{#Nome} (segundo plano)"; Filename: "{app}\{#Exe}"; Parameters: "--sem-janela"

[Run]
Filename: "{app}\{#Exe}"; Description: "Abrir o escritório agora"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM {#Exe}"; Flags: runhidden; RunOnceId: "FecharPrograma"

[Code]
var
  PaginaConfig: TInputFileWizardPage;
  PaginaCodigo: TInputQueryWizardPage;
  Ativado: Boolean;

function SetEnvironmentVariable(lpName: String; lpValue: String): BOOL;
  external 'SetEnvironmentVariableW@kernel32.dll stdcall';

function PastaConfig: String;
begin
  Result := ExpandConstant('{userappdata}\EscritorioRecrutamento');
end;

function ConfigDestino: String;
begin
  Result := PastaConfig + '\escritorio.env';
end;

function JaConfigurado: Boolean;
begin
  Result := FileExists(PastaConfig + '\escritorio.cfg') or FileExists(ConfigDestino);
end;

function EhPacote: Boolean;
begin
  Result := CompareText(ExtractFileExt(PaginaConfig.Values[0]), '.escritorio') = 0;
end;

{ 0 = ok, 2 = código errado, outro = arquivo inválido. O código vai por variável de ambiente, não na linha de comando. }
function Ativar(const Codigo: String): Integer;
var
  Saida: Integer;
begin
  if not FileExists(ExpandConstant('{tmp}\Ativador.exe')) then
    ExtractTemporaryFile('Ativador.exe');
  SetEnvironmentVariable('ESCRITORIO_CODIGO', Codigo);
  if not Exec(ExpandConstant('{tmp}\Ativador.exe'), '"' + PaginaConfig.Values[0] + '"', '', SW_HIDE,
              ewWaitUntilTerminated, Saida) then
    Saida := 3;
  SetEnvironmentVariable('ESCRITORIO_CODIGO', '');
  Result := Saida;
end;

function ErroDaAtivacao(Saida: Integer): String;
begin
  if Saida = 2 then
    Result := 'Código de ativação errado. Confira o código que veio junto com o arquivo e digite de novo.'
  else
    Result := 'Não consegui abrir este arquivo de configuração. Confira se é o arquivo .escritorio que você recebeu.';
end;

procedure InitializeWizard;
begin
  PaginaConfig := CreateInputFilePage(wpWelcome,
    'Configuração da empresa',
    'Escolha o arquivo .escritorio enviado pelo responsável pelo sistema.',
    'Ele liga este computador ao banco de dados da sua empresa. Se este computador já tem uma configuração, ' +
    'deixe em branco para mantê-la.');
  PaginaConfig.Add('Arquivo de configuração:',
    'Configuração da empresa (*.escritorio)|*.escritorio|Configuração legível (*.env)|*.env|Todos os arquivos|*.*',
    '.escritorio');
  PaginaConfig.Values[0] := ExpandConstant('{param:CONFIG|}');

  PaginaCodigo := CreateInputQueryPage(PaginaConfig.ID,
    'Código de ativação',
    'Digite o código de ativação que você recebeu.',
    'O código chega separado do arquivo (por exemplo, por WhatsApp). Pode digitar com ou sem os traços.');
  PaginaCodigo.Add('Código de ativação:', False);
  PaginaCodigo.Values[0] := ExpandConstant('{param:CODIGO|}');
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = PaginaCodigo.ID) and not EhPacote;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Saida: Integer;
begin
  Result := True;
  if CurPageID = PaginaConfig.ID then
  begin
    if PaginaConfig.Values[0] = '' then
    begin
      if not JaConfigurado then
      begin
        SuppressibleMsgBox('Escolha o arquivo .escritorio: sem ele o escritório não sabe a qual empresa se ligar.', mbError, MB_OK, IDOK);
        Result := False;
      end;
    end
    else if not FileExists(PaginaConfig.Values[0]) then
    begin
      SuppressibleMsgBox('Não encontrei esse arquivo. Confira o caminho.', mbError, MB_OK, IDOK);
      Result := False;
    end;
    Ativado := False;
  end
  else if CurPageID = PaginaCodigo.ID then
  begin
    if Trim(PaginaCodigo.Values[0]) = '' then
    begin
      SuppressibleMsgBox('Digite o código de ativação.', mbError, MB_OK, IDOK);
      Result := False;
      exit;
    end;
    Saida := Ativar(PaginaCodigo.Values[0]);
    Ativado := Saida = 0;
    if not Ativado then
    begin
      SuppressibleMsgBox(ErroDaAtivacao(Saida), mbError, MB_OK, IDOK);
      Result := False;
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Codigo: Integer;
begin
  Result := '';
  { instalação sem perguntas (/CONFIG e /CODIGO): a ativação acontece aqui; se falhar, nada é instalado }
  if EhPacote and not Ativado then
  begin
    Codigo := Ativar(PaginaCodigo.Values[0]);
    Ativado := Codigo = 0;
    if not Ativado then
    begin
      Result := ErroDaAtivacao(Codigo);
      exit;
    end;
  end;
  { o programa pode estar rodando em segundo plano: fecha antes de trocar os arquivos }
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM {#Exe}', '', SW_HIDE, ewWaitUntilTerminated, Codigo);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (PaginaConfig.Values[0] <> '') and not EhPacote then
  begin
    ForceDirectories(PastaConfig);
    { a configuração legível escolhida agora passa a valer no lugar de uma lacrada antiga }
    DeleteFile(PastaConfig + '\escritorio.cfg');
    if not FileCopy(PaginaConfig.Values[0], ConfigDestino, False) then
      SuppressibleMsgBox('Não consegui copiar a configuração para ' + ConfigDestino + '.', mbError, MB_OK, IDOK);
  end;
end;
