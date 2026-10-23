; Instalador do Escritório de Recrutamento para Windows (Inno Setup 6).
; Um instalador só para todas as empresas: o que muda é o arquivo de configuração (escritorio.env), escolhido
; numa página do instalador e guardado em %APPDATA%\EscritorioRecrutamento. Nenhum segredo vai dentro do programa.
;
;   pyinstaller windows\escritorio.spec          (gera dist\EscritorioRecrutamento)
;   iscc windows\escritorio.iss                  (gera dist-instalador\EscritorioRecrutamento-Setup.exe)
;
; Instalação sem perguntas (ex.: TI da empresa):  EscritorioRecrutamento-Setup.exe /VERYSILENT /CONFIG="C:\caminho\escritorio.env"

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

function ConfigDestino: String;
begin
  Result := ExpandConstant('{userappdata}\EscritorioRecrutamento\escritorio.env');
end;

procedure InitializeWizard;
begin
  PaginaConfig := CreateInputFilePage(wpWelcome,
    'Configuração da empresa',
    'Escolha o arquivo escritorio.env enviado pelo responsável pelo sistema.',
    'Ele liga este computador ao banco de dados da sua empresa. Se este computador já tem uma configuração, ' +
    'deixe em branco para mantê-la.');
  PaginaConfig.Add('Arquivo de configuração:', 'Configuração (*.env)|*.env|Todos os arquivos|*.*', '.env');
  PaginaConfig.Values[0] := ExpandConstant('{param:CONFIG|}');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = PaginaConfig.ID then
  begin
    if PaginaConfig.Values[0] = '' then
    begin
      if not FileExists(ConfigDestino) then
      begin
        MsgBox('Escolha o arquivo escritorio.env: sem ele o escritório não sabe a qual empresa se ligar.', mbError, MB_OK);
        Result := False;
      end;
    end
    else if not FileExists(PaginaConfig.Values[0]) then
    begin
      MsgBox('Não encontrei esse arquivo. Confira o caminho.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Codigo: Integer;
begin
  { o programa pode estar rodando em segundo plano: fecha antes de trocar os arquivos }
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM {#Exe}', '', SW_HIDE, ewWaitUntilTerminated, Codigo);
  Result := '';
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (PaginaConfig.Values[0] <> '') then
  begin
    ForceDirectories(ExtractFileDir(ConfigDestino));
    if not FileCopy(PaginaConfig.Values[0], ConfigDestino, False) then
      MsgBox('Não consegui copiar a configuração para ' + ConfigDestino + '.', mbError, MB_OK);
  end;
end;
