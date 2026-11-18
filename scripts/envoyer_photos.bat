@echo off
REM Copie des photos du PC dans la galerie de l'emulateur Android (dossier Pictures/AutoDamage).
REM L'emulateur doit etre demarre.
REM Utilisation (cmd, a la racine du projet) :
REM   scripts\envoyer_photos.bat "C:\chemin\vers\photo.jpg"
REM   scripts\envoyer_photos.bat "C:\chemin\vers\un_dossier"
REM   scripts\envoyer_photos.bat dataset\CarDD_COCO\test2017\000040.jpg

setlocal
set "ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
if not exist "%ADB%" (
    echo adb introuvable : %ADB%
    exit /b 1
)
if "%~1"=="" (
    echo Indiquer une photo ou un dossier, par exemple :
    echo   scripts\envoyer_photos.bat "C:\Users\%USERNAME%\Pictures\voiture.jpg"
    exit /b 1
)
if not exist "%~1" (
    echo Fichier ou dossier introuvable : %~1
    exit /b 1
)

"%ADB%" wait-for-device
"%ADB%" shell mkdir -p /sdcard/Pictures/AutoDamage
if exist "%~1\*" (
    for %%F in ("%~1\*.jpg" "%~1\*.jpeg" "%~1\*.png") do "%ADB%" push "%%~fF" /sdcard/Pictures/AutoDamage/
) else (
    "%ADB%" push "%~1" /sdcard/Pictures/AutoDamage/
)

REM Demande a Android d'indexer les nouvelles photos pour qu'elles apparaissent dans la galerie.
"%ADB%" shell content call --uri content://media --method scan_volume --arg external_primary >nul
echo.
echo Termine : dans l'application, "Choisir dans la galerie" (rubrique Recent ou Albums ^> AutoDamage).
endlocal
