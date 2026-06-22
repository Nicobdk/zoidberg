@echo off
REM Script pour générer toutes les visualisations manquantes
REM Projet Zoidberg

echo ========================================
echo   GENERATION DES VISUALISATIONS
echo ========================================
echo.

REM Activer l'environnement
call venv\Scripts\activate.bat

echo [1/2] Generation du Flow Diagram...
python generate_flow_diagram.py
if errorlevel 1 (
    echo [ERREUR] Flow diagram a echoue
) else (
    echo [OK] Flow diagram genere !
)

echo.
echo [2/2] Generation de l'Error Analysis...
python generate_error_analysis.py
if errorlevel 1 (
    echo [ERREUR] Error analysis a echoue
) else (
    echo [OK] Error analysis genere !
)

echo.
echo ========================================
echo   TERMINE !
echo ========================================
echo.
echo Consulte reports/figures/ pour voir les resultats
pause
