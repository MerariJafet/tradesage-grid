#!/bin/bash

# Dashboard de Limpieza del Proyecto - Bot Grid System
# Monitoreo en tiempo real del progreso de optimización

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Función para limpiar pantalla
clear_screen() {
    clear
    printf "\033[2J\033[H"
}

# Función para obtener tamaño en formato legible
get_size() {
    du -sh "$1" 2>/dev/null | cut -f1
}

# Función para obtener tamaño en MB
get_size_mb() {
    du -sm "$1" 2>/dev/null | cut -f1
}

# Función para calcular porcentaje
calculate_percentage() {
    local current=$1
    local total=$2
    echo "scale=1; ($current / $total) * 100" | bc
}

# Función para dibujar barra de progreso
draw_progress_bar() {
    local percentage=$1
    local width=50
    local filled=$(echo "scale=0; ($percentage * $width) / 100" | bc)
    local empty=$((width - filled))
    
    printf "${GREEN}"
    printf "█%.0s" $(seq 1 $filled)
    printf "${WHITE}"
    printf "░%.0s" $(seq 1 $empty)
    printf "${NC}"
    printf " ${BOLD}%.1f%%${NC}" "$percentage"
}

# Función principal de dashboard
show_dashboard() {
    clear_screen
    
    # Tamaños iniciales y objetivos
    INITIAL_SIZE=7200  # 7.2 GB en MB
    TARGET_SIZE=1500   # 1.5 GB objetivo en MB
    CURRENT_SIZE=$(get_size_mb ".")
    
    # Calcular progreso
    SPACE_RECOVERED=$((INITIAL_SIZE - CURRENT_SIZE))
    TOTAL_TO_RECOVER=$((INITIAL_SIZE - TARGET_SIZE))
    PROGRESS=$(calculate_percentage $SPACE_RECOVERED $TOTAL_TO_RECOVER)
    
    # Header
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${WHITE}${BOLD}          BOT GRID SYSTEM - DASHBOARD DE LIMPIEZA                      ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Progreso General
    echo -e "${YELLOW}${BOLD}📊 PROGRESO GENERAL${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    printf "  "
    draw_progress_bar $PROGRESS
    echo ""
    echo -e "  ${WHITE}Tamaño Inicial:${NC}    ${RED}${BOLD}7.2 GB${NC}"
    echo -e "  ${WHITE}Tamaño Actual:${NC}     ${YELLOW}${BOLD}$(get_size .)${NC}"
    echo -e "  ${WHITE}Tamaño Objetivo:${NC}   ${GREEN}${BOLD}1.5 GB${NC}"
    echo -e "  ${WHITE}Espacio Liberado:${NC} ${GREEN}${BOLD}$((SPACE_RECOVERED / 1024)) GB${NC} de $((TOTAL_TO_RECOVER / 1024)) GB"
    echo ""
    
    # Estado de Tareas
    echo -e "${YELLOW}${BOLD}📋 ESTADO DE TAREAS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Task 1: Comprimir real_binance_5d
    if [ -f "data/real_binance_5d_backup.tar.gz" ] && [ ! -d "data/real_binance_5d" ]; then
        echo -e "  ${GREEN}✅${NC} ${WHITE}Comprimir datos real_binance_5d${NC}    ${GREEN}[COMPLETADO]${NC}"
        echo -e "     └─ ${CYAN}$(get_size data/real_binance_5d_backup.tar.gz)${NC} comprimido"
    elif [ -f "data/real_binance_5d_backup.tar.gz" ]; then
        echo -e "  ${YELLOW}⚠️${NC}  ${WHITE}Comprimir datos real_binance_5d${NC}    ${YELLOW}[EN PROCESO]${NC}"
    else
        echo -e "  ${RED}❌${NC} ${WHITE}Comprimir datos real_binance_5d${NC}    ${RED}[PENDIENTE]${NC}"
    fi
    
    # Task 2: Eliminar datasets ML
    if [ ! -d "data/datasets" ] && [ ! -d "data/datasets_v2" ]; then
        echo -e "  ${GREEN}✅${NC} ${WHITE}Eliminar datasets ML antiguos${NC}     ${GREEN}[COMPLETADO]${NC}"
    else
        echo -e "  ${RED}❌${NC} ${WHITE}Eliminar datasets ML antiguos${NC}     ${RED}[PENDIENTE]${NC}"
    fi
    
    # Task 3: Comprimir CSVs
    CSV_GZ_COUNT=$(find data -name "*.csv.gz" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$CSV_GZ_COUNT" -gt 0 ]; then
        echo -e "  ${GREEN}✅${NC} ${WHITE}Comprimir CSVs históricos${NC}        ${GREEN}[COMPLETADO]${NC}"
        echo -e "     └─ ${CYAN}$CSV_GZ_COUNT archivos${NC} comprimidos"
    else
        echo -e "  ${RED}❌${NC} ${WHITE}Comprimir CSVs históricos${NC}        ${RED}[PENDIENTE]${NC}"
    fi
    
    # Task 4: Limpiar .git
    GIT_SIZE=$(get_size_mb ".git")
    if [ "$GIT_SIZE" -lt 500 ]; then
        echo -e "  ${GREEN}✅${NC} ${WHITE}Optimizar repositorio Git${NC}        ${GREEN}[COMPLETADO]${NC}"
        echo -e "     └─ ${CYAN}$(get_size .git)${NC}"
    else
        echo -e "  ${YELLOW}⏳${NC} ${WHITE}Optimizar repositorio Git${NC}        ${YELLOW}[PENDIENTE]${NC}"
        echo -e "     └─ ${RED}$(get_size .git)${NC} (necesita limpieza)"
    fi
    
    # Task 5: Limpiar logs
    LOGS_SIZE=$(get_size_mb "logs")
    if [ "$LOGS_SIZE" -lt 10 ]; then
        echo -e "  ${GREEN}✅${NC} ${WHITE}Limpiar logs antiguos${NC}             ${GREEN}[COMPLETADO]${NC}"
    else
        echo -e "  ${YELLOW}⏳${NC} ${WHITE}Limpiar logs antiguos${NC}             ${YELLOW}[PENDIENTE]${NC}"
        echo -e "     └─ ${RED}$(get_size logs)${NC}"
    fi
    
    echo ""
    
    # Desglose de Directorios
    echo -e "${YELLOW}${BOLD}📁 OCUPACIÓN POR DIRECTORIO${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Obtener top directorios
    echo "  Directorio           Tamaño      Estado"
    echo "  ────────────────── ─────────── ──────────"
    
    # .git
    GIT_SIZE_H=$(get_size ".git")
    if [ "$GIT_SIZE" -gt 1000 ]; then
        echo -e "  ${WHITE}.git${NC}              ${RED}${BOLD}$GIT_SIZE_H${NC}      ${RED}⚠ PESADO${NC}"
    else
        echo -e "  ${WHITE}.git${NC}              ${GREEN}$GIT_SIZE_H${NC}      ${GREEN}✓ OK${NC}"
    fi
    
    # data
    DATA_SIZE=$(get_size_mb "data")
    DATA_SIZE_H=$(get_size "data")
    if [ "$DATA_SIZE" -gt 1000 ]; then
        echo -e "  ${WHITE}data/${NC}             ${YELLOW}${BOLD}$DATA_SIZE_H${NC}      ${YELLOW}⚠ REVISAR${NC}"
    else
        echo -e "  ${WHITE}data/${NC}             ${GREEN}$DATA_SIZE_H${NC}      ${GREEN}✓ OK${NC}"
    fi
    
    # .venv
    VENV_SIZE_H=$(get_size ".venv")
    echo -e "  ${WHITE}.venv/${NC}            ${CYAN}$VENV_SIZE_H${NC}      ${CYAN}ℹ NORMAL${NC}"
    
    # frontend
    FRONTEND_SIZE_H=$(get_size "frontend")
    echo -e "  ${WHITE}frontend/${NC}         ${CYAN}$FRONTEND_SIZE_H${NC}      ${CYAN}ℹ NORMAL${NC}"
    
    # logs
    LOGS_SIZE_H=$(get_size "logs")
    if [ "$LOGS_SIZE" -gt 20 ]; then
        echo -e "  ${WHITE}logs/${NC}             ${YELLOW}$LOGS_SIZE_H${NC}       ${YELLOW}⚠ LIMPIAR${NC}"
    else
        echo -e "  ${WHITE}logs/${NC}             ${GREEN}$LOGS_SIZE_H${NC}       ${GREEN}✓ OK${NC}"
    fi
    
    # models
    MODELS_SIZE_H=$(get_size "models")
    echo -e "  ${WHITE}models/${NC}           ${CYAN}$MODELS_SIZE_H${NC}       ${CYAN}ℹ NORMAL${NC}"
    
    echo ""
    
    # Próximas Acciones
    echo -e "${YELLOW}${BOLD}🎯 PRÓXIMAS ACCIONES RECOMENDADAS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ "$GIT_SIZE" -gt 1000 ]; then
        echo -e "  ${RED}1.${NC} ${WHITE}Limpiar historial Git${NC} (puede recuperar ~${RED}2 GB${NC})"
        echo -e "     ${CYAN}→ git filter-branch o BFG Repo-Cleaner${NC}"
    fi
    
    if [ "$LOGS_SIZE" -gt 20 ]; then
        echo -e "  ${YELLOW}2.${NC} ${WHITE}Rotar logs antiguos${NC} (puede recuperar ~${YELLOW}$((LOGS_SIZE - 5)) MB${NC})"
        echo -e "     ${CYAN}→ find logs/ -name '*.log' -mtime +7 -delete${NC}"
    fi
    
    if [ -d "frontend/node_modules" ]; then
        echo -e "  ${YELLOW}3.${NC} ${WHITE}Agregar node_modules/ a .gitignore${NC}"
    fi
    
    if [ -d ".venv" ]; then
        echo -e "  ${YELLOW}4.${NC} ${WHITE}Agregar .venv/ a .gitignore${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${WHITE} Última actualización: $(date '+%Y-%m-%d %H:%M:%S')                              ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║${WHITE} Presiona Ctrl+C para salir | Actualización automática cada 5s       ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
}

# Loop principal
main() {
    while true; do
        show_dashboard
        sleep 5
    done
}

# Trap para salida limpia
trap 'clear_screen; echo -e "\n${GREEN}Dashboard cerrado.${NC}\n"; exit 0' INT TERM

# Ejecutar
main
