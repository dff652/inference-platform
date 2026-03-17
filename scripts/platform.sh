#!/bin/bash
# Inference Platform Service Manager
# Usage: ./scripts/platform.sh [start|stop|restart|status|logs|enable|disable]
#
# Services managed:
#   inference-backend  - FastAPI backend (:8100)
#   inference-celery   - Celery async worker
#   chatts-serve       - ChatTS-8B GPU serve (:8001)
#   qwen-serve         - Qwen3-VL-8B GPU serve (:8002)

set -e

SERVICES=(inference-backend inference-celery chatts-serve qwen-serve)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    echo "Usage: $0 {start|stop|restart|status|logs|enable|disable} [service]"
    echo ""
    echo "Commands:"
    echo "  start    - Start all services (or specific service)"
    echo "  stop     - Stop all services (or specific service)"
    echo "  restart  - Restart all services (or specific service)"
    echo "  status   - Show status of all services"
    echo "  logs     - Show recent logs (use: $0 logs [service] [-f])"
    echo "  enable   - Enable auto-start on login"
    echo "  disable  - Disable auto-start on login"
    echo ""
    echo "Services: ${SERVICES[*]}"
    exit 1
}

get_services() {
    if [ -n "$1" ] && [ "$1" != "-f" ]; then
        echo "$1"
    else
        echo "${SERVICES[@]}"
    fi
}

cmd_start() {
    local svcs
    read -ra svcs <<< "$(get_services "$1")"
    for svc in "${svcs[@]}"; do
        echo -n "Starting $svc... "
        systemctl --user start "$svc" 2>/dev/null && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAILED${NC}"
    done
}

cmd_stop() {
    local svcs
    read -ra svcs <<< "$(get_services "$1")"
    for svc in "${svcs[@]}"; do
        echo -n "Stopping $svc... "
        systemctl --user stop "$svc" 2>/dev/null && echo -e "${GREEN}OK${NC}" || echo -e "${YELLOW}not running${NC}"
    done
}

cmd_restart() {
    local svcs
    read -ra svcs <<< "$(get_services "$1")"
    for svc in "${svcs[@]}"; do
        echo -n "Restarting $svc... "
        systemctl --user restart "$svc" 2>/dev/null && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAILED${NC}"
    done
}

cmd_status() {
    echo "=== Inference Platform Services ==="
    echo ""
    for svc in "${SERVICES[@]}"; do
        local state
        state=$(systemctl --user is-active "$svc" 2>/dev/null || true)
        local enabled
        enabled=$(systemctl --user is-enabled "$svc" 2>/dev/null || true)

        local color="$RED"
        [ "$state" = "active" ] && color="$GREEN"
        [ "$state" = "activating" ] && color="$YELLOW"

        printf "  %-22s ${color}%-12s${NC} (enabled: %s)\n" "$svc" "$state" "$enabled"
    done

    echo ""
    echo "=== Health Checks ==="
    echo ""

    for url in "http://localhost:8100/health:Backend" "http://localhost:8001/health:ChatTS" "http://localhost:8002/health:Qwen-VL"; do
        local endpoint="${url%%:*}:${url#*:}"
        endpoint="${url%:*}"
        local name="${url##*:}"
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "$endpoint" 2>/dev/null || echo "000")

        local healthy="down"
        [ "$http_code" = "200" ] && healthy="ok"

        local color="$RED"
        [ "$healthy" = "ok" ] && color="$GREEN"

        printf "  %-22s ${color}%s${NC}\n" "$name" "$healthy"
    done
    echo ""
}

cmd_logs() {
    local svc="${1:-inference-backend}"
    shift 2>/dev/null || true
    local follow=""
    [ "$1" = "-f" ] && follow="-f"
    journalctl --user -u "$svc" --no-pager -n 50 $follow
}

cmd_enable() {
    systemctl --user daemon-reload
    local svcs
    read -ra svcs <<< "$(get_services "$1")"
    for svc in "${svcs[@]}"; do
        echo -n "Enabling $svc... "
        systemctl --user enable "$svc" 2>/dev/null && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAILED${NC}"
    done
    echo ""
    echo "Note: Run 'loginctl enable-linger $USER' (as root) for services to start without login."
}

cmd_disable() {
    local svcs
    read -ra svcs <<< "$(get_services "$1")"
    for svc in "${svcs[@]}"; do
        echo -n "Disabling $svc... "
        systemctl --user disable "$svc" 2>/dev/null && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAILED${NC}"
    done
}

case "${1:-}" in
    start)   cmd_start "$2" ;;
    stop)    cmd_stop "$2" ;;
    restart) cmd_restart "$2" ;;
    status)  cmd_status ;;
    logs)    cmd_logs "$2" "$3" ;;
    enable)  cmd_enable "$2" ;;
    disable) cmd_disable "$2" ;;
    *)       usage ;;
esac
