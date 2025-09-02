#!/usr/bin/env bash

# Script para executar todos os experimentos do Trabalho 2 em sequência.
# Garante que o ambiente virtual uv seja usado.

# Ativa o ambiente virtual (assumindo que está no diretório .venv)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Ambiente virtual UV ativado."
else
    echo "AVISO: Ambiente virtual .venv não encontrado. Tentando executar com o python do sistema."
fi

# Garante que o diretório de resultados exista
RESULTS_DIR="./results/TC2"
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "Diretório de resultados criado em $RESULTS_DIR"
fi


# Executa os scripts em ordem

echo "\n--- INICIANDO ATIVIDADES 1-4 ---"
python src/tc2_faces_A1_A4.py
if [ $? -ne 0 ]; then echo "Erro na A1-A4"; exit 1; fi

echo "\n--- INICIANDO ATIVIDADES 5-6 ---"
python src/tc2_faces_A5_A6.py
if [ $? -ne 0 ]; then echo "Erro na A5-A6"; exit 1; fi

echo "\n--- INICIANDO ATIVIDADE 7 ---"
python src/tc2_faces_A7.py
if [ $? -ne 0 ]; then echo "Erro na A7"; exit 1; fi

echo "\n--- INICIANDO ATIVIDADE 8 ---"
python src/tc2_faces_A8.py
if [ $? -ne 0 ]; then echo "Erro na A8"; exit 1; fi


echo "\n--- TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS COM SUCESSO ---"
