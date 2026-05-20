Com base no material fornecido da disciplina do Prof. Edjard Mota, o objetivo do projeto é construir um sistema híbrido de **Aprendizado Relacional Estatístico (SRL)**. Esse sistema une o poder de representação estruturada do **Prolog** (para tratar grafos sociais complexos) com a calibração numérica de incertezas do **Python** (via Regressão Logística no Scikit-Learn).

Abaixo está o guia passo a passo detalhado e explicado para a resolução completa do projeto, dividido entre os dois arquivos entregáveis e pronto para ser executado.

---

## Parte 1: Base de Fatos e Regras Lógicas (`rede_social.pl`)

O objetivo desta parte em Prolog é mapear as interações financeiras e calcular de forma **recursiva** o grau de distância (proximidade) de qualquer usuário em relação a uma entidade de risco conhecido (no exemplo fornecido, o `daniel`).

O código do slide original possui pequenas falhas de sintaxe e truncamentos devido à extração de texto (como `1) :- ...` e variáveis anônimas incompletas). Abaixo está a implementação corrigida, limpa e funcional:

```prolog
% =====================================================================
% Fatos: Conexões Diretas (Grafo Social) e Inadimplência Clássica
% =====================================================================

% transacao_entre(Origem, Destino, Valor)
transacao_entre(joao, ana, 1500).
transacao_entre(ana, carlos, 800).
transacao_entre(carlos, daniel, 50).

% Histórico de Inadimplência clássico
inadimplente(daniel).

% =====================================================================
% Regras de Propagação de Risco por Conectividade Recursiva
% =====================================================================

% Caso Base (Grau 1): Conexão direta bidirecional (o grafo não é direcionado)
% Se houve transação entre X e Y, eles estão a Grau 1 de distância.
risco_conexao(X, Y, 1) :- 
    transacao_entre(X, Y, _).
risco_conexao(X, Y, 1) :- 
    transacao_entre(Y, X, _).

% Caso Recursivo (Grau N): Encontra um caminho no grafo acumulando a distância
% X está conectado a Y com 'Grau' se X transacionou com um intermediário Z,
% e Z está conectado a Y com 'GrauAnterior'.
risco_conexao(X, Y, Grau) :-
    transacao_entre(X, Z, _),           % X tem conexão direta com Z
    X \== Y,                            % Evita caminhos cíclicos triviais
    risco_conexao(Z, Y, GrauAnterior),  % Busca recursiva de Z até Y
    Grau is GrauAnterior + 1.           % Incrementa o grau

```

### Explicação do comportamento lógico:

Se consultarmos `risco_conexao(joao, daniel, Grau).`, o motor do Prolog funcionará da seguinte forma:

1. `joao` transacionou com `ana` (Z = ana).
2. `ana` transacionou com `carlos` (Z = carlos).
3. `carlos` transacionou diretamente com `daniel` (caso base, Grau = 1).
4. Desempilhando a recursão, `ana` está a Grau 2 de `daniel` e `joao` está a **Grau 3**.

---

## Parte 2: Script Python, Ponte e Pipeline Estatístico

No ambiente Python, você lerá os dados financeiros tradicionais coletados de um arquivo CSV (atributos individuais como renda e score clássico) e usará a biblioteca `pyswip` para injetar o componente relacional (o grau de risco extraído do Prolog) como uma nova *feature*.

Primeiro, certifique-se de simular ou ter o arquivo estruturado `dados_financeiros.csv` no mesmo diretório:

```csv
cliente_id,renda_mensal,score_classico,inadimplente_historico
joao,5200,750,0
ana,3100,610,0
carlos,1800,420,1

```

Abaixo está o pipeline completo de engenharia de recursos e treinamento do modelo:

```python
import pandas as pd
from pyswip import Prolog
from sklearn.linear_model import LogisticRegression

# 1. Inicializar e conectar à base lógica do Prolog
prolog = Prolog()
prolog.consult("rede_social.pl")  # Carrega o arquivo feito na Parte 1

# 2. Carregar o dataset financeiro tradicional com Pandas
df = pd.read_csv("dados_financeiros.csv")

# 3. Função de extração de features lógicas (A Ponte)
def obter_grau_risco(nome):
    # Executa a consulta dinâmica no motor Prolog em relação ao alvo 'daniel'
    query = list(prolog.query(f"risco_conexao({nome}, daniel, Grau)"))
    
    if query:
        # Extrai o valor associado à variável unificada 'Grau'
        return query[0]["Grau"]
    else:
        # Penalidade padrão caso não haja nenhuma conexão identificada no grafo
        return 999

# Aplica a função de forma vetorizada criando a nova feature relacional
df['grau_risco_rede'] = df['cliente_id'].apply(obter_grau_risco)

# 4. Preparação dos dados para o modelo estatístico
# Recursos (X): Combinação de dados numéricos clássicos e a estrutura de rede do Prolog
X = df[['renda_mensal', 'score_classico', 'grau_risco_rede']]
y = df['inadimplente_historico']

# 5. Treinamento da Regressão Logística (Calibração Numérica)
modelo = LogisticRegression()
modelo.fit(X, y)

print("--- Treinamento Concluído ---")
print("Coeficientes Aprendidos (Pesos):", modelo.coef_)
print("Intercepto:", modelo.intercept_)

```

### Explicação do comportamento estatístico:

A Regressão Logística analisa os dados históricos. Ela perceberá que quanto menor o `grau_risco_rede` (ou seja, quanto mais próximo de uma pessoa inadimplente o cliente está), maior é o risco real dele se tornar inadimplente também. O algoritmo calibra matematicamente o peso exato desse impacto em relação à renda e ao score.

---

## Parte 3: Saída Relacional Estatística (Inferência Estilo ProbLog)

Para fechar o paradigma neuro-simbólico e atender aos requisitos de **IA Explicável (XAI)**, o sistema não deve apenas retornar um número abstrato. Ele deve gerar uma regra formatada que combine a estrutura lógica com a probabilidade estatística calculada pelo modelo matemático.

```python
# Suponha um novo cliente que entra no sistema
# Atributos: Renda = 2500, Score = 500, Grau de proximidade com Daniel = 2
cliente_novo_x = [[2500, 500, 2]]

# Prediz a probabilidade de pertencer à classe 1 (Inadimplente)
prob_inadimplencia = modelo.predict_proba(cliente_novo_x)[0][1]

# Formata a saída no formato padrão de cláusulas probabilísticas (Estilo ProbLog)
print("\n--- Saída Relacional Estatística (XAI) ---")
print(f"{prob_inadimplencia:.2f} :: risco(cliente_novo) :- conectado_a(cliente_novo, daniel, 2).")

```

Se o modelo calcular que o risco é de, por exemplo, 0.82, a saída gerada será exatamente:
`0.82 :: risco(cliente_novo) :- conectado_a(cliente_novo, daniel, 2).`

---

## 💡 Análise Crítica e Critérios de Sucesso (Para a Rubrica)

Ao finalizar o relatório ou a apresentação do seu projeto, certifique-se de ressaltar estes três pontos cruciais exigidos na rubrica de avaliação:

1. **Explicabilidade (XAI) e Auditoria:** Sistemas baseados puramente em Deep Learning funcionam como caixas-pretas. Ao usar SRL (Statistical Relational Learning), se um cliente tiver seu crédito negado, você pode auditar e explicar exatamente o motivo humano: *"O crédito foi negado porque você possui score X, e está a apenas 2 graus de distância de uma transação com uma entidade inadimplente ativa"*.
2. **Tratamento do Ruído:** O Prolog puro trabalha sob a hipótese de mundo fechado (se não está na base, o risco é zero ou erro). A inteligência híbrida mitiga isso usando a probabilidade da Regressão Logística para suavizar as arestas rígidas da lógica booleana tradicional.
3. **Ponte de Dados:** O `pyswip` atua como o tradutor que transforma estruturas de grafos complexas (difíceis de tratar em tabelas SQL comuns) em dados numéricos lineares vetorizados (`grau_risco_rede`) assimiláveis pelo Pandas e Scikit-Learn.
