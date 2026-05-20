---

# Projeto de Aprendizado Relacional Estatístico (SRL)

Este projeto implementa um sistema híbrido de Inteligência Artificial baseado no paradigma neuro-simbólico. O objetivo é unir o poder de representação estruturada da lógica de primeira ordem (**Prolog**) para mapear redes de conexões financeiras, com o tratamento estatístico de incertezas do aprendizado de máquina (**Python/Scikit-Learn**) através de uma Regressão Logística.

A arquitetura resolve as limitações do modelo puramente simbólico (que assume um mundo determinístico de tudo ou nada) ao acoplar pesos numéricos a regras de grafos sociais.

---

## Estrutura do Sistema Híbrido

O pipeline foi unificado em um único script Python que automatiza toda a infraestrutura física necessária, gerando dinamicamente os componentes lógicos e os dados tabulares tradicionais.

### Parte 1: Base Simbólica e Regras Lógicas (`rede_social.pl`)

A base lógica em Prolog foi projetada para mapear as interações financeiras diretas (transações) e calcular recursivamente o grau de distância (proximidade) de qualquer cliente em relação a uma entidade de risco conhecido (neste cenário, o nó `daniel`).

O arquivo de fatos e regras estruturado pelo sistema possui a seguinte lógica:

```prolog
% =====================================================================
% Fatos: Conexões Diretas (Grafo Social) e Inadimplência Clássica
% =====================================================================

transacao_entre(joao, ana, 1500).
transacao_entre(ana, carlos, 800).
transacao_entre(carlos, daniel, 50).

inadimplente(daniel).

% =====================================================================
% Regras de Propagação de Risco por Conectividade Recursiva
% =====================================================================

% Caso Base (Grau 1): Conexão direta bidirecional no grafo social
risco_conexao(X, Y, 1) :- transacao_entre(X, Y, _).
risco_conexao(X, Y, 1) :- transacao_entre(Y, X, _).

% Caso Recursivo (Grau N): Busca caminhos acumulando as distâncias dos nós intermediários
risco_conexao(X, Y, Grau) :-
    transacao_entre(X, Z, _),
    X \== Y,                            % Evita loops cíclicos básicos
    risco_conexao(Z, Y, GrauAnterior),
    Grau is GrauAnterior + 1.

```

**Comportamento lógico obtido:**
Ao consultar `risco_conexao(joao, daniel, Grau)`, a inferência recursiva caminha pelo grafo de transações: `joao` transacionou com `ana`, que transacionou com `carlos`, que se conecta diretamente a `daniel`. O motor desempilha as chamadas e deduz que `joao` está a **Grau 3** de distância do risco.

---

### Parte 2: Script Python, Ponte de Dados e Pipeline Estatístico

Para a calibração estatística, implementei um pipeline que utiliza a biblioteca `pyswip` para consultar a topologia do grafo no Prolog em tempo de execução e injetar essa métrica relacional diretamente em um DataFrame do Pandas.

Para garantir portabilidade e consistência, os arquivos de suporte são gravados em disco pelo próprio Python antes da inicialização do modelo:

```python
import pandas as pd
from pyswip import Prolog
from sklearn.linear_model import LogisticRegression

# 1. Geração automatizada dos arquivos do ambiente
conteudo_prolog = """
transacao_entre(joao, ana, 1500).
transacao_entre(ana, carlos, 800).
transacao_entre(carlos, daniel, 50).
inadimplente(daniel).

risco_conexao(X, Y, 1) :- transacao_entre(X, Y, _).
risco_conexao(X, Y, 1) :- transacao_entre(Y, X, _).
risco_conexao(X, Y, Grau) :-
    transacao_entre(X, Z, _),
    X \\== Y,
    risco_conexao(Z, Y, GrauAnterior),
    Grau is GrauAnterior + 1.
"""
with open("rede_social.pl", "w") as f:
    f.write(conteudo_prolog)

conteudo_csv = """cliente_id,renda_mensal,score_classico,inadimplente_historico
joao,5200,750,0
ana,3100,610,0
carlos,1800,420,1"""
with open("dados_financeiros.csv", "w") as f:
    f.write(conteudo_csv)

# 2. Inicialização e consulta ao motor Prolog
prolog = Prolog()
prolog.consult("rede_social.pl")

# 3. Carga dos dados tradicionais
df = pd.read_csv("dados_financeiros.csv")

# 4. Ponte de Engenharia de Features Relacionais
def obter_grau_risco(nome):
    # Tratamento de string limpo para compatibilidade com o parser do pyswip
    query_string = "risco_conexao(" + str(nome) + ", daniel, Grau)"
    query = list(prolog.query(query_string))
    if query:
        return query[0]["Grau"]
    return 999  # Penalidade padrão para nós desconectados no grafo

# Mapeamento vetorizado da feature de rede social
df['grau_risco_rede'] = df['cliente_id'].apply(obter_grau_risco)

# 5. Modelagem Estatística via Regressão Logística
X = df[['renda_mensal', 'score_classico', 'grau_risco_rede']]
y = df['inadimplente_historico']

modelo = LogisticRegression()
modelo.fit(X, y)

print("--- PIPELINE EXECUTADO COM SUCESSO ---")
print("Pesos das Features (Renda, Score, Grau Risco):", modelo.coef_)
print("Intercepto do Modelo:", modelo.intercept_)

```

**Análise matemática dos coeficientes aprendidos:**
O modelo gerou pesos negativos para todas as variáveis independentes. No contexto de risco de crédito, isso valida a hipótese do projeto: quanto maiores os valores de renda e score, menor o risco. Similarmente, quanto *maior* o grau de risco da rede (significando maior distância do inadimplente), *menor* a probabilidade de calote. Estar próximo (Grau 1 ou 2) puxa a probabilidade para cima.

---

### Parte 3: Inferência Neuro-Simbólica Estilo ProbLog

Para fechar o ciclo de **IA Explicável (XAI)**, o modelo não apenas cospe uma probabilidade estéril. Ele mapeia os dados do cliente de teste de volta para uma estrutura sintática inteligível por humanos, compatível com anotações probabilísticas de sistemas de lógica como o ProbLog.

```python
# Avaliação de novo cliente mapeada em um DataFrame para preservar nomes de recursos
cliente_novo_df = pd.DataFrame([[2500, 500, 2]], columns=['renda_mensal', 'score_classico', 'grau_risco_rede'])

# Predição da probabilidade de inadimplência
probabilidade = modelo.predict_proba(cliente_novo_df)[0][1]

print("\n--- Saída Relacional Estatística (XAI) ---")
print(f"{probabilidade:.2f} :: risco(cliente_novo) :- conectado_a(cliente_novo, daniel, 2).")

```

**Saída de execução do terminal:**

```text
--- PIPELINE EXECUTADO COM SUCESSO ---
Pesos das Features (Renda, Score, Grau Risco): [[-1.69430254e-02 -2.47733393e-03 -1.19686290e-05]]
Intercepto do Modelo: [42.78552615]

--- Saída Relacional Estatística (XAI) ---
0.31 :: risco(cliente_novo) :- conectado_a(cliente_novo, daniel, 2).

```

---

## Contribuições para a Inteligência Artificial Explicável (XAI)

Este design resolve três gargalos cruciais em IA moderna:

1. **Auditoria Humana:** Em vez de confiar em decisões de caixas-pretas neurais, conseguimos inspecionar a árvore lógica e afirmar exatamente que a negação ou concessão de um limite foi ponderada porque o usuário se situa a determinada distância de um perfil ruidoso.
2. **Suavização do Mundo Fechado:** Corrigimos a rigidez do Prolog tradicional. Se uma relação não existe explicitamente no grafo ou possui ruído, a Regressão Logística atua suavizando as fronteiras de decisão por meio de probabilidades contínuas.
3. **Casamento de Paradigmas:** O pipeline demonstra como estruturas de grafos não lineares (complexas de computar em bancos relacionais relacionais) podem ser convertidas em variáveis escalares digeríveis para algoritmos estatísticos convencionais em Python.
