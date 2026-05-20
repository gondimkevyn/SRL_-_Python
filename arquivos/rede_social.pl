% =====================================================================
% Fatos: Conexões Diretas (Grafo Social) e Inadimplência Clássica
% =====================================================================

% transacao_entre(Origem, Destino, Valor)
transacao_entre(joao, ana, 1500). [cite: 48]
transacao_entre(ana, carlos, 800). [cite: 48]
transacao_entre(carlos, daniel, 50). [cite: 49]

% Histórico de Inadimplência clássico
inadimplente(daniel). [cite: 51]

% =====================================================================
% Regras de Propagação de Risco por Conectividade Recursiva
% =====================================================================

% Caso Base (Grau 1): Conexão direta bidirecional (o grafo não é direcionado)
% Se houve transação entre X e Y, eles estão a Grau 1 de distância.
risco_conexao(X, Y, 1) :- 
    transacao_entre(X, Y, _). [cite: 53, 54]
risco_conexao(X, Y, 1) :- 
    transacao_entre(Y, X, _). [cite: 56, 57]

% Caso Recursivo (Grau N): Encontra um caminho no grafo acumulando a distância
% X está conectado a Y com 'Grau' se X transacionou com um intermediário Z,
% e Z está conectado a Y com 'GrauAnterior'.
risco_conexao(X, Y, Grau) :-
    transacao_entre(X, Z, _),           % X tem conexão direta com Z [cite: 62]
    X \== Y,                            % Evita caminhos cíclicos triviais
    risco_conexao(Z, Y, GrauAnterior),  % Busca recursiva de Z até Y [cite: 63]
    Grau is GrauAnterior + 1.           % Incrementa o grau