% Implementacao da rede Perceptron Simples
% Funcao de ativacao: heaviside (funcao sinal)
% Usando as funcoes built-in (internas) do matlab
%
% Exemplo para disciplina de ICA
% Autor: Guilherme de A. Barreto
% Date: 04/10/2017


% X = Vetor de entrada
% d = saida desejada (escalar)
% W = Matriz de pesos Entrada -> Camada Oculta
% M = Matriz de Pesos Camada Oculta -> Camada saida
% eta = taxa de aprendizagem
% alfa = fator de momento

clear; clc;

% Carrega DADOS
%=================
%dados=load('wine_input.txt');
%alvos=load('wine_target.txt');

dados=load('derm_input.txt');
alvos=load('derm_target.txt');
%dados=load('column_input.txt');
%alvos=load('column_target.txt');

alvos=2*alvos-1;  % Converte alvos de binário [0,1] para bipolar [-1,+1]

% Embaralha vetores de entrada e saidas desejadas
[LinD ColD]=size(dados);

% Normaliza componentes para media zero e variancia unitaria
for i=1:LinD,
	mi=mean(dados(i,:));  % Media das linhas
    di=std(dados(i,:));   % desvio-padrao das linhas 
	dados(i,:)= (dados(i,:) - mi)./di;
end 
Dn=dados;

% Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=0.8;    % Porcentagem usada para treino

% DEFINE ARQUITETURA DA REDE
%=========================
Ne = 100; % No. de epocas de treinamento
Nr = 10;   % No. de rodadas de treinamento/teste
No = 6;   % No. de neuronios na camada de saida

eta=0.01;   % Passo de aprendizagem

%% Inicio do Treino
for r=1:Nr,  % LOOP DE RODADAS TREINO/TESTE

    Rodada=r,

    I=randperm(ColD);
    Dn=Dn(:,I);
    alvos=alvos(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

    J=floor(ptrn*ColD);

    % Vetores para treinamento e saidas desejadas correspondentes
    P = Dn(:,1:J); T1 = alvos(:,1:J);
    [lP cP]=size(P);   % Tamanho da matriz de vetores de treinamento

    % Vetores para teste e saidas desejadas correspondentes
    Q = Dn(:,J+1:end); T2 = alvos(:,J+1:end);
    [lQ cQ]=size(Q);   % Tamanho da matriz de vetores de teste

    % Inicia matrizes de pesos
    WW=0.1*rand(No,lP+1);   % Pesos entrada -> camada saida

    %%% ETAPA DE TREINAMENTO
    for t=1:Ne,
        Epoca=t;
        I=randperm(cP); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento
        EQ=0;
        for tt=1:cP,   % Inicia LOOP de epocas de treinamento
            % CAMADA DE SAIDA
            X  = [-1; P(:,tt)];   % Constroi vetor de entrada com adicao da entrada x0=-1
            Ui = WW * X;          % Ativacao (net) dos neuronios de saida
            Yi = sign(Ui);        % Saidas quantizadas em -1 ou +1

            % CALCULO DO ERRO
            Ei = T1(:,tt) - Yi;           % erro entre a saida desejada e a saida da rede
            EQ = EQ + 0.5*sum(Ei.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA

            WW = WW + eta*Ei*X';  % AJUSTE DOS PESOS - CAMADA DE SAIDA

        end   % Fim de uma epoca

        EQM(t)=EQ/cP;  % MEDIA DO ERRO QUADRATICO POR EPOCA
    end   % Fim do loop de treinamento


    %% ETAPA DE GENERALIZACAO  %%%
    EQ2=0; HID2=[]; OUT2=[];
    for tt=1:cQ,
        % CAMADA OCULTA
        X=[-1; Q(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = sign(Ui);        % Saidas quantizadas em -1 ou +1
        OUT2=[OUT2 Yi];       % Armazena saida da rede

        % CALCULO DO ERRO DE GENERALIZACAO
        Ei = T2(:,tt) - Yi;
        EQ2 = EQ2 + 0.5*sum(Ei.^2);
    end

    % ERRO QUADRATICO MEDIO P/ DADOS DE TESTE
    EQM2=EQ2/cQ;

    % CALCULA TAXA DE ACERTO
    ERRO=T2-OUT2; % matriz de erros
    SumERRO=sum(abs(ERRO)); % Neste vetor, no. de componentes nao-nulas = no. erros
    Nerro=length(find(SumERRO ~= 0));
    
    Tx_erro(r) = 100*Nerro/cQ;  % Taxa de erros por rodada
    Tx_acerto(r)=100-Tx_erro(r); % Taxa de acerto por rodada

end % FIM DO LOOP DE RODADAS TREINO/TESTE

Tx_media=mean(Tx_acerto);  % Taxa media de acerto global
Tx_std=std(Tx_acerto); % Desvio padrao da taxa de acerto 
Tx_mediana=median(Tx_acerto); % Mediana da taxa de acerto 
Tx_min=min(Tx_acerto); % Taxa de acerto minima 
Tx_max=max(Tx_acerto); % Taxa de acerto maxima

STATS=[Tx_media; Tx_std; Tx_mediana; Tx_min; Tx_max]

% Plota Curva de Aprendizagem
plot(EQM)

