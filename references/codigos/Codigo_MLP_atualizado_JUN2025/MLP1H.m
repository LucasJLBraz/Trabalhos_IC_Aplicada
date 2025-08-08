function [W M Y H]=MLP1H(file_train,file_test,param)
% One-hidden-layered MLP trained with gradient descent plus momentum term
% Hyperbolic acivation function (hidden and output neurons)
% No specific toolboxes are required
%
% INPUTS
% ======
%   FILE_TRAIN: file containing input data 
%   FILE_TEST: file containing output data 
%   PARAM: vector of training parameters
%   PARAM =[Nh Lr Ne Nr Ptrain]
%       Nh: number of hidden neurons
%       Lr: learning rate
%       Nr: number of training/testing runs 
%       0 < Ptrain < 1: percentage of data used for training
%
% OUTPUTS
% =======
%   W: input to hidden layer weight matrix (p+1 x Nh)
%   M: hidden to output layer weight matrix (Nh+1 x m)
%   Y: output neurons activations' matrix (m x N1)
%   H: hidden neurons activations' matrix (Nh x N1)   
%   
%   where N1: number of training pattern vectors
%         m: number of output neurons (i.e. number of classes)
%         p: number of input features
%
% Author: Guilherme de A. Barreto
% Date: 25/12/2008


% X = Vetor de entrada
% d = saida desejada (escalar)
% W = Matriz de pesos Entrada -> Camada Oculta
% M = Matriz de Pesos Camada Oculta -> Camada saida
% eta = taxa de aprendizagem
% alfa = fator de momento

clear; clc;

% Carrega DADOS
%=================
%dados=load('derm_input.txt'); % Vetores (padroes) de entrada
%alvos=load('derm_target.txt'); % Saidas desejadas correspondentes
%alvos=2*alvos - 1;

dados=load('wine_input.txt');
alvos=load('wine_target.txt');
alvos=2*alvos-1;

%dados=load('column_input.txt');
%alvos=load('column_target2.txt');

[LinD ColD]=size(dados);  % LinD = num. de atributos // ColD = num. de exemplos 
[No ~] = size(alvos);   % No. de neuronios na camada de saida

% % Normaliza componentes para media zero e variancia unitaria
for i=1:LinD,
	mi=mean(dados(i,:));  % Media das linhas
    di=std(dados(i,:));   % desvio-padrao das linhas 
	dados(i,:)= (dados(i,:) - mi)./di;
end 
Dn=dados;

% Normaliza componentes para a faixa [-1,+1]
 for i=1:LinD,
 	Xmax=max(dados(i,:));  % Media das linhas
     Xmin=min(dados(i,:));   % desvio-padrao das linhas 
 	dados(i,:)= 2*( (dados(i,:) - Xmin)/(Xmax-Xmin) ) - 1;
 end 
 Dn=dados;

% Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=0.8;    % Porcentagem usada para treino

% DEFINE ARQUITETURA DA REDE
%=========================
Ne = 100; % No. de epocas de treinamento
Nr = 50;   % No. de rodadas de treinamento/teste
Nh = 12;   % No. de neuronios na camada oculta

etai=0.1;   % Passo de aprendizagem inicial
etaf=0.1;   % Passo de aprendizagem final

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
    WW=0.5*(2*rand(Nh,lP+1)-1);   % Pesos entrada -> camada oculta
    MM=0.5*(2*rand(No,Nh+1)-1);   % Pesos camada oculta -> camada de saida

    %%% ETAPA DE TREINAMENTO
    Tmax=Ne*cP;  % No. max. iteracoes de treinamento
    T=0;
    for t=1:Ne,
        Epoca=t;
        I=randperm(cP); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento
        EQ=0;
        for tt=1:cP,   % Inicia LOOP de epocas de treinamento
            % CAMADA OCULTA
            X  = [-1; P(:,tt)];   % Constroi vetor de entrada com adicao da entrada x0=-1
            Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
            Yi = (1-exp(-Ui))./(1+exp(-Ui)); % Saida entre [-1,1]

            % CAMADA DE SAIDA
            Y  = [-1; Yi];        % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
            Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
            Ok = (1-exp(-Uk))./(1+exp(-Uk)); % Saida entre [-1,1]

            % CALCULO DO ERRO
            Ek = T1(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede
            EQ = EQ + 0.5*sum(Ek.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA

            %%% CALCULO DOS GRADIENTES LOCAIS
            Dk = 0.5*(1 - Ok.^2);  % derivada da sigmoide logistica (camada de saida)
            DDk = Ek.*Dk;       % gradiente local (camada de saida)

            Di = 0.5*(1 - Yi.^2); % derivada da sigmoide logistica (camada oculta)
            DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)

            T=(t-1)*cP+tt;   % Iteracao atual
            eta=etai-((etai-etaf)/Tmax)*T;  % Passo de aprendizagem na iteracao T
            
            MM = MM + eta*DDk*Y';      % AJUSTE DOS PESOS - CAMADA DE SAIDA
            WW = WW + eta*DDi*X';    % AJUSTE DOS PESOS - CAMADA OCULTA
        end   % Fim de uma epoca

        EQM(t)=EQ/cP;  % MEDIA DO ERRO QUADRATICO POR EPOCA
    end   % Fim do loop de treinamento


    %% ETAPA DE GENERALIZACAO  %%%
    EQ2=0; HID2=[]; OUT2=[];
    for tt=1:cQ,
        % CAMADA OCULTA
        X = [-1; Q(:,tt)];      % Constroi vetor de entrada (x0=-1)
        Ui = WW * X;            % Ativacao dos neuronios da camada oculta
        Yi = (1-exp(-Ui))./(1+exp(-Ui));

        % CAMADA DE SAIDA
        Y=[-1; Yi];           % Constroi vetor de entrada (y0=-1)
        Uk = MM * Y;          % Ativacao dos neuronios da camada de saida
        Ok = (1-exp(-Uk))./(1+exp(-Uk));
        OUT2=[OUT2 Ok];       % Armazena saida da rede

        % CALCULO DO ERRO DE GENERALIZACAO
        Ek = T2(:,tt) - Ok;
        EQ2 = EQ2 + 0.5*sum(Ek.^2);
    end

    % MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TESTE)
    EQM2=EQ2/cQ;

    % CALCULA TAXA DE ACERTO GLOBAL E MATRIZ DE CONFUSAO
    count_OK=0;  % Zera contador de acertos
    CC=zeros(No); % Inicia matriz de confusao
    for t=1:cQ,
        [T2max Ireal]=max(T2(:,t));  % Indice da saida desejada de maior valor
        [OUT2_max Ipred]=max(OUT2(:,t)); % Indice do neuronio cuja saida eh a maior
        if Ireal==Ipred,   % Acerto se os dois indices coincidem
            count_OK=count_OK+1;
        end
        CC(Ireal,Ipred)=CC(Ireal,Ipred)+1;
    end

    Tx_OK(r)=100*(count_OK/cQ);  % Taxa de acerto global por realizacao

end % FIM DO LOOP DE RODADAS TREINO/TESTE

Tx_media=mean(Tx_OK),  % Taxa media de acerto global
Tx_std=std(Tx_OK), % Desvio padrao da taxa media de acerto 

% Plota Curva de Aprendizagem
plot(EQM)

