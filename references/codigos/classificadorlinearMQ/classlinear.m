clear; clc;

%%% CONJUNTO DERMATOLOGIA
X=load('dermato-input.txt');
Y=load('dermato-output.txt');
%Y=2*Y-1;  % Troca "0" por "-1"

%%% CONJUNTO GERMAN (credito bancario)
%X=load('german-input.txt');
%Y=load('german-output.txt');
%Y=2*Y-1;  % Troca "0" por "-1"

%%% CONJUNTO COLUNA VERTEBRAL
%X=load('coluna-input.txt');
%Y=load('coluna-output.txt');
%Y=2*Y-1;  % Troca "0" por "-1"

%%% CONJUNTO IONOSFERA
%X=load('ionosfera-input.txt');
%Y=load('ionosfera-output.txt');
%Y=2*Y-1;  % Troca "0" por "-1"

%%% CONJUNTO WINE (terroir)
X=load('wine-input.txt');
Y=load('wine-output.txt');
%Y=2*Y-1;  % Troca "0" por "-1"

d=size(X);
N=d(2);  % Numero de exemplos no banco de dados
Ptrn=0.7;  % Porcentagem de dados para treino
Ntrn=floor(Ptrn*N);  % Numero de exemplos de teste
Ntst=N-Ntrn;   % Numero de exemplos de teste
Nr=500; % Numero de rodadas treino-teste independentes

for r=1:Nr,  % Inicio do loop da simulacao de Monte Carlo
      rodada=r,

      I=randperm(N); X=X(:,I); Y=Y(:,I);  % embaralhamento dos dados

      % Separacao em dados de treino-teste
      Xtrn=X(:,1:Ntrn); Ytrn=Y(:,1:Ntrn);
      Xtst=X(:,Ntrn+1:end); Ytst=Y(:,Ntrn+1:end);

      %W=Ytrn*pinv(Xtrn);  % Estimacao da matriz de pesos (ou matriz prototipos)
      %W=Ytrn*Xtrn'*inv(Xtrn*Xtrn');
      W=Ytrn/Xtrn;

      Ypred=W*Xtst;    % Predicao da classe dos dados de teste

      Nacertos(r)=evalclassifier(Ytst,Ypred,Ntst);   % Calculo do numero de acertos

      Pacertos(r)=100*(Nacertos(r)/Ntst);  % Taxa de acerto da rodada "r"
      %pause
end

STATS=[mean(Pacertos) std(Pacertos) median(Pacertos) min(Pacertos) max(Pacertos)]








