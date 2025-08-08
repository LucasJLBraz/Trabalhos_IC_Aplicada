% Compressing a signal with PCA and reconstruct it.
%
% Last modification: 28/07/2009
% Author: Guilherme Barreto

clear; clc; close all

pkg load statistics
pkg load signal

% Le a imagem (imread) e converte (im2double) para double precision
[Y Fs]=audioread('kissing.wav');

X=buffer(Y,50);  % Criando vetores

%%%%%%%% PCA %%%%%%%%%%%
Cx=cov(X');  % matriz de covariancia
[V L]=eig(Cx);
L=diag(L);
[L I]=sort(L,'descend'); % Autovalores em ordem decrescente
V=V(:,I);  % Autovetores ordenados do maior para menor autovalor

%figure; bar(L);  % grafico da amplitude dos autovalores
SL=sum(L);  % Soma dos autovalores
aux=0;
for i=1:length(L),
    aux=aux+L(i);
    VE(i)=aux/SL;   % Variancia explicada
end

figure; plot(VE,'linewidth',3);
grid; set(gca, "fontsize", 16)

% escolha dos q maiores autovalores (componentes principais)
tol=0.90;  % tolerancia para VE aceitavel
q=length(find(VE<=tol));

% Matriz de transformacao resultante
Vq=V(:,1:q);

Q=Vq';

% Vetores transformados com blocos de tamanho K
Y=Q*X;

% Vetores reconstruidos (com perdas q < length(L))
Xrec=Q'*Y;

Srec = Xrec(:);
Srec=Srec/max(Srec);
sound(Srec,Fs);
audiowrite('sinal_reconstruido_PCA.wav', Srec,Fs);

