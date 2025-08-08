% Compressing an image with PCA
%
% Last modification: 04/08/2024
% Author: Guilherme Barreto

clear; clc; close all
pkg load image
pkg load statistics

% Le a imagem (imread) e converte (im2double) para double precision
A=imread('lena512.jpg');
%A=imread('yale1.happy');
Ar=imresize(A,[512 512]);
An=Ar; % An = imnoise(Ar,'gaussian');
B=im2double(An);
X=im2col(B,[16 16],'distinct'); % Matriz de dados original

%%%%%%%% PCA %%%%%%%%%%%
Cx=cov(X');  % matriz de covariancia
##[V L]=eig(Cx);
##L=diag(L);  % Vetor de autovalores
##[L I]=sort(L,'descend');  % Autovalores em ordem decrescente
##V=V(:,I);  % Autovetores ordenados do maior para menor autovalor
##
##%figure; bar(L);  % grafico da amplitude dos autovalores
##SL=sum(L);  % Soma dos autovalores
##aux=0;
##for i=1:length(L),
##    aux=aux+L(i);
##    VE(i)=aux/SL;   % Variancia explicada
##end

[V L VEi]=pcacov(Cx);

VE=cumsum(VEi)/sum(VEi);

VE100=100*VE;
figure; plot(VE100,'b-','linewidth',5); grid
set(gca, "fontsize", 16)

% escolha dos Q maiores autovalores (componentes principais)
tol=0.95;  % tolerancia para VE aceitavel
q=length(find(VE<=tol));

% Matriz de transformacao resultante
Vq=V(:,1:q);

Q=Vq';

% Imagem transformada com blocos de tamanho K
Z=Q*X;

% Z = Z + 0.1*randn(size(Z));  % Adiciona ruido gaussiano

% Imagem reconstruida (com perdas q < length(L))
Xrec=Q'*Z;

Ar = col2im(Xrec,[16 16],[512 512],'distinct');

Erec=norm(B-Ar)

figure; imshow(An)
figure; imshow(Ar)

