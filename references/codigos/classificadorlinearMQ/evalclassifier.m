function Nacertos=evalclassifier(Ytst,Ypred,Ntst)

Nacertos=0;
for k=1:Ntst,
        %Paciente=k,
        %[Ypred(:,k) Ytst(:,k)],
        %pause(5);
        
        [dummy Imax_pred]=max(Ypred(:,k));
        [dummy Imax_real]=max(Ytst(:,k));
     
        if Imax_pred == Imax_real,
          Nacertos=Nacertos+1;
        end
end
