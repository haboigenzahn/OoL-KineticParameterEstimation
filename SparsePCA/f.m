function Wsparse=f(X)
num_comp=size(X); num_comp=num_comp(2);
Wsp=sparsePCA(X,1,num_comp,0,1)
Wsparse=Wsp;