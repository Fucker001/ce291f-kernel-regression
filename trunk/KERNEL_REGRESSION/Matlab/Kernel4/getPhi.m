function phi = getPhi(u,fatKT,OTT)

l2=OTT-fatKT*u;
phi=sum(abs(u))+sqrt(l2'*l2);

end