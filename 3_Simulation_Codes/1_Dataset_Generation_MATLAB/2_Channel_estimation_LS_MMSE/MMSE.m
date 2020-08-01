function H_MMSE = MMSE(RxP,TxP,N,pilotFrequency,h_CIR,SNR)
% I modified the MMSE_CE function provided in :MIMO-OFDM Wireless 
% Communications with MATLAB¢ç   Yong Soo Cho, Jaekwon Kim, 
% Won Young Yang and Chung G. Kang
%2010 John Wiley & Sons (Asia) Pte Ltd
% The modification made the function more suitable with my OFDM Matlab code
noiseVar = 10^(SNR*0.1);
Np=N/pilotFrequency; % Number of Pilots
H_LS = RxP./TxP;  % LS estimate
k=0:length(h_CIR)-1; 
hh = h_CIR*h_CIR'; 
% tmp = h_CIR.*conj(h_CIR).*k;
r = sum(h_CIR.*conj(h_CIR).*k)/hh;  
r2 = (h_CIR.*conj(h_CIR).*k)*k.'/hh;
t_rms = sqrt(r2-r^2);     % rms delay
D = 1j*2*pi*t_rms/N; % Denomerator of Eq. (6.16) page 192
K1 = repmat([0:N-1].',1,Np);
K2 = repmat([0:Np-1],N,1);
rf = 1./(1+D*(K1-K2*pilotFrequency));
K3 = repmat([0:Np-1].',1,Np);
K4 = repmat([0:Np-1],Np,1);
rf2 = 1./(1+D*pilotFrequency*(K3-K4));
Rhp = rf;
Rpp = rf2 + eye(length(H_LS),length(H_LS))/noiseVar;
H_MMSE = transpose(Rhp*inv(Rpp)*H_LS);  % MMSE channel estimate
