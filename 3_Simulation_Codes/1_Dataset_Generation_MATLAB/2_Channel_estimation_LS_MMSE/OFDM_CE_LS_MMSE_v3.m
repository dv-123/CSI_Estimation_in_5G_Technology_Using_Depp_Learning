%% OFDM system with channel estimation:
clear all; close all; clc;
global W
%% define parameters 
m=input( ' how many OFDM symbols to be simulated m = ' ) ;
N=input( ' length of OFDM symbols N = ' ) ;
M=input( ' Constellation Order M = ' ) ;
pilotFrequency=input('Pilot Frequency = ' ) ;
E=input('Pilot Energy = ');
Ncp=input( ' Cyclic Prefix length = ' ) ;
L=input(' channel Length (number of taps) = ');
typ=input('Constellation Family Type is [1 for M-QAM, 2 for M-PSK]:- ');
%% ----------------------------------------------------------------------
%% define the modems Tx/Rx
switch typ
    case 1
        Tx=modem.qammod('M',M);
        Rx=modem.qamdemod ('M',M);
    case 2
        Tx=modem.pskmod('M',M);
        Rx=modem.pskdemod ('M',M);
    otherwise
        error('Error, Constellation Family not Defined');
end
%% data generation 
D=randi ([0 M-1],m,N);
%% mapping (baseband modulation )
D_Mod=modulate(Tx,D);
%% serial to parallel 
D_Mod_serial=D_Mod.';
%% specify Pilot & Date Locations
PLoc = 1:pilotFrequency:N; % location of pilots
DLoc = setxor(1:N,PLoc); % location of data
%% Pilot Insertion
D_Mod_serial(PLoc,:)=E*D_Mod_serial(PLoc,:);
figure;
imagesc(abs(D_Mod_serial ))
%% inverse discret Fourier transform (IFFT)
%  Amplitude Modulation
d_ifft=ifft(D_Mod_serial);
%% parallel to serail 
d_ifft_parallel=d_ifft.';
%% Adding Cyclic Prefix
CP_part=d_ifft_parallel(:,end-Ncp+1:end); % this is the Cyclic Prefix part to be appended.
ofdm_cp=[CP_part d_ifft_parallel];
%% generating random channel 
h= randn(1,L) + 1j * randn(1,L);
h = h./norm(h); % normalization 
H = fft(h,N); % Frequency-Domain Channel 
d_channelled = filter(h,1,ofdm_cp.').'; % channel effect
channel_length = length(h); % True channel and its time-domain length
H_power_dB = 10*log10(abs(H.*conj(H))); % True channel power in dB
%% add noise 
count=0;
snr_vector=0:4:40;
for snr=snr_vector
    SNR = snr + 10*log10(log2(M));
    count=count+1 ;
    disp(['step: ',num2str(count),' of: ',num2str(length(snr_vector))])
    ofdm_noisy_NoCH=awgn(ofdm_cp,SNR,'measured' ) ;
    ofdm_noisy_with_chann=awgn(d_channelled,SNR,'measured' ) ;
    %% receiver
    %Remove Cyclic Prefix
    ofdm_cp_removed_NoCH=ofdm_noisy_NoCH(:,Ncp+1:N+Ncp);
    ofdm_cp_removed_with_chann=ofdm_noisy_with_chann(:,Ncp+1:N+Ncp);
    % serial to parallel 
    ofdm_parallel_NoCH=ofdm_cp_removed_NoCH.';
    ofdm_parallel_chann=ofdm_cp_removed_with_chann.';
    %% Discret Fourier transform (FFT)
    %  Amplitude Demodulation
    d_parallel_fft_NoCH=fft(ofdm_parallel_NoCH) ;
    d_parallel_fft_channel=fft(ofdm_parallel_chann) ;


    %% channel estimation
    % Extracting received pilots
    TxP = D_Mod_serial(PLoc,:); % trnasmitted pilots
    RxP = d_parallel_fft_channel(PLoc,:); % received pilots
    % Least-Square Estimation
    Hpilot_LS= RxP./TxP; % LS channel estimation
    % MMSE Estimation:- 
    for r=1:m
        H_MMSE(:,r) = MMSE(RxP(:,r),TxP(:,r),N,pilotFrequency,h,SNR);
    end

    % Interpolation p--->N    
    for q=1:m
    HData_LS(:,q) = interpolate(Hpilot_LS(:,q).',PLoc,N,'spline'); % Linear/Spline interpolation
    end
    %% parallel to serial   
    HData_LS_parallel1=HData_LS.';
    HData_MMSE_parallel1=H_MMSE.';


    %% demapping 
    d_received_NoCH=demodulate(Rx,(d_parallel_fft_NoCH.')) ; % No Channel
    d_received_chann_LS=demodulate(Rx,(d_parallel_fft_channel.')./HData_LS_parallel1) ; % LS channel estimation
    d_received_chann_MMSE=demodulate(Rx,(d_parallel_fft_channel.')./(HData_MMSE_parallel1)) ; % MMSE channel estimation
    %% Removing Pilots from received data and original data 
    D_no_pilots=D(:,DLoc); % removing pilots from D
    Rec_d_NoCH=d_received_NoCH(:,DLoc); % removing pilots from d_received_NoCH
    Rec_d_LS=d_received_chann_LS(:,DLoc); % removing pilots from d_received_chann_LS
    Rec_d_MMSE=d_received_chann_MMSE(:,DLoc); % removing pilots from d_received_chann_MMSE
    %% Calculating BER
    [~,r_NoCH(count)]=symerr(D_no_pilots,Rec_d_NoCH) ;
    [~,r_LS(count)]=symerr(D_no_pilots,Rec_d_LS) ;
    [~,r_MMSE(count)]=symerr(D_no_pilots,Rec_d_MMSE) ;
end
figure;
semilogy(snr_vector,r_NoCH,'-+');hold on
semilogy(snr_vector,r_LS,'-o');
semilogy(snr_vector,r_MMSE,'-s');
legend('orig. No Channel','LS CE','MMSE CE');
grid ;
hold off;
H_power_esti_dB_LS     = 10*log10(abs(HData_LS_parallel1.*conj(HData_LS_parallel1))); % Estimated channel power in dB
H_power_esti_dB_MMSE     = 10*log10(abs(HData_MMSE_parallel1.*conj(HData_MMSE_parallel1))); % Estimated channel power in dB
figure;hold on;
plot(H_power_dB(1:8:end),'+k','LineWidth',3);
plot(H_power_esti_dB_LS(1,(1:8:end)),'or','LineWidth',3);
plot(H_power_esti_dB_MMSE(1,(1:8:end)),'Sb','LineWidth',1);
 title('ACTUAL AND ESTIMATED CHANNELS');
    xlabel('Time in samples');
    ylabel('Magnitude of coefficients');
    legend('Actual','estimated LSE','estimated MMSE')
