data = randi([0 3],1000,1);
modData = pskmod(data,4,pi/4);

ostbc = comm.OSTBCEncoder('NumTransmitAntennas',4,'SymbolRate',1/2);
txSig = ostbc(modData);

mimochannel = comm.MIMOChannel(...
    'SampleRate',1000, ...
    'PathDelays',[0 2e-3], ...
    'AveragePathGains',[0 -5], ...
    'MaximumDopplerShift',5, ...
    'SpatialCorrelationSpecification','None', ...
    'NumTransmitAntennas',4, ...
    'NumReceiveAntennas',4);

rxSig = mimochannel(txSig);

ts = 1/mimochannel.SampleRate;
t = (0:ts:(size(txSig,1)-1)*ts)';

pwrdB = 20*log10(abs(rxSig(:,1)));

plot(t,pwrdB)
xlabel('Time (s)')
ylabel('Power (dBW)')