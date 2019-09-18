%%  Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
%> @file sig.m
%> @brief Evaluation for Sigmoid function
%>
%> Evaluation for Sigmoid function
%>
%> @author Renzo Andri (andrire)
%%
import fixedPoint
import evalClass

% Configurations
firstPt          = -4;  % reference point (most left)
lastPt           = 3;   % reference point (most right)
stepSize         = 0.5; % reference point (step size)
referencePts_new = firstPt:stepSize:lastPt;
fxPtFormat = fixedPoint(true, 3,12)
%f= @(x) sigmf(x, [1,0]);
f= @tanh;


fxPtFormatExtended = fixedPoint(true, fxPtFormat.qInt,fxPtFormat.qFrac*2)
lib=evalClass(f, fxPtFormat);

numElements = 16;
rangepm = 2;
stepSize = 2*rangepm/numElements;
firstPt = -rangepm-stepSize/2;
lastPt = rangepm+stepSize/2;
referencePts_new = firstPt:stepSize:lastPt;
[a,b, ~,~]=lib.plot(-3,3,1/4096, referencePts_new);


results = []
results_maxerr = []
numElements_list = (4:8)'; %[2 4 8 16 32 64 128]'; %(2:4:128)';%[2 4 8 16 32 64 128]'
rangepm_list = [1:0.5:3]'
for i=1:size(numElements_list,1)
    for j=1:size(rangepm_list,1)
        numElements = numElements_list(i);%4;
        rangepm = rangepm_list(j);
        stepSize = 2*rangepm/numElements;
        firstPt = -rangepm-stepSize/2;
        lastPt = rangepm+stepSize/2;
        referencePts_new = firstPt:stepSize:lastPt;

        [a,b, ~,~]=lib.plot(-3,3,1/4096, referencePts_new);
        results(i, j) = a; %mse
        results_maxerr(i,j) = b;

end
end

[a,b,c,d]=lib.plot(-4,4,1/4096, referencePts_new);
ref_m_rel=ref_m((numElements/2+2):(numElements+1))
ref_q_rel=ref_q((numElements/2+2):(numElements+1))
disp(fxPtFormat.printC(ref_m_rel, 'lut_Tanh_m'));
disp(fxPtFormatExtended.printC((ref_q_rel), 'lut_Tanh_q'));


% MSE
mesh(numElements_list, rangepm_list, log10(results)')
legend_text = {};
ylabel('range')
xlabel('numElements')
zlabel('log_{10}(MSE)')
title("sig MSE for lin. approx")
pause
plot_rangePM_idx = 7;
for k=1:2:size(rangepm_list,1)
legend_text{end+1} = ['range = ' num2str(rangepm_list(k))]
plot(numElements_list, log10(results(:, k))')
hold on
end
title("sig MSE for lin. approx"); % with range="+num2str(rangepm_list(plot_rangePM_idx)))
xlabel('numElements')
ylabel('log_{10}(MSE)')
legend(legend_text)
hold off
grid on;
pause

%max error
% 3d plot
mesh(numElements_list, rangepm_list, log10(results_maxerr)')
ylabel('range')
xlabel('numElements')
zlabel('log_{10}(maxErr)')
title("sig maxErr for lin. approx")

pause

%2d plot
plot_rangePM_idx = 7;
%plot(numElements_list, log10(results_maxerr(:, plot_rangePM_idx))')
for k=1:2:size(rangepm_list,1)
legend_text{end+1} = ['range = ' num2str(rangepm_list(k))]
plot(numElements_list, log10(results_maxerr(:, k))')
hold on
end
title("sig maxerror for lin. approx with range="+num2str(rangepm_list(plot_rangePM_idx)))
xlabel('numElements')
ylabel('log_{10}(maxErr)')
grid on;




