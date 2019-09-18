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
% Define Interval approximation interval
firstPt = -4;
lastPt = 4;
stepSize = 0.5;

%define fixed-point format and function
fxPtFormat = fixedPoint(true, 3,12)
%f= @(x) sigmf(x, [1,0]);
f= @tanh;

referencePts_new = firstPt:stepSize:lastPt;
%define the format of the result of the multiplication.
fxPtFormatExtended = fixedPoint(true, fxPtFormat.qInt,fxPtFormat.qFrac*2)
lib=evalClass(f, fxPtFormat);

% 1. Calculate and Plot the linear approximation for the selected reference
% points

[a,b,ref_m,ref_q]=lib.plot(-5,5,0.01, referencePts_new)

% 2. Export parameters to C
disp(fxPtFormat.printC(ref_m, 'lut_sig_m'));
disp(fxPtFormatExtended.printC((ref_q), 'lut_sig_q'));

% 3. Calculate the linear approximation for different ranges and different
% number of intervals
results = []
results_maxerr = []
numElements_list = (1:8)'; %[2 4 8 16 32 64 128]'; %(2:4:128)';%[2 4 8 16 32 64 128]'
rangepm_list = [1:0.5:9]'
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

% Part 3a) MSE 3D plot
mesh(numElements_list, rangepm_list, log10(results)')
legend_text = {};
ylabel('range')
xlabel('numElements')
zlabel('log_{10}(MSE)')
title("sig MSE for lin. approx")
pause

% Part 3b) MSE 2D plot
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
% Part 3c) maximum error 3D plot
mesh(numElements_list, rangepm_list, log10(results_maxerr)')
ylabel('range')
xlabel('numElements')
zlabel('log_{10}(maxErr)')
title("sig maxErr for lin. approx")

pause

% Part 3d) maximum error 3D plot
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




