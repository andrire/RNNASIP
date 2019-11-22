%%  Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
%> @file tanh_eval.m
%> @brief Evaluation for tanh function
%>
%> Evaluation for tangent hyperolic function 
%>
%> @author Renzo Andri (andrire)
%%

% This file has been used for evaluation and is therefore not well
% commented, but comments have been added in sig.m Please have a look
% there. :)

import fixedPoint
import evalClass
close all
f= @tanh;
%f= @(x) sigmf(x, [1,0]);
fxPtFormat = fixedPoint(true, 3,12)
fxPtFormatExtended = fixedPoint(true, 3,12*2)
lib=evalClass(f, fxPtFormat);
results = []
results_maxerr = []
numElements_list = (2:4:128)'; %[2 4 8 16 32 64 128]'; %(2:4:128)';%[2 4 8 16 32 64 128]'
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

numElements = 16;
rangepm = 4;
stepSize = 2*rangepm/numElements;
firstPt = -rangepm-stepSize/2;
lastPt = rangepm+stepSize/2;
referencePts_new = firstPt:stepSize:lastPt;

[a,b,ref_m,ref_q]=lib.plot(-4,4,1/4096, referencePts_new);
ref_m_rel=ref_m((numElements/2+2):(numElements+1))
ref_q_rel=ref_q((numElements/2+2):(numElements+1))
disp(fxPtFormat.printC(ref_m_rel, 'lut_Tanh_m'));
disp(fxPtFormatExtended.printC((ref_q_rel), 'lut_Tanh_q'));
title('Linear Approx. for tanh with range = 4, 32 elements and Q3,12')
close all

% MSE
mesh(numElements_list, rangepm_list, log10(results)')
legend_text = {};
ylabel('range')
xlabel('numElements')
zlabel('log_{10}(MSE)')
title("tanh MSE for lin. approx")
plot_rangePM_idx = 7;
for k=1:2:size(rangepm_list,1)
legend_text{end+1} = ['range = ' num2str(rangepm_list(k))]
plot(numElements_list, log10(results(:, k))')
hold on
end
title("tanh MSE for lin. approx"); % with range="+num2str(rangepm_list(plot_rangePM_idx)))
xlabel('numElements')
ylabel('log_{10}(MSE)')
legend(legend_text)
hold off
grid on;

%max error
mesh(numElements_list, rangepm_list, log10(results_maxerr)')
ylabel('range')
xlabel('numElements')
zlabel('log_{10}(maxErr)')
title("tanh maxErr for lin. approx")
plot_rangePM_idx = 7;
plot(numElements_list, log10(results_maxerr(:, plot_rangePM_idx))')
title("tanh MSE for lin. approx with range="+num2str(rangepm_list(plot_rangePM_idx)))
xlabel('numElements')
ylabel('log_{10}(maxErr)')
grid on;

x=2;

% calculate index from float
%floor((x-firstPt+0.5)*2)
%t=(-firstPt+0.5)*2
%x*2+t

% calculate index from FixedPt
x_real = 0;
x=floor(4096*x_real);
t=(-firstPt+0.5)*pow2(fxPtFormat.qFrac+1)
curr_index=bitshift((x*2+t), -fxPtFormat.qFrac)
ref_m(curr_index)*x_real+ref_b(curr_index)
tanh(x_real)


t=(-firstPt+0.5)*pow2(fxPtFormat.qFrac+1);
a=@(x)ref_m(+bitshift((x*2+t), -fxPtFormat.qFrac)).*x./4096+ref_b(+bitshift((x*2+t), -fxPtFormat.qFrac));
plot(-3:0.1:3, a(floor(4096.*(-3:0.1:3))), '*')
hold
plot(-4:0.01:3, tanh(-4:0.01:3), 'g')
hold off


