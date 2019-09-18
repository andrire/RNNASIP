%%  Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
%> @file evalClass.m
%> @brief Helper Class to evaluate tanh, sigmoid or other function for piecewise linear approximation.
%>
%> Helper Class for Evaluation of piecewise-linear approximation
%>
%> @author Renzo Andri (andrire)
%%
classdef evalClass
   properties
      func
      q_format
   end
   methods
      % Constructor
      function obj = evalClass(func, q_format)
               obj.func = func;
               obj.q_format = q_format;
      end
      % Helper function to get a string with the ranges
      function str_out=range2str(obj, x_range)
          range_size = size(x_range, 2);
          if range_size < 7
              str_out = num2str(x_range(1));
              for ids = 2:range_size
                  str_out = [str_out ', ' num2str(x_range(ids))];
              end
          else 
              str_out = num2str(x_range(1));
              for ids = 2:3
                  str_out = [str_out ', ' num2str(x_range(ids))];
              end
              str_out = [str_out ', ...'];
              for ids = range_size-2:range_size
                  str_out = [str_out ', ' num2str(x_range(ids))];
              end
          end
          
      end
      
      % plots the function with the range [lim1, lim2] with prec steps and
      % reference Points in refPts and output the mean square error, the
      % max error and the parameters of the linear approximation where as
      % f(x)=ref_m[i]*x+ref_b[i] for interval i.
      function [mse, maxErr, ref_m, ref_b] = plot(obj, lim1, lim2, prec, refPts)
         %configure 
         DELTA = 0.0000001;
         referencePts_new = refPts; %-4:0.5:3;    
         
         ref_range = [lim1 0.5*(referencePts_new(2:end)+referencePts_new(1:end-1)) lim2];
             
         for i=1:size(referencePts_new,2)
             eval_range = ref_range(i):prec:ref_range(i+1);
            [r,m,b]=regression(eval_range, obj.func(eval_range));
            ref_m(i) = m;
            ref_b(i) = b;
         end
         
         
         plot(lim1:prec:lim2, obj.func(lim1:prec:lim2))
         hold on
         
         title(['Linear Approx. for ', func2str(obj.func), ' for ref. points \{',obj.range2str(referencePts_new), '\} and ', obj.q_format.tostring()])
         %plot(obj.q_format.discretize(lim1:prec:lim2), obj.q_format.discretize(obj.func(lim1:prec:lim2)))

         x_range = obj.q_format.discretize(lim1:prec:lim2);
         

         [~,i]=min(abs(bsxfun(@minus,x_range,referencePts_new')));
         referencePts = referencePts_new(i);
         
         grad = ref_m(i);
         bias = ref_b(i);

         plot(x_range, obj.q_format.discretize(bias)+obj.q_format.discretize(x_range).*obj.q_format.discretize(grad), '.')
         error_array = (obj.q_format.discretize(bias)+obj.q_format.discretize(x_range).*obj.q_format.discretize(grad))-obj.func(x_range);
         errorMSQ = mean(((obj.q_format.discretize(bias)+obj.q_format.discretize(x_range).*obj.q_format.discretize(grad))-obj.func(x_range)).^2);
         
         plot(x_range, 10*error_array, '.')
         plot(x_range, referencePts, '.')
         axPos = get(gca, 'Position');
         xMinMax = xlim;
         yMinMax = ylim;
         xAnnotation = axPos(1) + ((-2.5 - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
         yAnnotation = axPos(2) + ((1.5 - yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
         %annotation('textbox',...
   % [xAnnotation yAnnotation 0.1 0.1],...
   % 'String',{['MSE =' num2str(errorMSQ)], ['maxError = ' num2str(max(abs(error_array)))]},...
   % 'FontSize',14,...
   % 'FontName','Arial',...
   % 'LineWidth',1,...
   % 'Color',[0.84 0.16 0]);
         grid()
         errorMSQ
         legend('Original function', 'Approximation', '10\times l1-error', 'Reference Point')
         hold off
         mse=errorMSQ;
         maxErr=max(abs(error_array))
      end
      
   end
end
