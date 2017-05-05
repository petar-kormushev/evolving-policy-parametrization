function Evolving_Policy_Parametrization
%
% Author: Petar Kormushev
%
close all;


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 200;

% Target unknown goal function to be appoximated using spline tuned by RL
target.ax = 0:0.01:1;
target.ay = 0.5 + 0.2*sin(10*target.ax) + 0.07*sin(20*target.ax) + 0.04*sin(30*target.ax) + 0.04*sin(50*target.ax);


for i=1:4
    [Return, s_Return, param, rl] = RL_PoWER(26, 26, target, nbData, 0); % relearn = 0
    ReturnWithFixedKnots(:,i) = Return;
end
%close all;
for i=1:4
    [Return, s_Return, param, rl] = RL_PoWER(4, 26, target, nbData, 0); % relearn = 0
    ReturnWithEvolvingKnots(:,i) = Return;
end
%close all;

hFigAvgResults = figure('Name', 'Return over rollouts', 'position',[1350,250,600,500]); axis on; grid on; hold on;
plotAvgResults(hFigAvgResults, ReturnWithFixedKnots, ReturnWithEvolvingKnots);

end

%% RL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Return, s_Return, param, rl] = RL_PoWER(fromKnots, toKnots, target, nbData, relearn, Return, s_Return, param, rl);
disp('Starting RL...');
%  rl_m = m; % make a copy of m

hFig = figure('Name', 'Rollouts', 'position', [100, 400, 800, 600]);
hFigResults = figure('Name', 'Results', 'position',[1000,600,800,400]); axis on; grid on; hold on;
plotRollouts(hFig, target, 0);

% START of PoWER algorithm for PSRL - Policy Search RL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of parameters of the policy = number of spline knots here
n_rfs = fromKnots;  % equal to the size of nbVar*sizeof(b)
% number of iterations
n_iter = 100;
importanceSamplingTopCount = 3;  % how many top rollouts to use for importance sampling
nbTrials = n_iter + 1; % for RL

if relearn==0
  Return = zeros(1,n_iter+1);
  s_Return = zeros(n_iter+1,2);

  % initialize parameters
  param = zeros(n_rfs,n_iter+1);
end

% use adaptive variance or not
adaptive_variance = 0;

if adaptive_variance==1
    % set the initial variance
    variance(1:3,1) = 0.05.*ones(3,1); % velocity vel0
else
    % set fixed variance
    variance(1:n_rfs) = 0.01.*ones(n_rfs,1); % spline knots
    disp('Variance for a single nbState is:');
    variance'
end
varianceLog = []; % just to declare the variable

% initial spline
policy(1).n = n_rfs; % number of spline knots
policy(1).x = 0:1/(policy(1).n-1):1; % x positions of knots 
policy(1).y = 0.5*ones(1,policy(1).n); % set initial policy y values to the middle 0.5
policy(1).pp = spline(policy(1).x, policy(1).y); % calculate the initial spline
plotSpline(policy(1).pp, policy(1).n, policy(1).x, policy(1).y, 'green', 3); % visualize the spline

% initialize the parameters to optimize with their current values
param(:,1) = policy(1).y';

disp('param - in the BEGINNING');
param(:,1)'

% pause

disp('Running PoWER algorithm...');
% do the iterations
for iter=1:n_iter
    if (mod(iter,100)==0)
        disp(['Iter ', num2str(iter)]);
    end

    if fromKnots < toKnots
        % regularly increase the number of knots
        increaseInterval = round(n_iter / (toKnots - fromKnots));
        if (mod(iter,increaseInterval)==0)
            n2 = policy(iter).n + 1;
%             disp(['Increasing the # of knots to ', num2str(n2)]);
            n_rfs = n2;
            param = zeros(n_rfs,n_iter+1);
            variance(1:n_rfs) = 1.0 * variance(1:1) * ones(n_rfs,1); % copy prev. variance
            for i=1:iter
                [policy(i).pp, policy(i).x, policy(i).y] = getNewSpline(policy(i).pp, n2);
                policy(i).n = n2;
                param(:,i) = policy(i).y';
            end
        end
    end
    
    % reproduce EE trajectory
    rl(iter).Data = reproduction(policy(iter), target);
    % calculate the return of the current rollout
 	Return(iter) = ReturnOfRollout(rl(iter).Data, target);

 	disp(['Current rollout return: ', num2str(Return(iter))]);
     
  
    % this lookup table will be used for the importance sampling
    s_Return(1,:) = [Return(iter) iter];
    s_Return = sortrows(s_Return);
    
    % update the policy parameters
    param_nom = zeros(n_rfs,1);
    if adaptive_variance==1
        param_dnom = zeros(n_rfs,1);
    else
        param_dnom = 0;
    end
    
    if relearn==0
        min_count = iter; % only the rollouts from current batch so far
    else
        min_count = n_iter; % all previous experiences
    end
    % calculate the expectations (the normalization is taken care of by the division)
    % as importance sampling we take the 10 best rollouts
    for i=1:min(min_count,importanceSamplingTopCount) % TODO: how many are optimal??
        % get the rollout number for the 10 best rollouts
        j = s_Return(end+1-i,2);

        % calculate the exploration with respect to the current parameters
        temp_explore = (param(:,j)-param(:,iter));
        
        if adaptive_variance==1
            % calculate W
            temp_W = variance(:,j).^-1;
            % we assume that always only one basis functions is active we get 
            % these simple sums
            param_nom = param_nom + temp_W.*temp_explore*Return(j);
            param_dnom = param_dnom + temp_W.*Return(j);
        else
            % always have the same exploration variance,
            % and assume that always only one basis functions is active we get 
            % these simple sums
            param_nom = param_nom + temp_explore*Return(j);
            param_dnom = param_dnom + Return(j);
        end
    end
    
    % update the parameters
    param(:,iter+1) = param(:,iter) + param_nom./(param_dnom+1.e-10);
    
    if adaptive_variance==1
        % update the variances
        var_nom = zeros(n_rfs,1);
        var_dnom = 0;
        if iter>1
            % we use more rollouts for the variance calculation to avoid 
            % rapid convergence to 0
            for i=1:min(iter,100)
                % get the rollout number for the 100 best rollouts
                j = s_Return(end+1-i,2);
                % calculate the exploration with respect to the current parameters
                temp_explore = (param(:,j)-param(:,iter));
                var_nom = var_nom + temp_explore.^2.*Return(j);
                var_dnom = var_dnom + Return(j);
            end
            % apply and an upper and a lower limit to the exploration
            variance(:,iter+1) = max(var_nom./(var_dnom+1.e-10),.1.*variance(:,1));
            variance(:,iter+1) = min(variance(:,iter+1),10.*variance(:,1));
        end
    end  
    
    % decay the variance
    variance(:) = variance(:).*0.965; % velocity vel0
    varianceLog(1,iter) = variance(1); % TODO: fix size and uncomment line
    
    % in the last rollout we use the params which produced the best rollout
    if iter==n_iter
        best_j = s_Return(end,2); % takes the rollout with the max Return from the saved ones
        param(:,iter+1) = param(:, best_j);
    else
        if adaptive_variance==1
            param(:,iter+1) = param(:,iter+1) + variance(:,iter+1).^.5.*randn(n_rfs,1);
        else
            param(:,iter+1) = param(:,iter+1) + variance(:).^.5.*randn(n_rfs,1);
        end
    end
       
    % apply the new parameters from RL
    policy(iter+1) = policy(iter); % copy all fields from old policy
    policy(iter+1).y = param(:, iter+1)';
    policy(iter+1).pp = spline(policy(iter+1).x, policy(iter+1).y); % re-calculate the spline 
end

disp('param - after the RL optimization');
param(:,end)'

disp('Initial return, before optimization with RL');
ReturnOfRollout(rl(1).Data, target)

% reproduce EE trajectory
rl(iter+1).Data = reproduction(policy(iter+1), target);

% calculate the return of the last rollout
Return(iter+1) = ReturnOfRollout(rl(iter+1).Data, target);

disp('Final reward:');
Return(iter+1)

% plot the rollout / all rollouts so far
plotResults(hFigResults, iter+1, adaptive_variance, variance, varianceLog(:,1:iter+1-1), Return(:,1:iter+1));
plotRollouts(hFig, target, iter, policy);

if adaptive_variance==1
    figure('Name', 'Variance over rollouts'); hold on;
    for i=1:n_rfs
    %    plot(variance(:,i), 'color', [rand(1) rand(1) rand(1)]);
        plot(variance(:,i), 'color', [0 0 0]);
    end
    ylabel('variance');
    xlabel('rollouts');
end
 
disp(['Final Return ', num2str(Return(end))]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end



%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotResults(hFig, n_rfs, adaptive_variance, variance, varianceLog, Return)
figure(hFig);

subplot(1,2,1);
for i=1:size(varianceLog,1)
    plot(varianceLog(i,:), 'color', [0 0 0]);
end
ylabel('variance');
xlabel('rollouts');
 
subplot(1,2,2);
plot(Return);
axis([0 size(Return,2)-1 0 1]);
ylabel('return');
xlabel('rollouts');

end


%% Do reproduction of trajectory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Data = reproduction(policy, target);

Data.policy = policy;
    
%save('data/data_01.mat','Data');
end % function
%load('data/data_01.mat');

%% get a new spline pp2, for a new number of spline knots n2,
% replacing the old spline pp,
function [pp2, x2, y2] = getNewSpline(pp, n2)
  x2 = 0:1/(n2-1):1; % x positions of knots 
  y2 = ppval(pp, x2); % get the value of the old spline pp for the new knots x2
  pp2 = spline(x2, y2);
end

%% plots a spline pp
function plotSpline(pp, n, x, y, clr, wdt)
%return;
  plot(x,y, 'o', 'color', clr, 'linewidth', wdt);

  % visualize the spline
  sx = 0:0.01:1; % controls the smoothness of visualization of the spline
  sy = ppval(pp, sx);
  plot(sx, sy, 'color', clr, 'linewidth', wdt); % Problem!! The spline might go
% outside [0,1] interval!!! maybe better to use DMP/attractors?? Or Gaussians?

end

%% plot all rollouts so far
function plotRollouts(hFig, target, n_iter, policy)
%return;
    figure(hFig);
    clf; % clears the figure
    hold on;
    axis([0 1 0 1]);

%    clrmap = min(colormap('Gray')+0.4,1); % 'green'
%    colormap(clrmap);
%    colormap('Summer');
    
	for iter=1:n_iter
%        clr = 'green';
        clr = (0.5 + 0.5*iter/n_iter) * [0 1 0]; % gradient green
        if iter==n_iter % only for the last
            clr = [1 0 0]; % red
        end
        plotSpline(policy(iter).pp, policy(iter).n, policy(iter).x, policy(iter).y, clr, 2); % visualize the spline
    end
    
    plot(target.ax, target.ay, 'color', 'black', 'linewidth', 3);
   
end


%% averaged results
function plotAvgResults(hFig, ReturnWithFixedKnots, ReturnWithEvolvingKnots)

figure(hFig);
clf; hold on; box on; % grid on; % axis equal; % clear the figure to redraw everything again

xlabel('Number of rollouts','fontsize',18);
ylabel('Return','fontsize',18);
set(gca,'FontSize',14);

if max(size(ReturnWithEvolvingKnots)) > 0
    % select just a subset of the samples, in order to reduce the number of errorbars
    x=(1:3:size(ReturnWithEvolvingKnots, 1))';
    meanVals = mean(ReturnWithEvolvingKnots(x,:), 2); % avg across 2nd dimension, i.e. sessions
    errors = std(ReturnWithEvolvingKnots(x,:),1,2);  % TODO: or 1* ????????????????????????????????????????????????????????????????????????
    errorbar(x,meanVals,errors,'-b');
    p(1)=plot(x,meanVals,'linewidth',2.0,'color',[0 0 1]);
end

if max(size(ReturnWithFixedKnots)) > 0
    % select just a subset of the samples, in order to reduce the number of errorbars
    x=(1:3:size(ReturnWithFixedKnots, 1))'; % vector row
    meanVals = mean(ReturnWithFixedKnots(x,:), 2); % avg across 2nd dimension, i.e. sessions
    errors = std(ReturnWithFixedKnots(x,:),1,2);  % TODO: or 1* ????????????????????????????????????????????????????????????????????????
    errorbar(x,meanVals,errors,'-r');
    p(2)=plot(x,meanVals,'linewidth',2.0,'color',[1 0 0]);
end

axis([0 size(ReturnWithFixedKnots,1)-1 0 1]);
% add only desired legend entries
h = legend([p(1) p(2)],'Evolving knots', 'Fixed knots');
set(h,'Location','NorthWest') ;

end
