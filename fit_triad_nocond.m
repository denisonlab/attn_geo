function fit_triad_nocond
%% fitting triad models forward (unconstrained -> constrained)

load('data/triad_data_collated.mat','beh');

% set up parallel processing
ncores = str2num(getenv('NSLOTS'));
maxNumCompThreads(ncores);

% check if parpool is running
pool = gcp('nocreate');
if isempty(pool)
	myCluster = parcluster('local');
	myCluster.JobStorageLocation = getenv('TMPDIR');
	pool = parpool(myCluster,ncores); % buffer
end

fit_options = optimoptions('fminunc',...
    'MaxFunctionEvaluations',2e4,...
    'MaxIterations',2e4,...
    'FunctionTolerance',1e-10,...
    'StepTolerance',1e-10,...
    'OptimalityTolerance',1e-10,...
    'Display','none',...
    'HessianApproximation',{"lbfgs",20},...
    'SpecifyObjectiveGradient',true,...
    'UseParallel',false);

%% concatenate full data
n = length(beh);
beh_all.attnCue = [beh.attnCue];
beh_all.triadChoice = vertcat(beh.triadChoice);
beh_all.triadRef = [beh.triadRef];
beh_all.triadChosen = [beh.triadChosen];
beh_all.triadChosenLoc = [beh.triadChosenLoc];
beh_all.attnCue = [beh.attnCue];

% triad stimuli are coded with different values depending on attention condition
% so that they can be fitted with difference model coordinates
beh_all.triadRefA = beh_all.triadRef + (beh_all.attnCue-1).*36;
beh_all.triadChoiceA = beh_all.triadChoice + (beh_all.attnCue'-1).*36;

beh_all.sid = [];
for ii=1:n
    beh(ii).triadRefA = beh(ii).triadRef + (beh(ii).attnCue-1).*36;
    beh(ii).triadChoiceA = beh(ii).triadChoice + (beh(ii).attnCue'-1).*36;
    beh_all.sid = [beh_all.sid ii.*ones(1,length(beh(ii).attnCue))];
end

%% fitting
% get params loops
rng(1234,'twister'); % for replicability
m = str2num(getenv('SGE_TASK_ID')); % dimensionality of fit

cv_cond = 100.*beh_all.attnCue + beh_all.sid;
k = 10; % how many folds
c1b = cvpartition(cv_cond,"KFold",k);
fit_cond = false; % no separate condition coordinates
lam_S = [0 10.^linspace(-1.5,3,20) Inf]; % subject ridge
lam_Fs = [0 10.^linspace(-1.5,2.5,20)]; % fusion ridge

%% Step 1: fit fully unconstrained models
fprintf('--- Fitting fully unconstrained models ---\n');

cv1_vec = combvec(m,1:k,0,0,0);
cv1_fit_coords = nan(36,m,1,n,length(cv1_vec));

% set up parfor loop tracker
q = parallel.pool.DataQueue;
afterEach(q,@parforTracker);
parforTracker(size(cv1_vec,2),m);

parOpts = parforOptions(pool,"RangePartitionMethod","fixed","SubrangeSize",1);
parfor (r1=1:size(cv1_vec,2),parOpts)
    this_m = cv1_vec(1,r1);
    test_set = cv1_vec(2,r1);
    this_lamS = cv1_vec(3,r1); % subject ridge
    this_lamFs = cv1_vec(4,r1); % fusion ridge

    % subset training and testing data
    train_idx = c1b.training(test_set);
    beh_train = struct(); beh_test = struct();
    trainTrialsPerSub = nan(1,n); testTrialsPerSub = nan(1,n);
    for ii=1:n
        sub_train = train_idx' & beh_all.sid==ii;
        beh_train(ii).triadRefA = beh_all.triadRefA(sub_train);
        beh_train(ii).triadChoiceA = beh_all.triadChoiceA(sub_train,:);
        beh_train(ii).triadChosenLoc = beh_all.triadChosenLoc(sub_train);
        trainTrialsPerSub(ii) = length(beh_train(ii).triadRefA);
    end

    % get starting coordinates
    seedMat = zeros(36,36);
    combStim = combvec(1:36,1:36);
    for ii=1:length(combStim)
    	seedMat(ii) = 1-sum(ismember(beh_all.triadRef(train_idx),combStim(:,ii)) & ismember(beh_all.triadChosen(train_idx),combStim(:,ii))) ./ ...
            sum(ismember(beh_all.triadRef(train_idx),combStim(:,ii)) & any(ismember(beh_all.triadChoice(train_idx,:),combStim(:,ii))'));
    end
    seedMat(isnan(seedMat)) = 0;

    % MDS on RDM
    init_coords = mdscale(seedMat,m,'Start','random','Replicates',10,'Options',statset('MaxIter',1000));

    % fit model to training data
    % m1 = averaged across participants
    % m2 = separate participant spaces
    m1_negLL = @(p) pred_triad_resp(p,beh_train,Inf,0,fit_cond,true);
    m1_fit = fminunc(m1_negLL,init_coords,fit_options);

    m2_negLL = @(p) pred_triad_resp(p,beh_train,0,0,fit_cond,true);
    m2_fit = fminunc(m2_negLL,repmat(init_coords,[1 1 1 n]),fit_options);

    fit_coords = 0.5 .* m1_fit + 0.5 .* m2_fit; % blend fits

    cv1_fit_coords(:,:,:,:,r1) = fit_coords;

    send(q,[]);
end

cv_nll1.fit_coords = cv1_fit_coords;

%% Step 2: fit constrained models
fprintf('--- Starting constrained modeling ---\n');

cv2_vec = combvec(m,1:k,lam_S,lam_Fs);
cv2_nll_train = nan(1,length(cv2_vec));
cv2_train_exit = nan(1,length(cv2_vec));
cv2_train_iter = nan(1,length(cv2_vec));
cv2_train_func = nan(1,length(cv2_vec));
cv2_train_opti = nan(1,length(cv2_vec));
cv2_nll_test = nan(1,length(cv2_vec));
cv2_nll_test_sub = nan(length(cv2_vec),length(beh));
cv2_acc_test_sub = nan(length(cv2_vec),length(beh));
cv2_fit_coords = nan(36,m,1,n,length(cv2_vec));

% set up parfor loop tracker
q = parallel.pool.DataQueue;
afterEach(q,@parforTracker);
parforTracker(size(cv2_vec,2),m);

parOpts = parforOptions(pool,"RangePartitionMethod","fixed","SubrangeSize",1);
parfor (r2=1:size(cv2_vec,2),parOpts)
    this_m = cv2_vec(1,r2);
    test_set = cv2_vec(2,r2);
    this_lamS = cv2_vec(3,r2); % subject ridge
    this_lamFs = cv2_vec(4,r2); % fusion ridge

    % get starting coordinates
    if isinf(this_lamS)
        init_coords = mean(cv1_fit_coords(:,:,:,:,test_set),4);
        test_lamS = Inf;
    else
        init_coords = cv1_fit_coords(:,:,:,:,test_set);
        test_lamS = 0;
    end

    % subset training and testing data
    train_idx = c1b.training(test_set);
    beh_train = struct(); beh_test = struct();
    trainTrialsPerSub = nan(1,n); testTrialsPerSub = nan(1,n);
    for ii=1:n
        sub_train = train_idx' & beh_all.sid==ii;
        beh_train(ii).triadRefA = beh_all.triadRefA(sub_train);
        beh_train(ii).triadChoiceA = beh_all.triadChoiceA(sub_train,:);
        beh_train(ii).triadChosenLoc = beh_all.triadChosenLoc(sub_train);
        trainTrialsPerSub(ii) = length(beh_train(ii).triadRefA);

        sub_test = ~train_idx' & beh_all.sid==ii;
        beh_test(ii).triadRefA = beh_all.triadRefA(sub_test);
        beh_test(ii).triadChoiceA = beh_all.triadChoiceA(sub_test,:);
        beh_test(ii).triadChosenLoc = beh_all.triadChosenLoc(sub_test);
        testTrialsPerSub(ii) = length(beh_test(ii).triadRefA);
    end

    % fit model to training data
    m1b_negLL = @(p) pred_triad_resp(p,beh_train,this_lamS,this_lamFs,fit_cond,true);
    [fit_coords,fval,exitflag,output] = fminunc(m1b_negLL,init_coords,fit_options);

    % test model on held-out data
    [~,~,out_test] = pred_triad_resp(fit_coords,beh_test,test_lamS,0,fit_cond,false);
    % if size(fit_coords,3)==1, fit_coords = repmat(fit_coords,[1 1 3 1]); end
    if size(fit_coords,4)==1, fit_coords = repmat(fit_coords,[1 1 1 n]); end

    cv2_nll_train(r2) = fval;
    cv2_train_exit(r2) = exitflag;
    cv2_train_iter(r2) = output.iterations;
    cv2_train_func(r2) = output.funcCount;
    cv2_train_opti(r2) = output.firstorderopt;
    cv2_nll_test(r2) = out_test.nll;
    cv2_nll_test_sub(r2,:) = out_test.nll_sub;
    cv2_acc_test_sub(r2,:) = out_test.acc_sub;
    %cv2_fit_coords(:,:,:,:,r2) = fit_coords;

    send(q,[]);
end

cv_nll2.train_nll = reshape(cv2_nll_train,[],length(lam_S),length(lam_Fs));
cv_nll2.train_exit = reshape(cv2_train_exit,[],length(lam_S),length(lam_Fs));
cv_nll2.train_iter = reshape(cv2_train_iter,[],length(lam_S),length(lam_Fs));
cv_nll2.train_func = reshape(cv2_train_func,[],length(lam_S),length(lam_Fs));
cv_nll2.train_opti = reshape(cv2_train_opti,[],length(lam_S),length(lam_Fs));
cv_nll2.test_nll = reshape(cv2_nll_test,[],length(lam_S),length(lam_Fs));
cv_nll2.test_nll_sub = reshape(cv2_nll_test_sub,[],length(lam_S),length(lam_Fs),n);
cv_nll2.test_acc_sub = reshape(cv2_acc_test_sub,[],length(lam_S),length(lam_Fs),n);
%cv_nll2.fit_coords = reshape(cv2_fit_coords,36,m,3,n,[],length(lam_S),length(lam_Fs));

save(sprintf('data/fit_nocond/nocond_ridge_nll_%dd_cv%d.mat',m,k),'cv_nll1','cv_nll2');
