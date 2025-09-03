function fit_triad_alt_models
%% assessing alternative models

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

fit_options_ang = optimoptions('fmincon',...
    'MaxFunctionEvaluations',1e6,...
    'MaxIterations',1e6,...
    'FunctionTolerance',1e-10,...
    'StepTolerance',1e-10,...
    'OptimalityTolerance',1e-10,...
    'Display','none',...
    'HessianApproximation',{"lbfgs",20},...
    'SpecifyObjectiveGradient',false,...
    'UseParallel',true);

fit_options_rdm = optimoptions('fmincon',...
    'MaxFunctionEvaluations',Inf,...
    'MaxIterations',1e6,...
    'FunctionTolerance',1e-4,...
    'StepTolerance',1e-4,...
    'OptimalityTolerance',1e-4,...
    'Display','iter-detailed',...
    'HessianApproximation',{"lbfgs",20},...
    'SpecifyObjectiveGradient',false,...
    'UseParallel',true);

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

cv_cond = 100.*beh_all.attnCue + beh_all.sid;
k = 10; % how many folds
c1b = cvpartition(cv_cond,"KFold",k);

%% fit angular distance model
fprintf('--- Fitting angular distance model ---\n');

for k1=k:-1:1
    % subset training and testing data
    train_idx = c1b.training(k1);
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

    % fit angular distance model to training data
    ang_nll = @(p) pred_triad_angdist(beh_train,p);
    fit_vals_nocond = fmincon(ang_nll,ones(1,n),[],[],[],[],zeros(1,n),[],[],fit_options_ang);
    fit_vals_cond = fmincon(ang_nll,ones(3,n),[],[],[],[],zeros(3,n),[],[],fit_options_ang);

    % test model on held-out data
    cv_ang_nll_nocond(k1) = pred_triad_angdist(beh_test,fit_vals_nocond);
    cv_ang_nll_cond(k1) = pred_triad_angdist(beh_test,fit_vals_cond);

    fprintf('Finished fold %d/%d\n',k-k1+1,k);
end

%% fit RDM model
fprintf('--- Fitting RDM model ---\n');

lam = [0 10.^linspace(-3,3,20)];

cv2_vec = combvec(1:k,lam,[true false]);
cv2_nll_test = nan(1,length(cv2_vec));
cv2_nll_train = nan(1,length(cv2_vec));
cv2_train_exit = nan(1,length(cv2_vec));
cv2_train_iter = nan(1,length(cv2_vec));
cv2_train_func = nan(1,length(cv2_vec));
cv2_train_opti = nan(1,length(cv2_vec));

% set up parfor loop tracker
q = parallel.pool.DataQueue;
afterEach(q,@parforTracker);
parforTracker(size(cv2_vec,2));

parOpts = parforOptions(pool,"RangePartitionMethod","fixed","SubrangeSize",1);
parfor (k2=1:size(cv2_vec,2),parOpts)
    test_set = cv2_vec(1,k2);
    this_lam = cv2_vec(2,k2);
    fit_cond = cv2_vec(3,k2);

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
        beh_train(ii).sid = ii.*ones(1,trainTrialsPerSub(ii));

        sub_test = ~train_idx' & beh_all.sid==ii;
        beh_test(ii).triadRefA = beh_all.triadRefA(sub_test);
        beh_test(ii).triadChoiceA = beh_all.triadChoiceA(sub_test,:);
        beh_test(ii).triadChosenLoc = beh_all.triadChosenLoc(sub_test);
        testTrialsPerSub(ii) = length(beh_test(ii).triadRefA);
        beh_test(ii).sid = ii.*ones(1,testTrialsPerSub(ii));
    end

    % get starting coordinates
    seedMat = zeros(36,36);
    combStim = combvec(1:36,1:36);
    for ii=1:length(combStim)
    	seedMat(ii) = 1-sum(ismember(beh_all.triadRef(train_idx),combStim(:,ii)) & ismember(beh_all.triadChosen(train_idx),combStim(:,ii))) ./ ...
            sum(ismember(beh_all.triadRef(train_idx),combStim(:,ii)) & any(ismember(beh_all.triadChoice(train_idx,:),combStim(:,ii))'));
    end
    seedMat(isnan(seedMat)) = 0;

    if fit_cond
        init_p = repmat(squareform(seedMat)',1,n,3);
    else
        init_p = repmat(squareform(seedMat)',1,n);
    end

    % save time by feeding input to the model
    input = struct();
    all_triadRef = [beh_train.triadRefA];
    all_triadChoice = vertcat(beh_train.triadChoiceA);
    input.all_triadChosenLoc = [beh_train.triadChosenLoc];
    input.all_sid = [beh_train.sid];

    input.iL = sub2ind([108 108 n],all_triadRef,all_triadChoice(:,1)',input.all_sid);
    input.iR = sub2ind([108 108 n],all_triadRef,all_triadChoice(:,2)',input.all_sid);
    nTrials = length(all_triadRef);
    input.idxMat = sparse(input.iR,1:nTrials,1,108*108*n,nTrials) - ...
        sparse(input.iL,1:nTrials,1,108*108*n,nTrials);

    % fit angular distance model to training data
    rdm_nll = @(p) pred_triad_distmat(beh_train,p,this_lam,input,true);
    [fit_vals,fval,exitflag,output] = fmincon(rdm_nll,init_p,[],[],[],[],zeros(size(init_p)),8.2.*ones(size(init_p)),[],fit_options_rdm);

    % test model on held-out data
    cv2_nll_test(k2) = pred_triad_distmat(beh_test,fit_vals,0,[],false);
    cv2_nll_train(k2) = fval;
    cv2_train_exit(k2) = exitflag;
    cv2_train_iter(k2) = output.iterations;
    cv2_train_func(k2) = output.funcCount;
    cv2_train_opti(k2) = output.firstorderopt;

    send(q,[]);
end

cv2_nll_test = reshape(cv2_nll_test,k,[],2);

cv_nll_b.test_nll_ang_cond = cv_ang_nll_cond;
cv_nll_b.test_nll_ang_nocond = cv_ang_nll_nocond;
cv_nll_b.test_nll_rdm_cond = cv2_nll_test(:,:,1);
cv_nll_b.test_nll_rdm_nocond = cv2_nll_test(:,:,2);
cv_nll_b.lam_rdm = lam;
cv_nll_b.train_nll_rdm = reshape(cv2_nll_train,k,[],2);
cv_nll_b.train_exit_rdm = reshape(cv2_train_exit,k,[],2);
cv_nll_b.train_iter_rdm = reshape(cv2_train_iter,k,[],2);
cv_nll_b.train_func_rdm = reshape(cv2_train_func,k,[],2);
cv_nll_b.train_opti_rdm = reshape(cv2_train_opti,k,[],2);

save('data/fit_alt_models.mat','cv_nll_b');

delete(pool);
