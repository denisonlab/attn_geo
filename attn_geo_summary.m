%% Data analysis for attn_geo

% load component files
load('data/fit_alt_models.mat','cv_nll_b');
load('data/fit_geo_models.mat','cond','nocond');
load('data/triad_data_collated.mat','beh');

%% Analyze attention task performance

attn_beh.targ = nan(4,4,3,10);
attn_beh.dist = nan(4,4,3,10);
for subno=1:10
    attn_beh.targ(:,:,:,subno) = crosstab(beh(subno).attnResponse,beh(subno).nTargTilt,beh(subno).attnCue);
    attn_beh.dist(:,:,:,subno) = crosstab(beh(subno).attnResponse,beh(subno).nDistTilt,beh(subno).attnCue);
end
attn_beh.targ(:,:,3,:) = [];
attn_beh.dist(:,:,3,:) = [];

attn_beh.targ = attn_beh.targ ./ sum(sum(attn_beh.targ,1),2);
attn_beh.dist = attn_beh.dist ./ sum(sum(attn_beh.dist,1),2);

% response relative to target tilt
figure
for subno=1:10
  for jj=1:2
    attn_beh.cor_targ(jj,subno) = corr(beh(subno).nTargTilt(beh(subno).attnCue==jj)',...
             beh(subno).attnResponse(beh(subno).attnCue==jj)');
    subplot(2,10,subno+(jj-1)*10)
    imagesc(0:3,0:3,attn_beh.targ(:,:,jj,subno))
    axis square
    clim([0 .25])
    title(sprintf('Subject %d, r = %.03f',subno, attn_beh.cor_targ(jj,subno)));
  end
  attn_beh.cor_targ(3,subno) = corr(beh(subno).nTargTilt(beh(subno).attnCue<3)',...
      beh(subno).attnResponse(beh(subno).attnCue<3)');
end

% response relative to distractor tilt
figure
for subno=1:10
  for jj=1:2
    attn_beh.cor_dist(jj,subno) = corr(beh(subno).nDistTilt(beh(subno).attnCue==jj)',...
             beh(subno).attnResponse(beh(subno).attnCue==jj)');
    subplot(2,10,subno+(jj-1)*10)
    imagesc(0:3,0:3,attn_beh.dist(:,:,jj,subno))
    axis square
    clim([0 .25])
    title(sprintf('Subject %d, r = %.03f',subno, attn_beh.cor_dist(jj,subno)));
  end
  attn_beh.cor_dist(3,subno) = corr(beh(subno).nDistTilt(beh(subno).attnCue<3)',...
      beh(subno).attnResponse(beh(subno).attnCue<3)');
end

% averaged across conditions and subjects
figure
subplot(121)
imagesc(0:3,0:3,mean(mean(attn_beh.targ,3),4))
axis square
clim([0 .2])
title(sprintf('Target-centered, r = %.03f',mean(attn_beh.cor_targ(3,:),2)))

subplot(122)
imagesc(0:3,0:3,mean(mean(attn_beh.dist,3),4))
axis square
clim([0 .2])
title(sprintf('Distractor-centered, r = %.03f',mean(attn_beh.cor_dist(3,:),2)))

% stats
attn_beh.acc = squeeze(sum(sum(mean(attn_beh.targ,3).*eye(4),1),2)); % accuracy
[~,attn_beh.p,~,attn_beh.stats] = ttest(attn_beh.acc-.25);

sum(mean(mean(attn_beh.targ,4),3),1)*(0:3)' % mean tilts reported
sum(mean(mean(attn_beh.targ,4),3),2)'*(0:3)' % true mean tilts
sum(mean(mean(attn_beh.targ,4),3),1) % proportion of each response

%% Refit geometric models with best ridge parameters
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

% concatenate full data
n = length(beh);
beh_all.attnCue = [beh.attnCue];
beh_all.triadChoice = vertcat(beh.triadChoice);
beh_all.triadRef = [beh.triadRef];
beh_all.triadChosen = [beh.triadChosen];
beh_all.triadChosenLoc = [beh.triadChosenLoc];
beh_all.attnCue = [beh.attnCue];
beh_all.triadRefA = beh_all.triadRef + (beh_all.attnCue-1).*36;
beh_all.triadChoiceA = beh_all.triadChoice + (beh_all.attnCue'-1).*36;

for ii=1:n
    beh(ii).triadRefA = beh(ii).triadRef + (beh(ii).attnCue-1).*36;
    beh(ii).triadChoiceA = beh(ii).triadChoice + (beh(ii).attnCue'-1).*36;
end

% get starting coordinates
seedMat = zeros(36,36);
combStim = combvec(1:36,1:36);
for ii=1:length(combStim)
	seedMat(ii) = 1-sum(ismember(beh_all.triadRef,combStim(:,ii)) & ismember(beh_all.triadChosen,combStim(:,ii))) ./ ...
        sum(ismember(beh_all.triadRef,combStim(:,ii)) & any(ismember(beh_all.triadChoice,combStim(:,ii))'));
end
seedMat(isnan(seedMat)) = 0;

for mm=10:-1:1
    % MDS on RDM
    init_coords = mdscale(seedMat,mm,'Start','random','Replicates',10,'Options',statset('MaxIter',1000));

    % get best fitting parameters
    [nocond.min_nll(ii),nocond.idx_nll(ii)] = min(sum(nocond.test_nll(ii,:,:,:),2),[],"all");
    [nocond.idx_S(ii),nocond.idx_F(ii)] = ind2sub([22 21],nocond.idx_nll(ii));
    nocond.min_nll_sub(ii,:) = squeeze(sum(nocond.test_nll_sub(ii,:,nocond.idx_S(ii),nocond.idx_F(ii),:),2));
    nocond.max_acc_sub(ii,:) = squeeze(mean(nocond.test_acc_sub(ii,:,nocond.idx_S(ii),nocond.idx_F(ii),:),2));

    [cond.min_nll(ii),cond.idx_nll(ii)] = min(sum(cond.test_nll(ii,:,:,:),2),[],"all");
    [cond.idx_S(ii),cond.idx_F(ii)] = ind2sub([22 21],cond.idx_nll(ii));
    cond.min_nll_sub(ii,:) = squeeze(sum(cond.test_nll_sub(ii,:,cond.idx_S(ii),cond.idx_F(ii),:),2));

    % fit model to data
    % m1 = averaged across participants
    % m2 = separate participant spaces
    m1_negLL = @(p) pred_triad_resp(p,beh,Inf,Inf,0,0,true);
    m1_fit = fminunc(m1_negLL,init_coords,fit_options);

    m2_negLL = @(p) pred_triad_resp(p,beh,Inf,0,0,0,true);
    m2_fit = fminunc(m2_negLL,repmat(init_coords,[1 1 1 n]),fit_options);

    fit_coords = 0.5 .* m1_fit + 0.5 .* m2_fit; % blend fits

    % fit no condition model
    m1_negLL = @(p) pred_triad_resp(p,beh,Inf,nocond.lam_S(nocond.idx_S(mm)),...
        0,nocond.lam_Fs(nocond.idx_F(mm)),true);
    m1_fit = fminunc(m1_negLL,repmat(init_coords,[1 1 1 n]),fit_options);

    % fit attention model
    m2_negLL = @(p) pred_triad_resp(p,beh,0,cond.lam_S(cond.idx_S(mm)),...
        0,cond.lam_Fs(cond.idx_F(mm)),true);
    m2_fit = fminunc(m2_negLL,repmat(init_coords,[1 1 3 n]),fit_options);

    % save fitted coordinates
    nocond.fit(:,1:mm,:,mm) = reshape(m1_fit,36,mm,n);
    cond.fit(:,1:mm,:,:,mm) = m2_fit;

    fprintf('Finished dim %d\n',mm);
end

%% No condition modelling

% negative log-likelihood plot
figure, hold on
plot(1:10,nocond.min_nll,'k-')
yline(sum(cv_nll_b.test_nll_ang_nocond),'r--')
yline(min(sum(cv_nll_b.test_nll_rdm_nocond,1)),'g--')

% plot 4-D coordinates
figure
for mm=1:4
    subplot(1,4,mm), hold on
    plot(0:5:175,squeeze(nocond.fit(:,mm,:,4)))
    plot(0:5:175,mean(nocond.fit(:,mm,:,4),3),'k-','LineWidth',2)
    axis([0 180 -2.5 2.5])
end

% plot dimensions against each other
figure
subplot(121), hold on
plot(squeeze(nocond.fit(:,1,:,4)),squeeze(nocond.fit(:,2,:,4)),'-')
plot(mean(nocond.fit(:,1,:,4),3),mean(nocond.fit(:,2,:,4),3),'k-','LineWidth',2)
axis([-2.5 2.5 -2.5 2.5])
axis square

subplot(122), hold on
plot(squeeze(nocond.fit(:,3,:,4)),squeeze(nocond.fit(:,4,:,4)),'-')
plot(mean(nocond.fit(:,3,:,4),3),mean(nocond.fit(:,4,:,4),3),'k-','LineWidth',2)
axis([-1.25 1.25 -1.25 1.25])
axis square

% plot distance from origin
figure, hold on
plot(0:5:175,squeeze(vecnorm(nocond.fit(:,:,:,4),2,2)))
plot(0:5:175,mean(vecnorm(nocond.fit(:,:,:,4),2,2),3),'k-','LineWidth',2)
axis([0 180 0 2.5])

%% Models with attention

% negative log-likelihood plot
figure, hold on
plot(1:10,cond.min_nll,'k-')
yline(sum(cv_nll_b.test_nll_ang_cond),'r--')
yline(min(sum(cv_nll_b.test_nll_rdm_cond,1)),'g--')

% plot 4-D coordinates (note: 135 deg = -45 deg)
figure
for mm=1:4
    subplot(1,4,mm)
    p = plot(0:5:175,squeeze(mean(cond.fit(:,mm,:,:,4),4)));
    p(1).Color = [1 .4 .4];
    p(2).Color = [.4 .4 1];
    p(3).Color = [.4 .4 .4];
    axis([0 180 -2.5 2.5])
end
legend({'Attend -45','Attend +45','No attention'})

% plot 3-D coordinate space
norm_coords = [cond.fit; cond.fit(1,:,:,:,:)]./mean(vecnorm(cond.fit,2,2),1);
figure
p = plot3(squeeze(mean(norm_coords(:,1,:,:,3),4)), ...
    squeeze(mean(norm_coords(:,2,:,:,3),4)), ...
    squeeze(mean(norm_coords(:,3,:,:,3),4)),'.-',...
    'MarkerSize',10,'LineWidth',2);
p(1).Color = [1 .4 .4];
p(2).Color = [.4 .4 1];
p(3).Color = [.8 .8 .8];
axis([-1.25 1.25 -1.25 1.25 -1.25 1.25])
axis square
view(-30,15)

%% Calculate local lengths
length.total = squeeze(sum(vecnorm(diff([cond.fit; cond.fit(1,:,:,:,:)],[],1),2,2),1));

test_pos = [33:36 1:5; 6:14; 15:23; 24:32];
for kk=4:-1:1
    length.cond_bin(:,:,:,kk) = squeeze(sum(vecnorm(diff(cond.fit(test_pos(kk,:),:,:,:,:),[],1),2,2),1));
end

length.cond_bin_norm = length.cond_bin ./ length.total;

figure
for kk=1:4
  for mm=1:10
    subplot(4,10,(kk-1)*10+mm), hold on
    plot(1:3,length.cond_bin_norm(:,:,mm,kk))
    plot(1:3,mean(length.cond_bin_norm(:,:,mm,kk),2),'k-','LineWidth',2)
    axis([0.5 3.5 0 0.4])
    if kk==1, title(sprintf('%d-D',mm)), end
  end
end

% 4-D lengths
figure
for kk=1:4
    subplot(1,4,kk), hold on
    plot(1:3,length.cond_bin_norm(:,:,4,kk))
    plot(1:3,mean(length.cond_bin_norm(:,:,4,kk),2),'k-','LineWidth',2)
    axis([0.5 3.5 0 0.4])
    title(sprintf('Bin: %ddeg',(kk-1)*45))
end
