function parforTracker(iterations,loops)

persistent count N k

if nargin<2
    loops = [];
end

if ~isempty(iterations)
    count = 0;
    N = iterations;
    k = loops;
    if isempty(loops)
        fprintf('Tracker initiated...\n');
    else
        fprintf('Job %02d, tracker initiated...\n',k)
    end
else
    % update count
    count = count+1;
    if isempty(k)
        fprintf('Completed iteration: %d/%d\n',count,N);
    else
        fprintf('Job %02d, iteration: %d/%d\n',k,count,N);
    end
end
