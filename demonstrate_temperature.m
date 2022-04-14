clc;
close all;

% Demonstrate the effects of temperature
% On random logit selection
% Also called "diversity"

probs = [0.1, 0.3, 0.6];

p1 = apply_temperature(probs, 0.5);
p2 = apply_temperature(probs, 1.0);
p3 = apply_temperature(probs, 1.5);

% Plot side by side
bar([p2; p1; p3].')
title("Effect of Temperature on Probabilities")
legend("T = 1.0", "T = 0.5", "T = 1.5", 'Location', 'northwest')

function p = apply_temperature(probs, T)
    p = log(probs) / T;
    p = exp(p) / sum(exp(p));  % Softmax
end